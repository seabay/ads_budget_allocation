

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class ROITimeseriesDataset(Dataset):
    def __init__(self, df, window_size=7, horizon=3, feature_cols=None, target_col='roi'):
        self.window_size = window_size
        self.horizon = horizon
        self.feature_cols = feature_cols or ['spend', 'ctr', 'cvr']
        self.target_col = target_col

        # Construct future ROI as 7-day rolling sum of registers
        df['roi'] = (
            df.groupby('segment_id')['registers']
              .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
        )

        # Preprocess all segments
        self.samples = []
        segment_ids = df['segment_id'].unique()
        for seg_id in segment_ids:
            df_seg = df[df['segment_id'] == seg_id].sort_values('date')
            df_seg = self._encode_time_features(df_seg)
            for i in range(len(df_seg) - window_size - horizon):
                x = df_seg.iloc[i:i+window_size][self.feature_cols].values
                y = df_seg.iloc[i+window_size:i+window_size+horizon][self.target_col].values
                self.samples.append((x, y))

    def _encode_time_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek / 6.0
        df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
        df['month'] = df['date'].dt.month / 12.0
        time_features = ['dayofweek', 'weekofyear', 'month']
        for f in time_features:
            if f not in self.feature_cols:
                self.feature_cols.append(f)
        return df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMROIRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, horizon=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        return self.fc(final_hidden)


class CEMAllocator:
    def __init__(self, model, num_segments, horizon, pop_size=100, elite_frac=0.2, n_iters=10):
        self.model = model.eval()
        self.num_segments = num_segments
        self.horizon = horizon
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.n_iters = n_iters

    def optimize(self, recent_features):
        mu = np.ones(self.num_segments) / self.num_segments
        sigma = 0.1

        for _ in range(self.n_iters):
            samples = np.random.normal(loc=mu, scale=sigma, size=(self.pop_size, self.num_segments))
            samples = np.clip(samples, 0, 1)
            samples /= samples.sum(axis=1, keepdims=True)

            rewards = []
            for alloc in samples:
                roi_pred = self._simulate_roi(alloc, recent_features)
                rewards.append(roi_pred.mean())

            elite_idx = np.argsort(rewards)[-int(self.pop_size * self.elite_frac):]
            elite = samples[elite_idx]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0)

        return mu

    def _simulate_roi(self, allocation, recent_features):
        input_tensor = torch.tensor(recent_features, dtype=torch.float32)
        preds = self.model(input_tensor).detach().numpy()
        weighted_roi = (preds.mean(axis=1) * allocation).sum()
        return weighted_roi


class UCBAgent:
    def __init__(self, model, c=1.0):
        self.model = model.eval()
        self.c = c
        self.counts = None
        self.roi_means = None

    def select(self, recent_features):
        input_tensor = torch.tensor(recent_features, dtype=torch.float32)
        preds = self.model(input_tensor).detach().numpy()
        means = preds.mean(axis=1)
        stds = preds.std(axis=1)

        if self.counts is None:
            self.counts = np.ones_like(means)
            self.roi_means = means

        ucb = means + self.c * stds / np.sqrt(self.counts)
        allocation = ucb / ucb.sum()
        self.counts += 1
        self.roi_means = (self.roi_means * (self.counts - 1) + means) / self.counts
        return allocation


class ThompsonSamplingAgent:
    def __init__(self, model):
        self.model = model.eval()

    def select(self, recent_features):
        input_tensor = torch.tensor(recent_features, dtype=torch.float32)
        preds = self.model(input_tensor).detach().numpy()
        samples = np.random.normal(loc=preds.mean(axis=1), scale=preds.std(axis=1))
        samples = np.maximum(samples, 0)
        allocation = samples / samples.sum()
        return allocation


if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    window_size = 7
    horizon = 7
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = ROITimeseriesDataset(train_df, window_size=window_size, horizon=horizon)
    val_dataset = ROITimeseriesDataset(val_df, window_size=window_size, horizon=horizon, feature_cols=train_dataset.feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = len(train_dataset[0][0][0])
    model = LSTMROIRegressor(input_dim=input_dim, horizon=horizon)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = F.mse_loss(preds, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    recent_window = val_dataset[:train_dataset.samples[0][0].shape[0]]
    recent_features = np.stack([x.numpy() for x, _ in recent_window])

    cem = CEMAllocator(model, num_segments=recent_features.shape[0], horizon=horizon)
    best_alloc_cem = cem.optimize(recent_features)
    print("[CEM] Optimized allocation:", best_alloc_cem)

    ucb_agent = UCBAgent(model, c=1.0)
    best_alloc_ucb = ucb_agent.select(recent_features)
    print("[UCB] Optimized allocation:", best_alloc_ucb)

    ts_agent = ThompsonSamplingAgent(model)
    best_alloc_ts = ts_agent.select(recent_features)
    print("[Thompson Sampling] Optimized allocation:", best_alloc_ts)
