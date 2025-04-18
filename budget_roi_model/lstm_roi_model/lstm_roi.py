import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


class ROITimeseriesDataset(Dataset):
    def __init__(self, df, window_size=7, feature_cols=None, target_col='roi'):
        self.window_size = window_size
        self.feature_cols = feature_cols or ['spend', 'ctr', 'cvr']
        self.target_col = target_col

        # Preprocess all segments
        self.samples = []
        segment_ids = df['segment_id'].unique()
        for seg_id in segment_ids:
            df_seg = df[df['segment_id'] == seg_id].sort_values('date')
            df_seg = self._encode_time_features(df_seg)
            for i in range(len(df_seg) - window_size):
                x = df_seg.iloc[i:i+window_size][self.feature_cols].values
                y = df_seg.iloc[i+window_size][self.target_col]
                self.samples.append((x, y))

    def _encode_time_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek / 6.0
        df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
        df['month'] = df['date'].dt.month / 12.0
        time_features = ['dayofweek', 'weekofyear', 'month']
        self.feature_cols += time_features
        return df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMROIRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]  # use the last hidden state
        return self.fc(final_hidden).squeeze(-1)


# 完整训练逻辑
if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = ROITimeseriesDataset(train_df, window_size=7)
    val_dataset = ROITimeseriesDataset(val_df, window_size=7, feature_cols=train_dataset.feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = len(train_dataset[0][0][0])
    model = LSTMROIRegressor(input_dim=input_dim)
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


# ======================================================== 


import os
import argparse

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3, early_stop_patience=5, save_path="best_model.pt"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = F.mse_loss(preds, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="your_data.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = ROITimeseriesDataset(train_df, window_size=7)
    val_dataset = ROITimeseriesDataset(val_df, window_size=7, feature_cols=train_dataset.feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = len(train_dataset[0][0][0])
    model = LSTMROIRegressor(input_dim=input_dim)

    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        device=args.device,
        num_epochs=30,
        lr=1e-3,
        early_stop_patience=5,
        save_path="best_lstm_model.pt"
    )
