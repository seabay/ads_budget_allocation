
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class QuarterBudgetAllocator:
    def __init__(self, model_path, scaler, alpha_ucb=0.5, use_ucb=True, use_thompson=False):
        self.model = torch.load(model_path)
        self.model.eval()
        self.scaler = scaler
        self.alpha_ucb = alpha_ucb
        self.use_ucb = use_ucb
        self.use_thompson = use_thompson

    def _prepare_state(self, df):
        features = [
            'roi_p10', 'roi_p50', 'roi_p90',
            'cvr', 'ctr', 'prev_spend',
            'platform_id', 'geo_id'
        ]
        X = df[features].copy()
        X[['platform_id', 'geo_id']] = X[['platform_id', 'geo_id']].astype(float)
        X_scaled = self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)

    def _select_action_ucb(self, state_tensor, roi_p90, roi_p10):
        with torch.no_grad():
            q_values = self.model(state_tensor)
            if self.use_ucb:
                ucb_bonus = self.alpha_ucb * (roi_p90 - roi_p10)
                ucb_bonus_tensor = torch.tensor(ucb_bonus, dtype=torch.float32).unsqueeze(-1)
                q_values = q_values + ucb_bonus_tensor
            actions = torch.argmax(q_values, dim=1).numpy()
        return actions

    def _select_action_thompson(self, state_tensor):
        with torch.no_grad():
            if isinstance(self.model, torch.nn.ModuleList):
                sampled_model = random.choice(self.model)
                q_values = sampled_model(state_tensor)
            elif hasattr(self.model, 'sample_heads'):
                sampled_model = self.model.sample_heads()
                q_values = sampled_model(state_tensor)
            else:
                q_values = self.model(state_tensor)
            actions = torch.argmax(q_values, dim=1).numpy()
        return actions

    def allocate(self, segment_df, total_budget, platform_max_ratio=None, geo_min_budget=None):
        df = segment_df.copy()
        state_tensor = self._prepare_state(df)

        if self.use_thompson:
            actions = self._select_action_thompson(state_tensor)
        else:
            actions = self._select_action_ucb(
                state_tensor,
                roi_p90=df['roi_p90'].values,
                roi_p10=df['roi_p10'].values
            )

        df['score'] = actions + 1
        df['allocated_budget'] = (df['score'] / df['score'].sum()) * total_budget

        if platform_max_ratio:
            for pid, max_ratio in platform_max_ratio.items():
                mask = df['platform_id'] == pid
                platform_budget = df.loc[mask, 'allocated_budget'].sum()
                max_budget = total_budget * max_ratio
                if platform_budget > max_budget:
                    scale = max_budget / platform_budget
                    df.loc[mask, 'allocated_budget'] *= scale

        if geo_min_budget:
            for gid, min_budget in geo_min_budget.items():
                mask = df['geo_id'] == gid
                current = df.loc[mask, 'allocated_budget'].sum()
                if current < min_budget:
                    scale = min_budget / current if current > 0 else 1.0
                    df.loc[mask, 'allocated_budget'] *= scale

        return df

    def simulate_reward(self, df, roi_column='roi'):
        return (df['allocated_budget'] * df[roi_column]).sum()


class RuleBasedAllocator:
    def __init__(self, roi_column='roi_p50'):
        self.roi_column = roi_column

    def allocate(self, segment_df, total_budget):
        df = segment_df.copy()
        df['score'] = df[self.roi_column]
        df['allocated_budget'] = (df['score'] / df['score'].sum()) * total_budget
        return df

    def simulate_reward(self, df, roi_column='roi'):
        return (df['allocated_budget'] * df[roi_column]).sum()


# --- TD target computation helper for training ---
def compute_td_target(r, done, s_next, gamma, target_net):
    with torch.no_grad():
        next_q_values = target_net(s_next)
        max_next_q = torch.max(next_q_values, dim=1, keepdim=True)[0]
        return r + gamma * (1 - done) * max_next_q

# --- Training loss computation ---
def compute_td_loss(q_net, target_net, s, a, r, s_next, done, gamma=0.99):
    q_values = q_net(s)
    q_value = q_values.gather(1, a)  # shape (batch, 1)
    td_target = compute_td_target(r, done, s_next, gamma, target_net)
    return F.mse_loss(q_value, td_target)


# --- Offline training loop ---
def train_q_network(
    q_net,
    target_net,
    dataset,
    scaler,
    num_epochs=100,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    target_update_freq=10,
    log_every=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    q_net.to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.to(device)
    target_net.eval()

    features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']
    X = scaler.transform(dataset[features].astype(float))
    s = torch.tensor(X, dtype=torch.float32)
    a = torch.tensor(dataset['action'].values, dtype=torch.int64).unsqueeze(1)
    r = torch.tensor(dataset['reward'].values, dtype=torch.float32).unsqueeze(1)
    done = torch.tensor(dataset['done'].values, dtype=torch.float32).unsqueeze(1)

    if 's_next' in dataset.columns:
        X_next = scaler.transform(dataset['s_next'].tolist())
        s_next = torch.tensor(X_next, dtype=torch.float32)
    else:
        s_next = s

    dataset_tensor = TensorDataset(s, a, r, s_next, done)
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        q_net.train()
        total_loss = 0
        for batch in loader:
            s_batch, a_batch, r_batch, s_next_batch, done_batch = [x.to(device) for x in batch]
            loss = compute_td_loss(q_net, target_net, s_batch, a_batch, r_batch, s_next_batch, done_batch, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epoch % log_every == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss / len(loader):.4f}")

    torch.save(q_net, "trained_q_model.pt")
    print("Training complete and model saved.")
