import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

# 基于 Thompson Sampling 的 _select_action_thompson 方法，并通过 use_thompson=True 参数启用。
# 该方法假设：
# 你使用了多头 Q 网络（如 nn.ModuleList）
# 或者模型中提供 sample_heads() 方法用于抽样

# 构建一个 ModuleList 多头网络模型作为 model

class MultiHeadQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=5, hidden_dim=64):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        return self.heads[0](x)  # default head

    def sample_heads(self):
        return random.choice(self.heads)

# model = MultiHeadQNetwork(input_dim=8, output_dim=1, num_heads=10)
# torch.save(model, "multihead_q_model.pt")


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
