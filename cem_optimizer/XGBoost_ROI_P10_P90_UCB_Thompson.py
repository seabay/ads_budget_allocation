

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------------------------
# Quantile Regression Model for ROI
# ---------------------------
class QuantileROIModel:
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.models = {}
        self.quantiles = quantiles
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        for q in self.quantiles:
            model = xgb.XGBRegressor(objective='reg:quantile', alpha=q, n_estimators=100, max_depth=3, learning_rate=0.1)
            model.fit(X_scaled, y)
            self.models[q] = model

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = {}
        for q, model in self.models.items():
            preds[f'p{int(q*100)}'] = model.predict(X_scaled)
        return pd.DataFrame(preds)

    def predict_tensor(self, X):
        df_preds = self.predict(X)
        return torch.tensor(df_preds.values, dtype=torch.float32)


# ---------------------------
# Example UCB/TS Action Selection using Quantile ROI Model
# ---------------------------
def ucb_scores(p10, p50, p90, alpha=0.5):
    return p50 + alpha * (p90 - p10)

def thompson_sampling_scores(p10, p50, p90):
    samples = np.random.normal(loc=p50, scale=(p90 - p10) / 2)
    return samples

# ---------------------------

class QuantileAllocator:
    def __init__(self, quantile_model, scaler, exploration="ucb", alpha=0.3):
        self.model = quantile_model
        self.scaler = scaler
        self.exploration = exploration  # 'ucb' or 'thompson'
        self.alpha = alpha

    def predict_roi_quantiles(self, segment_df):
        features = [
            'platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend',
            'cumulative_spend', 'time_index'
        ]
        X = segment_df[features].astype(float)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        segment_df['roi_p10'] = preds[0.1]
        segment_df['roi_p50'] = preds[0.5]
        segment_df['roi_p90'] = preds[0.9]
        return segment_df

    def compute_scores(self, df):
        if self.exploration == 'ucb':
            df['score'] = df['roi_p50'] + self.alpha * (df['roi_p90'] - df['roi_p10'])
        elif self.exploration == 'thompson':
            sigma = (df['roi_p90'] - df['roi_p10']) / 2
            sampled = np.random.normal(df['roi_p50'], sigma)
            df['score'] = sampled
        else:
            df['score'] = df['roi_p50']  # default baseline
        return df

    def allocate_budget(self, df, total_budget):
        df['allocated_budget'] = (df['score'] / df['score'].sum()) * total_budget
        return df

    def run(self, segment_df, total_budget):
        df = self.predict_roi_quantiles(segment_df.copy())
        df = self.compute_scores(df)
        df = self.allocate_budget(df, total_budget)
        return df


# ---------------------------


class CEMAllocator:
    def __init__(self, roi_predictor, scaler, n_samples=1000, elite_frac=0.1, n_iter=10):
        self.roi_predictor = roi_predictor  # quantile model or any ROI simulator
        self.scaler = scaler
        self.n_samples = n_samples
        self.elite_frac = elite_frac
        self.n_iter = n_iter

    def evaluate_allocation(self, df, weights):
        df = df.copy()
        df['sim_allocated'] = weights * df['max_budget']  # optional constraint
        features = [
            'platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend',
            'cumulative_spend', 'time_index'
        ]
        X = df[features].astype(float)
        X_scaled = self.scaler.transform(X)
        roi_preds = self.roi_predictor.predict(X_scaled)[0.5]  # use P50 or mean ROI
        total_roi = np.sum(roi_preds * df['sim_allocated'])
        return total_roi

    def run(self, segment_df, total_budget):
        n_segments = len(segment_df)
        mu = np.ones(n_segments) / n_segments
        sigma = 0.1 * np.ones(n_segments)

        for _ in range(self.n_iter):
            samples = np.random.normal(loc=mu, scale=sigma, size=(self.n_samples, n_segments))
            samples = np.abs(samples)
            samples = samples / samples.sum(axis=1, keepdims=True)

            rewards = np.array([
                self.evaluate_allocation(segment_df, w) for w in samples
            ])

            elite_idx = rewards.argsort()[-int(self.n_samples * self.elite_frac):]
            elite_samples = samples[elite_idx]

            mu = elite_samples.mean(axis=0)
            sigma = elite_samples.std(axis=0)

        final_weights = mu / mu.sum()
        segment_df['allocated_budget'] = final_weights * total_budget
        return segment_df


# ---------------------------
# Example Integration
# ---------------------------
if __name__ == '__main__':
    # Example synthetic data
    df = pd.DataFrame({
        'ctr': np.random.rand(1000),
        'cvr': np.random.rand(1000),
        'prev_spend': np.random.rand(1000) * 1000,
        'platform_id': np.random.randint(0, 3, 1000),
        'geo_id': np.random.randint(0, 15, 1000),
        'roi': np.random.rand(1000) * 3
    })

    features = ['ctr', 'cvr', 'prev_spend', 'platform_id', 'geo_id']
    X = df[features]
    y = df['roi']

    model = QuantileROIModel()
    model.fit(X, y)

    roi_quantiles = model.predict(X)
    df = pd.concat([df, roi_quantiles], axis=1)

    # Example UCB allocation score
    df['score_ucb'] = ucb_scores(df['p10'], df['p50'], df['p90'])
    df['score_ts'] = thompson_sampling_scores(df['p10'], df['p50'], df['p90'])

    print(df[['score_ucb', 'score_ts']].head())

