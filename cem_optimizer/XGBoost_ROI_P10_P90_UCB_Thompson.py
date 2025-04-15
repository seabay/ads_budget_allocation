

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

