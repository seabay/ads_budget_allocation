

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class XGBROIPredictor:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_cols = []

    def construct_features(self, df, window_size=7, horizon=7):
        df = df.copy()
        df['roi'] = (
            df.groupby('segment_id')['registers']
              .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
        )

        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek / 6.0
        df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
        df['month'] = df['date'].dt.month / 12.0

        all_samples = []
        for seg_id in df['segment_id'].unique():
            df_seg = df[df['segment_id'] == seg_id].sort_values('date')
            for i in range(len(df_seg) - window_size - horizon):
                x_slice = df_seg.iloc[i:i+window_size]
                y_value = df_seg.iloc[i+window_size]['roi']
                features = {
                    'segment_id': seg_id,
                    'spend_sum': x_slice['spend'].sum(),
                    'ctr_mean': x_slice['ctr'].mean(),
                    'cvr_mean': x_slice['cvr'].mean(),
                    'dayofweek_last': x_slice['dayofweek'].iloc[-1],
                    'weekofyear_last': x_slice['weekofyear'].iloc[-1],
                    'month_last': x_slice['month'].iloc[-1],
                    'target_roi': y_value
                }
                all_samples.append(features)
        return pd.DataFrame(all_samples)

    def train(self, df_features):
        X = df_features.drop(columns=['target_roi'])
        y = df_features['target_roi']

        self.encoder = OneHotEncoder(sparse_output=False)
        X_encoded = self.encoder.fit_transform(X[['segment_id']])
        X_rest = X.drop(columns=['segment_id']).values
        X_full = np.hstack([X_encoded, X_rest])

        dtrain = xgb.DMatrix(X_full, label=y)
        self.model = xgb.train({'objective': 'reg:squarederror'}, dtrain, num_boost_round=100)

    def predict(self, df_features):
        X = df_features.drop(columns=['target_roi'])
        X_encoded = self.encoder.transform(X[['segment_id']])
        X_rest = X.drop(columns=['segment_id']).values
        X_full = np.hstack([X_encoded, X_rest])
        dtest = xgb.DMatrix(X_full)
        return self.model.predict(dtest)


class CEMAllocator:
    def __init__(self, model, recent_features, segment_ids, pop_size=100, elite_frac=0.2, n_iters=10):
        self.model = model
        self.recent_features = recent_features
        self.segment_ids = segment_ids
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.n_iters = n_iters

    def optimize(self):
        num_segments = len(self.segment_ids)
        mu = np.ones(num_segments) / num_segments
        sigma = 0.1

        for _ in range(self.n_iters):
            samples = np.random.normal(loc=mu, scale=sigma, size=(self.pop_size, num_segments))
            samples = np.clip(samples, 0, 1)
            samples /= samples.sum(axis=1, keepdims=True)

            rewards = []
            for alloc in samples:
                reward = self._simulate_roi(alloc)
                rewards.append(reward)

            elite_idx = np.argsort(rewards)[-int(self.pop_size * self.elite_frac):]
            elite = samples[elite_idx]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0)

        return mu

    def _simulate_roi(self, allocation):
        weighted_roi = 0
        for i, seg_id in enumerate(self.segment_ids):
            df = self.recent_features[self.recent_features['segment_id'] == seg_id].copy()
            pred = self.model.predict(df).mean()
            weighted_roi += allocation[i] * pred
        return weighted_roi


if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    window_size = 7
    horizon = 7
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    predictor = XGBROIPredictor()
    train_features = predictor.construct_features(train_df, window_size, horizon)
    val_features = predictor.construct_features(val_df, window_size, horizon)

    predictor.train(train_features)

    recent_features = val_features[val_features['segment_id'].isin(val_df['segment_id'].unique())]
    segment_ids = sorted(recent_features['segment_id'].unique())

    cem = CEMAllocator(model=predictor, recent_features=recent_features, segment_ids=segment_ids)
    best_alloc = cem.optimize()
    print("[CEM] Optimized allocation:", best_alloc)


# ===================================================


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import torch

# Step 1: 构造 XGBoost 数据集（含 channel 和 geo 特征）
def construct_xgb_dataset(df, feature_cols, target_col, window_size=7, horizon=3):
    df = df.copy()
    df['roi'] = (
        df.groupby('segment_id')['registers']
          .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
    )

    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek / 6.0
    df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
    df['month'] = df['date'].dt.month / 12.0

    feature_cols += ['dayofweek', 'weekofyear', 'month']

    encoder = pd.get_dummies(df[['channel', 'geo']])
    df = pd.concat([df, encoder], axis=1)
    feature_cols += list(encoder.columns)

    features, targets = [], []
    for segment_id in df['segment_id'].unique():
        df_seg = df[df['segment_id'] == segment_id].sort_values('date')
        for i in range(len(df_seg) - window_size - horizon):
            x = df_seg.iloc[i:i+window_size][feature_cols].values.mean(axis=0)
            y = df_seg.iloc[i+window_size:i+window_size+horizon][target_col].mean()
            features.append(x)
            targets.append(y)

    return np.array(features), np.array(targets)


# Step 2: 训练 XGBoost 模型
def train_xgboost_model(df, feature_cols, target_col, window_size=7, horizon=3):
    features, targets = construct_xgb_dataset(df, feature_cols, target_col, window_size, horizon)
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_mse = np.mean((val_preds - y_val) ** 2)
    print(f"Validation MSE: {val_mse:.4f}")
    return model


# Step 3: CEM Allocator
class CEMAllocator:
    def __init__(self, model, num_segments, horizon, pop_size=100, elite_frac=0.2, n_iters=10):
        self.model = model
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
                rewards.append(roi_pred)

            elite_idx = np.argsort(rewards)[-int(self.pop_size * self.elite_frac):]
            elite = samples[elite_idx]
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0)

        return mu

    def _simulate_roi(self, allocation, recent_features):
        preds = self.model.predict(recent_features)
        weighted_roi = (preds * allocation).sum()
        return weighted_roi


# Step 4: UCB Agent
class UCBAgent:
    def __init__(self, model, c=1.0):
        self.model = model
        self.c = c
        self.counts = None
        self.roi_means = None

    def select(self, recent_features):
        preds = self.model.predict(recent_features)
        means = preds
        stds = np.std(preds)

        if self.counts is None:
            self.counts = np.ones_like(means)
            self.roi_means = means

        ucb = means + self.c * stds / np.sqrt(self.counts)
        allocation = ucb / ucb.sum()
        self.counts += 1
        self.roi_means = (self.roi_means * (self.counts - 1) + means) / self.counts
        return allocation


# Step 5: Thompson Sampling
class ThompsonSamplingAgent:
    def __init__(self, model):
        self.model = model

    def select(self, recent_features):
        preds = self.model.predict(recent_features)
        samples = np.random.normal(loc=preds, scale=np.std(preds))
        samples = np.maximum(samples, 0)
        allocation = samples / samples.sum()
        return allocation


# Step 6: 运行优化流程
def run_optimization(df, feature_cols, target_col, window_size=7, horizon=3):
    model = train_xgboost_model(df, feature_cols, target_col, window_size, horizon)

    segment_ids = df['segment_id'].unique()
    recent_features = []
    # recent_features 是通过对每个 segment_id 的最近 window_size 天的特征数据取平均得到
    for seg_id in segment_ids:
        df_seg = df[df['segment_id'] == seg_id].sort_values('date')
        if len(df_seg) >= window_size:
            x = df_seg.iloc[-window_size:][feature_cols].values.mean(axis=0)
            recent_features.append(x)
    recent_features = np.stack(recent_features)
    # (num_segments, feature_dim)
    # num_segments：等于 segment 的数量，例如 45（3 个 channel × 15 个 geo）
    # feature_dim：等于处理后的特征维度，包含：
    #     原始的 feature_cols（例如：spend, ctr, cvr）
    #     时间特征（dayofweek, weekofyear, month）
    #     one-hot 编码的 channel 和 geo 列


    cem = CEMAllocator(model, num_segments=recent_features.shape[0], horizon=horizon)
    best_alloc_cem = cem.optimize(recent_features)
    print("[CEM] Optimized allocation:", best_alloc_cem)

    ucb_agent = UCBAgent(model)
    best_alloc_ucb = ucb_agent.select(recent_features)
    print("[UCB] Optimized allocation:", best_alloc_ucb)

    ts_agent = ThompsonSamplingAgent(model)
    best_alloc_ts = ts_agent.select(recent_features)
    print("[Thompson Sampling] Optimized allocation:", best_alloc_ts)


if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    feature_cols = ['spend', 'ctr', 'cvr']
    target_col = 'roi'
    run_optimization(df, feature_cols, target_col, window_size=7, horizon=7)
