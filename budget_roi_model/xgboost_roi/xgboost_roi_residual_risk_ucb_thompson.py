
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch

# Step 1: 构造 XGBoost 数据集（含增强特征）
def construct_xgb_dataset(df, feature_cols, target_col, window_size=7, horizon=3):
    df = df.copy()

    # 缺失值填补
    df[feature_cols] = SimpleImputer(strategy="mean").fit_transform(df[feature_cols])

    # 异常值过滤（按分位数）
    for col in feature_cols:
        q_low, q_high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q_low, q_high)

    # 目标变量生成
    df[target_col] = (
        df.groupby('segment_id')['registers']
          .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
    )

    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek / 6.0
    df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
    df['month'] = df['date'].dt.month / 12.0

    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'])
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'])
    df['month_sin'] = np.sin(2 * np.pi * df['month'])
    df['month_cos'] = np.cos(2 * np.pi * df['month'])

    df['spend_cumsum'] = df.groupby('segment_id')['spend'].cumsum()
    df['spend_cumsum'] = df.groupby('segment_id')['spend_cumsum'].transform(lambda x: x / (x.max() + 1e-5))

    # 添加交叉特征
    df['spend_ctr'] = df['spend'] * df['ctr']
    df['ctr_cvr'] = df['ctr'] * df['cvr']

    encoder = pd.get_dummies(df[['channel', 'geo']], prefix=['ch', 'geo'])
    df = pd.concat([df, encoder], axis=1)
    feature_cols += list(encoder.columns) + [
        'dayofweek', 'weekofyear', 'month',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'spend_cumsum', 'spend_ctr', 'ctr_cvr'
    ]

    features, targets = [], []
    for segment_id in df['segment_id'].unique():
        df_seg = df[df['segment_id'] == segment_id].sort_values('date')
        arr = df_seg[feature_cols + [target_col]].values
        for i in range(len(df_seg) - window_size - horizon):
            window = arr[i:i+window_size, :-1]
            decay_weights = 0.95 ** np.arange(window_size)[::-1]
            decay_weights /= decay_weights.sum()
            weighted_mean = (window * decay_weights[:, None]).sum(axis=0)
            window_std = window.std(axis=0)
            window_diff = window[-1] - window[0]
            trend = window_diff / (window_size + 1e-5)
            x = np.concatenate([weighted_mean, window_std, trend])
            y = arr[i+window_size:i+window_size+horizon, -1].mean()
            features.append(x)
            targets.append(y)

    return np.array(features), np.array(targets)


# Step 2: 训练 XGBoost 模型，并估计残差标准差（segment级）
def train_xgboost_model(df, feature_cols, target_col, window_size=7, horizon=3):
    features, targets = construct_xgb_dataset(df, feature_cols, target_col, window_size, horizon)
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_mse = np.mean((val_preds - y_val) ** 2)
    print(f"Validation MSE: {val_mse:.4f}")

    residuals = y_train - model.predict(X_train)
    segment_ids = df['segment_id'].unique()
    residual_std_map = {i: np.std(residuals) for i in range(len(segment_ids))}
    return model, residual_std_map


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

    def _simulate_roi_residual(self, allocation, recent_features):
        preds = self.model.predict(recent_features)
        mean_roi = (preds * allocation).sum()
        risk = np.sqrt(((preds - preds.mean()) ** 2 * allocation).sum())
        score = mean_roi - self.risk_lambda * risk  # 风险调整目标函数
        return score



# Step 4: UCB Agent
class UCBAgent:
    def __init__(self, model, residual_std_map, c=1.0):
        self.model = model
        self.residual_std_map = residual_std_map
        self.c = c
        self.counts = None
        self.roi_means = None

    def select(self, recent_features):
        preds = self.model.predict(recent_features)
        stds = np.array([self.residual_std_map.get(i, 0.1) for i in range(len(preds))])

        if self.counts is None:
            self.counts = np.ones_like(preds)
            self.roi_means = preds

        ucb = preds + self.c * stds / np.sqrt(self.counts)
        allocation = ucb / ucb.sum()
        self.counts += 1
        self.roi_means = (self.roi_means * (self.counts - 1) + preds) / self.counts
        return allocation


# Step 5: Thompson Sampling
class ThompsonSamplingAgent:
    def __init__(self, model, residual_std_map):
        self.model = model
        self.residual_std_map = residual_std_map

    def select(self, recent_features):
        preds = self.model.predict(recent_features)
        stds = np.array([self.residual_std_map.get(i, 0.1) for i in range(len(preds))])
        samples = np.random.normal(loc=preds, scale=stds)
        samples = np.maximum(samples, 0)
        allocation = samples / samples.sum()
        return allocation

    def select_risk(self, recent_features):
        preds = self.model.predict(recent_features)
        samples = np.random.normal(loc=preds, scale=np.std(preds))
        samples = np.maximum(samples, 0)
        samples -= self.risk_lambda * np.std(preds)  # 风险调整
        allocation = samples / samples.sum()
        return allocation


# Step 6: 运行优化流程
def run_optimization(df, feature_cols, target_col, window_size=7, horizon=3):
    model, residual_std_map = train_xgboost_model(df, feature_cols, target_col, window_size, horizon)

    segment_ids = df['segment_id'].unique()
    recent_features = []
    for seg_id in segment_ids:
        df_seg = df[df['segment_id'] == seg_id].sort_values('date')
        if len(df_seg) >= window_size:
            x = df_seg.iloc[-window_size:][feature_cols].values.mean(axis=0)
            recent_features.append(x)
    recent_features = np.stack(recent_features)

    cem = CEMAllocator(model, num_segments=recent_features.shape[0], horizon=horizon)
    best_alloc_cem = cem.optimize(recent_features)
    print("[CEM] Optimized allocation:", best_alloc_cem)

    ucb_agent = UCBAgent(model, residual_std_map)
    best_alloc_ucb = ucb_agent.select(recent_features)
    print("[UCB] Optimized allocation:", best_alloc_ucb)

    ts_agent = ThompsonSamplingAgent(model, residual_std_map)
    best_alloc_ts = ts_agent.select(recent_features)
    print("[Thompson Sampling] Optimized allocation:", best_alloc_ts)


if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    feature_cols = ['spend', 'ctr', 'cvr']
    target_col = 'roi'
    run_optimization(df, feature_cols, target_col, window_size=7, horizon=7)
