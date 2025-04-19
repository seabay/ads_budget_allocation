

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch

# 边际效用函数
def diminishing_return(allocation, alpha=3.0):
    return 1 - np.exp(-alpha * allocation)

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

    features, targets, residuals = [], [], []
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
            y_seq = arr[i+window_size:i+window_size+horizon, -1]
            y = y_seq.mean()
            features.append(x)
            targets.append(y)
            residuals.append(y_seq.std())   #  这里不对，应该是：residuals = actual_roi - predicted_roi

    return np.array(features), np.array(targets), np.array(residuals)


# Step 2: 训练 XGBoost 模型
class XGBoostROIModel:
    def __init__(self):
        self.mean_model = None
        self.std_model = None

    def train(self, df, feature_cols, target_col, window_size=7, horizon=3):
        features, targets, residuals = construct_xgb_dataset(df, feature_cols, target_col, window_size, horizon)
        X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

        self.mean_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
        self.mean_model.fit(X_train, y_train)
        val_preds = self.mean_model.predict(X_val)
        val_mse = np.mean((val_preds - y_val) ** 2)
        print(f"[XGB] Validation MSE: {val_mse:.4f}")

        self.std_model = xgb.XGBRegressor(objective='reg:squarederror')
        self.std_model.fit(features, residuals)

    def predict(self, X):
        mean = self.mean_model.predict(X)
        std = self.std_model.predict(X)
        return mean, std


# Step 3: 策略优化器 with 风险 + 效用调节
class RiskAdjustedThompsonSampler:
    def __init__(self, roi_model, risk_aversion=1.0):
        self.roi_model = roi_model
        self.risk_aversion = risk_aversion

    def select(self, recent_features):
        mu, sigma = self.roi_model.predict(recent_features)
        samples = np.random.normal(mu, sigma)
        samples = np.maximum(samples, 0)
        utility = diminishing_return(samples)
        adjusted_utility = utility - self.risk_aversion * sigma
        adjusted_utility = np.maximum(adjusted_utility, 1e-6)
        alloc = adjusted_utility / adjusted_utility.sum()
        return alloc


# Step 4: 运行优化流程
def run_optimization(df, feature_cols, target_col, window_size=7, horizon=3):
    model = XGBoostROIModel()
    model.train(df, feature_cols, target_col, window_size, horizon)

    segment_ids = df['segment_id'].unique()
    recent_features = []
    for seg_id in segment_ids:
        df_seg = df[df['segment_id'] == seg_id].sort_values('date')
        if len(df_seg) >= window_size:
            x = df_seg.iloc[-window_size:][feature_cols].values.mean(axis=0)
            recent_features.append(x)
    recent_features = np.stack(recent_features)

    ts_agent = RiskAdjustedThompsonSampler(model, risk_aversion=0.3)
    best_alloc_ts = ts_agent.select(recent_features)
    print("[Thompson + Risk + Utility] Allocation:", best_alloc_ts)


if __name__ == '__main__':
    df = pd.read_csv("your_data.csv")
    feature_cols = ['spend', 'ctr', 'cvr']
    target_col = 'roi'
    run_optimization(df, feature_cols, target_col, window_size=7, horizon=7)
