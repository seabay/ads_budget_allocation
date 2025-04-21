

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

    # 添加全体 segment 的 spend 总和（全局预算特征）
    df['total_spend_all_segments'] = df.groupby('date')['spend'].transform('sum')
    df['spend_ratio_to_total'] = df['spend'] / (df['total_spend_all_segments'] + 1e-5)

    # 可选：添加同 geo 内的其他 segment spend
    df['geo_total_spend'] = df.groupby(['geo', 'date'])['spend'].transform('sum')
    df['other_geo_spend'] = df['geo_total_spend'] - df['spend']

    encoder = pd.get_dummies(df[['channel', 'geo']], prefix=['ch', 'geo'])
    df = pd.concat([df, encoder], axis=1)
    feature_cols += list(encoder.columns) + [
        'dayofweek', 'weekofyear', 'month',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'spend_cumsum', 'spend_ctr', 'ctr_cvr',
        'total_spend_all_segments', 'spend_ratio_to_total',
        'geo_total_spend', 'other_geo_spend'
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

# 以下内容保持不变...
# ...（略） ...
