

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def construct_xgb_dataset(df, window_size=7, horizon=1):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['segment_id', 'date'])

    # 使用 sklearn 的 OneHotEncoder 对 segment_id 进行编码
    encoder = OneHotEncoder(sparse=False)
    segment_encoded = encoder.fit_transform(df[['segment_id']])
    segment_encoded_df = pd.DataFrame(segment_encoded, columns=encoder.get_feature_names_out(['segment_id']))

    # 将 One-Hot 编码的列合并回原始数据
    df = pd.concat([df, segment_encoded_df], axis=1)

    # 时间特征
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    all_features = []

    for seg_id in df['segment_id'].unique():
        df_seg = df[df['segment_id'] == seg_id].reset_index(drop=True)

        for i in range(window_size, len(df_seg) - horizon + 1):
            window = df_seg.iloc[i - window_size:i]
            target = df_seg.iloc[i + horizon - 1]

            features = {
                # 静态特征（包括 One-Hot 编码的 segment_id）
                **{col: target[col] for col in segment_encoded_df.columns},
                'dayofweek': target['dayofweek'],
                'month': target['month'],
                'is_weekend': target['is_weekend'],

                # 统计特征（滑动窗口）
                'mean_spend': window['spend'].mean(),
                'std_spend': window['spend'].std(),
                'max_spend': window['spend'].max(),
                'mean_ctr': window['ctr'].mean(),
                'mean_cvr': window['cvr'].mean(),
                'mean_roi': window['roi'].mean(),

                # 滞后特征（过去 1～3 天）
                'spend_lag1': df_seg.iloc[i - 1]['spend'],
                'spend_lag2': df_seg.iloc[i - 2]['spend'],
                'spend_lag3': df_seg.iloc[i - 3]['spend'],
                'roi_lag1': df_seg.iloc[i - 1]['roi'],
                'roi_lag2': df_seg.iloc[i - 2]['roi'],
                'roi_lag3': df_seg.iloc[i - 3]['roi'],
                'ctr_lag1': df_seg.iloc[i - 1]['ctr'],
                'cvr_lag1': df_seg.iloc[i - 1]['cvr'],
            }

            # 目标变量
            features['target_roi'] = target['roi']
            all_features.append(features)

    return pd.DataFrame(all_features)



# ==============================================


dataset = construct_xgb_dataset(df, window_size=7, horizon=1)

# 训练数据拆分
X = dataset.drop(columns=['target_roi'])
y = dataset['target_roi']

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, max_depth=6)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
