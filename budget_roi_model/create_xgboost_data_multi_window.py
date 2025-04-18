

# 支持多窗口大小：我们将 window_size 参数更改为支持多个窗口大小的列表 window_sizes，例如 [7, 14]。

# 计算多窗口特征：对于每个窗口大小，我们都会计算该窗口的统计特征（均值、标准差、最大值等），
# 并将这些特征以动态命名的方式加入到特征中，如 mean_spend_7 和 mean_spend_14 分别表示窗口大小为 7 和 14 的平均消费。

# 滞后特征：对于每个窗口，我们还计算滞后特征（例如，过去1天、2天、3天的花费和 ROI），这些特征同样按窗口大小组织。

# 衰减因子的计算：通过 get_decay_weight 函数，为每个窗口中的数据引入了衰减因子。
# 可以根据需要调整衰减因子（decay_factor），默认是 0.9，表示每向后一步，权重减少 10%。
# 加权特征：在为每个窗口计算统计特征时，我们不再直接使用原始的 spend、ctr、cvr 和 roi，而是将这些特征值与衰减因子相乘，从而实现加权。
# 增强的窗口特征：每个窗口的加权统计特征被命名为 weighted_spend_{window_size}、weighted_ctr_{window_size}、weighted_cvr_{window_size} 
# 和 weighted_roi_{window_size}，表示该窗口特征经过衰减加权后的值



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 时间衰减因子（指数衰减）
def get_decay_weight(index, decay_factor=0.9):
    return decay_factor ** index  # 使用指数衰减

def construct_xgb_dataset(df, window_sizes=[7, 14], horizon=1, decay_factor=0.9):
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

        for i in range(max(window_sizes), len(df_seg) - horizon + 1):  # 最小窗口大小作为循环起始点
            target = df_seg.iloc[i + horizon - 1]

            # 聚合所有窗口大小的特征
            features = {
                # 静态特征（包括 One-Hot 编码的 segment_id）
                **{col: target[col] for col in segment_encoded_df.columns},
                'dayofweek': target['dayofweek'],
                'month': target['month'],
                'is_weekend': target['is_weekend'],
                'target_roi': target['roi']
            }

            # 为每个窗口大小计算特征
            for window_size in window_sizes:
                window = df_seg.iloc[i - window_size:i]
                
                # 应用时间衰减因子
                weighted_spend = sum(window['spend'].iloc[j] * get_decay_weight(j, decay_factor) for j in range(window_size))
                weighted_ctr = sum(window['ctr'].iloc[j] * get_decay_weight(j, decay_factor) for j in range(window_size))
                weighted_cvr = sum(window['cvr'].iloc[j] * get_decay_weight(j, decay_factor) for j in range(window_size))
                weighted_roi = sum(window['roi'].iloc[j] * get_decay_weight(j, decay_factor) for j in range(window_size))

                # 更新特征字典
                features[f'weighted_spend_{window_size}'] = weighted_spend
                features[f'weighted_ctr_{window_size}'] = weighted_ctr
                features[f'weighted_cvr_{window_size}'] = weighted_cvr
                features[f'weighted_roi_{window_size}'] = weighted_roi

                # 滞后特征（过去 1～3 天，适用于所有窗口）
                if i - 1 >= 0:
                    features[f'spend_lag1_{window_size}'] = df_seg.iloc[i - 1]['spend']
                    features[f'roi_lag1_{window_size}'] = df_seg.iloc[i - 1]['roi']
                    features[f'ctr_lag1_{window_size}'] = df_seg.iloc[i - 1]['ctr']
                    features[f'cvr_lag1_{window_size}'] = df_seg.iloc[i - 1]['cvr']

                if i - 2 >= 0:
                    features[f'spend_lag2_{window_size}'] = df_seg.iloc[i - 2]['spend']
                    features[f'roi_lag2_{window_size}'] = df_seg.iloc[i - 2]['roi']

                if i - 3 >= 0:
                    features[f'spend_lag3_{window_size}'] = df_seg.iloc[i - 3]['spend']
                    features[f'roi_lag3_{window_size}'] = df_seg.iloc[i - 3]['roi']

            all_features.append(features)

    return pd.DataFrame(all_features)


# =============================================

df = pd.read_csv("your_data.csv")
window_sizes = [7, 14]  # 选择多个窗口大小
dataset = construct_xgb_dataset(df, window_sizes=window_sizes, horizon=1, decay_factor=0.9)

# 训练数据拆分
X = dataset.drop(columns=['target_roi'])
y = dataset['target_roi']

from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, max_depth=6)
model.fit(X, y)

# 预测
y_pred = model.predict(X)


