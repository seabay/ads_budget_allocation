import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQL

# 1. 读取原始数据
df = pd.read_csv("offline_ads_data.csv")

# 平台/地区离散编码
platform_ids = df['platform'].astype('category').cat.codes
geo_ids = df['geo'].astype('category').cat.codes
df['platform_id'] = platform_ids
df['geo_id'] = geo_ids

# 2. 数值特征（StandardScaler）
numeric_features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend']
numeric_data = df[numeric_features].values.astype(np.float32)
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_data)

# 3. One-hot 编码平台和 geo
platform_onehot = pd.get_dummies(df['platform_id'], prefix='platform')
geo_onehot = pd.get_dummies(df['geo_id'], prefix='geo')

# 4. 构建最终 state 向量
state_df = pd.DataFrame(scaled_numeric, columns=numeric_features)
state_df = pd.concat([state_df, platform_onehot, geo_onehot], axis=1)
states = state_df.values.astype(np.float32)

# 5. 动作（预算）归一化
actions = df['budget'].values.reshape(-1, 1).astype(np.float32)
actions = actions / np.max(actions)

# 6. 奖励（ROI）
rewards = df['roi'].values.astype(np.float32)

# 7. 每条数据看作单步 episode，done=True
terminals = np.ones_like(rewards).astype(bool)

# 8. 构造 MDPDataset
dataset = MDPDataset(
    observations=states,
    actions=actions,
    rewards=rewards,
    terminals=terminals
)

# 9. 训练 CQL
cql = CQL(use_gpu=True)
cql.fit(dataset, n_steps=100_000)

# 10. 保存模型
cql.save_model("cql_ads_budget_model.pt")



=====================================================

# 🧪 使用训练好的策略


# 构造一个测试样本
test_row = {
    'roi_p10': 1.2,
    'roi_p50': 1.5,
    'roi_p90': 1.8,
    'cvr': 0.03,
    'ctr': 0.12,
    'prev_spend': 10000,
    'platform_id': 0,  # 例如 Facebook
    'geo_id': 5        # 例如 UK
}

# 数值特征标准化
numeric_part = np.array([[test_row[f] for f in numeric_features]], dtype=np.float32)
numeric_scaled = scaler.transform(numeric_part)

# one-hot 编码
platform_onehot = np.zeros((1, platform_onehot.shape[1]), dtype=np.float32)
platform_onehot[0, test_row['platform_id']] = 1.0

geo_onehot = np.zeros((1, geo_onehot.shape[1]), dtype=np.float32)
geo_onehot[0, test_row['geo_id']] = 1.0

# 拼接成完整 state
state = np.concatenate([numeric_scaled, platform_onehot, geo_onehot], axis=1)

# 预测预算（归一化值）
pred_action = cql.predict(state)[0][0]
print("Predicted normalized budget:", pred_action)


