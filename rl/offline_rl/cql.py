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


# 模拟一个新 segment 的状态输入
sample_state = np.array([[1.2, 1.5, 1.8, 0.03, 0.12, 10000, 0, 5]])  # 注意平台和 geo 要映射
sample_state = scaler.transform(sample_state)

# 推理建议的 budget（归一化值）
predicted_budget = cql.predict(sample_state)[0][0]
print("Recommended budget (normalized):", predicted_budget)

