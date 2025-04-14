import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset

class QuarterBudgetAllocator:
    def __init__(self, model_path, scaler, num_platforms=3, num_geos=15):
        self.model = CQL()
        self.model.build_with_dataset(self._dummy_dataset())
        self.model.load_model(model_path)
        self.scaler = scaler
        self.num_platforms = num_platforms
        self.num_geos = num_geos

    def _dummy_dataset(self):
        # 用于 model.build_with_dataset 的占位 dataset
        obs = np.random.rand(10, 10).astype(np.float32)
        act = np.random.rand(10, 1).astype(np.float32)
        rew = np.random.rand(10).astype(np.float32)
        done = np.ones(10).astype(bool)
        return MDPDataset(obs, act, rew, done)

    def allocate(self, segment_df, total_budget):
        # 数值特征处理
        numeric_features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend']
        numeric_scaled = self.scaler.transform(segment_df[numeric_features].values.astype(np.float32))

        # One-hot 编码
        platform_ids = segment_df['platform_id'].astype(int).values
        geo_ids = segment_df['geo_id'].astype(int).values

        platform_onehot = np.zeros((len(segment_df), self.num_platforms), dtype=np.float32)
        geo_onehot = np.zeros((len(segment_df), self.num_geos), dtype=np.float32)
        platform_onehot[np.arange(len(segment_df)), platform_ids] = 1.0
        geo_onehot[np.arange(len(segment_df)), geo_ids] = 1.0

        # 拼接 state
        states = np.concatenate([numeric_scaled, platform_onehot, geo_onehot], axis=1)

        # 策略预测（归一化 budget）
        predicted = self.model.predict(states).reshape(-1)
        ratios = predicted / np.sum(predicted)
        allocated = ratios * total_budget

        segment_df = segment_df.copy()
        segment_df['predicted_ratio'] = ratios
        segment_df['allocated_budget'] = allocated
        return segment_df

    def simulate_reward(self, allocated_df, roi_column='roi'):
        """
        模拟：使用历史 ROI 来计算该策略的收益
        """
        if roi_column not in allocated_df.columns:
            raise ValueError(f"{roi_column} not in dataframe")
        reward = np.sum(allocated_df['allocated_budget'] * allocated_df[roi_column])
        return reward


# 例如按历史 ROI 平均分配预算的简单规则：
def rule_based_allocation(segment_df, total_budget):
    df = segment_df.copy()
    df["score"] = df["roi_p50"]  # 或 roi_mean
    df["allocated_budget"] = (df["score"] / df["score"].sum()) * total_budget
    return df



==================================

# 1. 预加载 scaler
import joblib
scaler = joblib.load("scaler.pkl")

# 2. 初始化分配器
allocator = QuarterBudgetAllocator(
    model_path="cql_ads_budget_model.pt",
    scaler=scaler,
    num_platforms=3,
    num_geos=15
)

# 3. 输入 segment 的当前状态
segment_df = pd.read_csv("current_segments_states.csv")  # 包含 roi_p10, ctr, ..., roi, platform_id, geo_id

# 4. 执行分配
result_df = allocator.allocate(segment_df, total_budget=1_000_000)

# 5. 模拟收益（offline policy evaluation）
simulated_roi = allocator.simulate_reward(result_df, roi_column="roi")
print("Simulated ROI total reward:", simulated_roi)

# 6. 查看每个 segment 的预算分配
print(result_df[['platform_id', 'geo_id', 'allocated_budget', 'roi']])

