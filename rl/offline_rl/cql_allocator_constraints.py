

class QuarterBudgetAllocator:
    def __init__(self, model_path, scaler, num_platforms=3, num_geos=15):
        self.model = CQL()
        self.model.build_with_dataset(self._dummy_dataset())
        self.model.load_model(model_path)
        self.scaler = scaler
        self.num_platforms = num_platforms
        self.num_geos = num_geos

    def _dummy_dataset(self):
        obs = np.random.rand(10, 10).astype(np.float32)
        act = np.random.rand(10, 1).astype(np.float32)
        rew = np.random.rand(10).astype(np.float32)
        done = np.ones(10).astype(bool)
        return MDPDataset(obs, act, rew, done)

    def allocate(
        self,
        segment_df,
        total_budget,
        platform_max_ratio=None,   # 如 {0: 0.4, 1: 0.4, 2: 0.4}
        geo_min_budget=None        # 如 {5: 100_000, 8: 80000}
    ):
        # 1. 特征准备
        numeric_features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend']
        numeric_scaled = self.scaler.transform(segment_df[numeric_features].values.astype(np.float32))
        platform_ids = segment_df['platform_id'].astype(int).values
        geo_ids = segment_df['geo_id'].astype(int).values

        platform_onehot = np.zeros((len(segment_df), self.num_platforms), dtype=np.float32)
        geo_onehot = np.zeros((len(segment_df), self.num_geos), dtype=np.float32)
        platform_onehot[np.arange(len(segment_df)), platform_ids] = 1.0
        geo_onehot[np.arange(len(segment_df)), geo_ids] = 1.0

        states = np.concatenate([numeric_scaled, platform_onehot, geo_onehot], axis=1)

        # 2. 策略输出预算比例
        predicted = self.model.predict(states).reshape(-1)
        raw_ratios = predicted / np.sum(predicted)
        raw_alloc = raw_ratios * total_budget

        segment_df = segment_df.copy()
        segment_df['predicted_ratio'] = raw_ratios
        segment_df['allocated_budget'] = raw_alloc

        # 3. 处理 geo 最小预算限制
        if geo_min_budget:
            for geo_id, min_budget in geo_min_budget.items():
                mask = segment_df['geo_id'] == geo_id
                current_geo_budget = segment_df.loc[mask, 'allocated_budget'].sum()
                if current_geo_budget < min_budget:
                    # 增加这个 geo 的预算占比
                    deficit = min_budget - current_geo_budget
                    segment_df.loc[mask, 'allocated_budget'] += (deficit / mask.sum())
                    segment_df['allocated_budget'] *= total_budget / segment_df['allocated_budget'].sum()

        # 4. 处理 platform 最大预算占比限制
        if platform_max_ratio:
            for pid, max_ratio in platform_max_ratio.items():
                mask = segment_df['platform_id'] == pid
                current_platform_budget = segment_df.loc[mask, 'allocated_budget'].sum()
                max_budget = max_ratio * total_budget
                if current_platform_budget > max_budget:
                    scaling = max_budget / current_platform_budget
                    segment_df.loc[mask, 'allocated_budget'] *= scaling
                    # 将剩余预算重新分配到其他 platform
                    overage = total_budget - segment_df['allocated_budget'].sum()
                    non_mask = ~mask
                    total_other = segment_df.loc[non_mask, 'allocated_budget'].sum()
                    if total_other > 0:
                        segment_df.loc[non_mask, 'allocated_budget'] += (
                            segment_df.loc[non_mask, 'allocated_budget'] / total_other * overage
                        )

        segment_df['allocated_budget'] = np.round(segment_df['allocated_budget'], 2)
        return segment_df

    def simulate_reward(self, allocated_df, roi_column='roi'):
        if roi_column not in allocated_df.columns:
            raise ValueError(f"{roi_column} not in dataframe")
        reward = np.sum(allocated_df['allocated_budget'] * allocated_df[roi_column])
        return reward


===========================================

# 1. 创建 allocator
allocator = QuarterBudgetAllocator("cql_ads_budget_model.pt", scaler)

# 2. 分配 + 限制平台预算上限，限制 geo 最小预算
allocated = allocator.allocate(
    segment_df,
    total_budget=1_000_000,
    platform_max_ratio={0: 0.4, 1: 0.4, 2: 0.4},
    geo_min_budget={0: 100_000, 2: 80_000, 5: 120_000}
)

# 3. 查看结果
print(allocated[['platform_id', 'geo_id', 'allocated_budget']])

# 4. 模拟收益
sim_reward = allocator.simulate_reward(allocated, roi_column="roi")
print("Simulated Reward:", sim_reward)


