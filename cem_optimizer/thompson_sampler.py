
import numpy as np
import pandas as pd

class ThompsonSampler:
    def __init__(self, segment_df, tau=0.5):
        """
        segment_df: 包含 ['segment', 'roi_p10', 'roi_p50', 'roi_p90']
        tau: softmax 温度参数，控制探索 vs. exploitation
        """
        self.df = segment_df.copy()
        self.tau = tau
        self.total_budget = 1_000_000  # 默认总预算，可后设
        self._initialize_distributions()

    def _initialize_distributions(self):
        self.df["mu"] = self.df["roi_p50"]
        self.df["std"] = (self.df["roi_p90"] - self.df["roi_p10"]) / 2.56
        self.df["std"] = self.df["std"].clip(lower=1e-3)  # 避免为0

    def _softmax(self, x):
        x = np.clip(x, a_min=-10, a_max=10)
        e_x = np.exp(x / self.tau)
        return e_x / e_x.sum()

    def sample_once(self):
        """采样一次并返回预算分配"""
        roi_sampled = np.random.normal(self.df["mu"], self.df["std"])
        budget_ratio = self._softmax(roi_sampled)
        allocated = budget_ratio * self.total_budget

        return pd.DataFrame({
            "segment": self.df["segment"],
            "roi_sampled": roi_sampled,
            "budget_ratio": budget_ratio,
            "allocated_budget": allocated,
        })

    def sample_avg(self, n_samples=50):
        """多次采样后平均输出预算分配"""
        allocations = np.zeros(len(self.df))
        for _ in range(n_samples):
            sampled = self.sample_once()
            allocations += sampled["allocated_budget"].values
        allocations /= n_samples

        result = self.df[["segment"]].copy()
        result["avg_allocated_budget"] = allocations
        return result.sort_values("avg_allocated_budget", ascending=False)

    def update_posterior(self, new_obs: pd.DataFrame, alpha: float = 0.3):
        """
        用历史观测值更新每个 segment 的 ROI 分布参数
        new_obs: 包含 ['segment', 'observed_roi']
        alpha: 衰减系数（新旧数据融合权重）
        """
        update_df = self.df.merge(new_obs, on="segment", how="left")

        for i, row in update_df.iterrows():
            if pd.isna(row["observed_roi"]):
                continue
            mu_old = row["mu"]
            std_old = row["std"]
            roi_new = row["observed_roi"]

            # 指数移动平均更新均值 & 假设固定方差（或也可 EMA 更新）
            mu_updated = (1 - alpha) * mu_old + alpha * roi_new
            std_updated = std_old  # 也可以引入 variance 更新机制

            self.df.loc[i, "mu"] = mu_updated
            self.df.loc[i, "std"] = std_updated

    def set_total_budget(self, budget: float):
        self.total_budget = budget

    def get_current_distribution(self):
        return self.df[["segment", "mu", "std"]]

############################################

# 假设你已有预测好的 ROI 分布（P10 / P50 / P90）
segments = [f"{p}_{g}" for p in ["FB", "GOOG", "TW"] for g in range(1, 16)]
df_roi = pd.DataFrame({
    "segment": segments,
    "roi_p10": np.random.uniform(0.5, 0.8, size=45),
    "roi_p50": np.random.uniform(0.9, 1.2, size=45),
    "roi_p90": np.random.uniform(1.2, 1.6, size=45),
})

sampler = ThompsonSampler(df_roi, tau=0.5)

# 每次采样一轮
allocation = sampler.sample_once()
print(allocation.head())

# 或者多轮平均采样（推荐用于部署）
avg_alloc = sampler.sample_avg(n_samples=100)
print(avg_alloc.head())

# 假设新一轮你获得了真实 ROI 表现
new_data = pd.DataFrame({
    "segment": ["FB_1", "GOOG_5", "TW_9"],
    "observed_roi": [1.4, 1.1, 0.7],
})
sampler.update_posterior(new_data)

# 查看更新后的分布
print(sampler.get_current_distribution().head())
