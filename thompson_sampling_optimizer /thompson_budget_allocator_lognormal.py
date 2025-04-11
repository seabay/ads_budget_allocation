

# 你一共有 45 个 segment（3 个 platform × 15 个 geo），现在我们会添加：
# 平台层级约束：如 Facebook 总预算 ≤ 40%
# 地理区域约束：如 geo_1 总预算 ≥ 5%


import pandas as pd
import numpy as np

from scipy.stats import beta, lognorm
from scipy.optimize import minimize
import numpy as np

def fit_beta_from_quantiles(p10, p50, p90):
    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return np.inf
        err = (
            (beta.ppf(0.1, a, b) - p10) ** 2 +
            (beta.ppf(0.5, a, b) - p50) ** 2 +
            (beta.ppf(0.9, a, b) - p90) ** 2
        )
        return err
    result = minimize(objective, [2.0, 2.0], method='Nelder-Mead')
    return result.x if result.success else (1.0, 1.0)

def fit_lognormal_from_quantiles(p10, p50, p90):
    log_p10, log_p50, log_p90 = np.log([p10, p50, p90])
    def objective(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        samples = np.random.normal(mu, sigma, 10000)
        err = (
            (np.percentile(samples, 10) - log_p10) ** 2 +
            (np.percentile(samples, 50) - log_p50) ** 2 +
            (np.percentile(samples, 90) - log_p90) ** 2
        )
        return err
    result = minimize(objective, [log_p50, 0.5], method='Nelder-Mead')
    return result.x if result.success else (log_p50, 0.5)


class ThompsonBudgetAllocator:
    def __init__(self, segment_df, platform_map, geo_map, tau=0.5):
        """
        segment_df: 包含 ['segment', 'roi_p10', 'roi_p50', 'roi_p90']
        platform_map: segment → platform 的映射 dict
        geo_map: segment → geo 的映射 dict
        """
        self.df = segment_df.copy()
        self.platform_map = platform_map
        self.geo_map = geo_map
        self.df["platform"] = self.df["segment"].map(platform_map)
        self.df["geo"] = self.df["segment"].map(geo_map)
        self.total_budget = 1_000_000
        self.tau = tau
        self.dist_type = dist_type  # <-- 支持 normal / beta / lognormal
        self.constraints = {"platform": {}, "geo": {}}
        self._initialize_distributions()

    def _initialize_distributions(self):

        if self.dist_type == "normal":
            self.df["mu"] = self.df["roi_p50"]
            self.df["std"] = (self.df["roi_p90"] - self.df["roi_p10"]) / 2.56
            self.df["std"] = self.df["std"].clip(lower=1e-3)
        elif self.dist_type == "lognormal":
            mus, sigmas = [], []
            for _, row in self.df.iterrows():
                mu, sigma = fit_lognormal_from_quantiles(row["roi_p10"], row["roi_p50"], row["roi_p90"])
                mus.append(mu)
                sigmas.append(sigma)
            self.df["mu"] = mus
            self.df["sigma"] = sigmas
        elif self.dist_type == "beta":
            a_list, b_list = [], []
            for _, row in self.df.iterrows():
                # 假设 ROI 已归一化到 [0, 1]
                a, b = fit_beta_from_quantiles(row["roi_p10"], row["roi_p50"], row["roi_p90"])
                a_list.append(a)
                b_list.append(b)
            self.df["alpha"] = a_list
            self.df["beta"] = b_list
        else:
            raise ValueError("dist_type must be one of: normal, lognormal, beta")

    def set_total_budget(self, budget):
        self.total_budget = budget

    def set_group_constraints(self, platform_limits=None, geo_limits=None):
        """设置预算比例约束，如 {'FB': 0.4} 表示 ≤ 40%"""
        if platform_limits:
            self.constraints["platform"] = platform_limits
        if geo_limits:
            self.constraints["geo"] = geo_limits

    def _softmax(self, x):
        x = np.clip(x, a_min=-10, a_max=10)
        e_x = np.exp(x / self.tau)
        return e_x / e_x.sum()

    def _apply_group_constraints(self, df_alloc):
        # Step 1: group by platform & geo
        for level in ["platform", "geo"]:
            limits = self.constraints[level]
            if not limits:
                continue
            group_sum = df_alloc.groupby(level)["allocated_budget"].sum().to_dict()
            for group, limit_ratio in limits.items():
                max_budget = self.total_budget * limit_ratio
                actual_budget = group_sum.get(group, 0)
                if actual_budget > max_budget:
                    scaling = max_budget / actual_budget
                    mask = df_alloc[level] == group
                    df_alloc.loc[mask, "allocated_budget"] *= scaling
        return df_alloc

    def sample_once(self):
        if self.dist_type == "normal":
            roi_sampled = np.random.normal(self.df["mu"], self.df["std"])
        elif self.dist_type == "lognormal":
            roi_sampled = lognorm(s=self.df["sigma"], scale=np.exp(self.df["mu"])).rvs()
        elif self.dist_type == "beta":
            roi_sampled = beta(self.df["alpha"], self.df["beta"]).rvs()
        else:
            raise ValueError("Invalid dist_type")

        budget_ratio = self._softmax(roi_sampled)
        allocated = budget_ratio * self.total_budget

        df_alloc = self.df.copy()
        df_alloc["roi_sampled"] = roi_sampled
        df_alloc["budget_ratio"] = budget_ratio
        df_alloc["allocated_budget"] = allocated
        df_alloc = self._apply_group_constraints(df_alloc)

        return df_alloc[["segment", "platform", "geo", "roi_sampled", "allocated_budget"]]


    def sample_avg(self, n_samples=50):
        allocs = np.zeros(len(self.df))
        for _ in range(n_samples):
            sampled = self.sample_once()
            allocs += sampled["allocated_budget"].values
        allocs /= n_samples
        result = self.df[["segment", "platform", "geo"]].copy()
        result["avg_allocated_budget"] = allocs
        return result.sort_values("avg_allocated_budget", ascending=False)

    def update_posterior(self, new_obs: pd.DataFrame, alpha: float = 0.3):
        """
        new_obs: ['segment', 'observed_roi']
        alpha: EMA 更新速率
        """
        update_df = self.df.merge(new_obs, on="segment", how="left")
        for i, row in update_df.iterrows():
            if pd.isna(row["observed_roi"]):
                continue
            mu_old = row["mu"]
            roi_new = row["observed_roi"]
            mu_updated = (1 - alpha) * mu_old + alpha * roi_new
            self.df.loc[i, "mu"] = mu_updated

    def get_current_distribution(self):
        return self.df[["segment", "mu", "std", "platform", "geo"]]


###################################################

# 1. 输入数据准备
segments = [f"{p}_{g}" for p in ["FB", "GOOG", "TW"] for g in range(1, 16)]
df_roi = pd.DataFrame({
    "segment": segments,
    "roi_p10": np.random.uniform(0.5, 0.8, size=45),
    "roi_p50": np.random.uniform(0.9, 1.2, size=45),
    "roi_p90": np.random.uniform(1.2, 1.6, size=45),
})
platform_map = {seg: seg.split("_")[0] for seg in segments}
geo_map = {seg: f"geo_{seg.split('_')[1]}" for seg in segments}

# 2. 初始化 allocator
# allocator = ThompsonBudgetAllocator(df_roi, platform_map, geo_map, tau=0.5)

allocator = ThompsonBudgetAllocator(
    segment_df=df_roi,
    platform_map=platform_map,
    geo_map=geo_map,
    dist_type="lognormal",  # <--- 可切换为 "beta" 或 "normal"
    tau=0.5
)


# 3. 设置预算总额 & 分组约束（如 FB ≤ 40%，geo_3 ≤ 5%）
allocator.set_total_budget(1_000_000)
allocator.set_group_constraints(platform_limits={"FB": 0.4},
                                geo_limits={"geo_3": 0.05})

# 4. 平均采样 50 次后输出稳定预算分配
alloc_result = allocator.sample_avg(n_samples=50)
print(alloc_result.head())

# 5. 接入真实 ROI 更新
new_obs = pd.DataFrame({
    "segment": ["FB_1", "GOOG_5"],
    "observed_roi": [1.5, 0.9],
})
allocator.update_posterior(new_obs)
