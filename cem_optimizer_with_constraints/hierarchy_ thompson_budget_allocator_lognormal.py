
import pyro
import pyro.distributions as dist
import torch
import numpy as np
import pandas as pd

class ThompsonBudgetAllocator:
    def __init__(self, segment_df, platform_map, geo_map, tau=0.5, dist_type="normal"):
        self.df = segment_df.copy()
        self.platform_map = platform_map
        self.geo_map = geo_map
        self.df["platform"] = self.df["segment"].map(platform_map)
        self.df["geo"] = self.df["segment"].map(geo_map)
        self.total_budget = 1_000_000  # 设定总预算
        self.tau = tau
        self.dist_type = dist_type
        self.constraints = {"platform": {}, "geo": {}}
        self._initialize_distributions()

    def _initialize_distributions(self):
        """
        初始化分布类型
        """
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
                a, b = fit_beta_from_quantiles(row["roi_p10"], row["roi_p50"], row["roi_p90"])
                a_list.append(a)
                b_list.append(b)
            self.df["alpha"] = a_list
            self.df["beta"] = b_list
        else:
            raise ValueError("dist_type must be one of: normal, lognormal, beta")

    def hierarchical_roi_model(self, roi_data, segment_ids):
        n_segments = segment_ids.max().item() + 1
        mu_global = pyro.sample("mu_global", dist.Normal(0., 5.))
        tau_mu = pyro.sample("tau_mu", dist.HalfNormal(5.))
        tau_sigma = pyro.sample("tau_sigma", dist.HalfNormal(5.))

        with pyro.plate("segments", n_segments):
            mu_segment = pyro.sample("mu_segment", dist.Normal(mu_global, tau_mu))
            sigma_segment = pyro.sample("sigma_segment", dist.HalfNormal(tau_sigma))

        with pyro.plate("observations", len(roi_data)):
            roi_mu = mu_segment[segment_ids]
            roi_sigma = sigma_segment[segment_ids]
            pyro.sample("obs", dist.Normal(roi_mu, roi_sigma), obs=roi_data)

    def run_bayesian_inference(self, roi_data, segment_ids, num_samples=1000, warmup_steps=200):
        nuts_kernel = pyro.infer.NUTS(self.hierarchical_roi_model)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(roi_data, segment_ids)
        posterior_samples = mcmc.get_samples()
        return posterior_samples

    def sample_once(self, roi_data, segment_ids):
        posterior = self.run_bayesian_inference(roi_data, segment_ids)
        posterior_mu = posterior['mu_segment'].numpy()
        posterior_sigma = posterior['sigma_segment'].numpy()

        # 从后验分布中进行采样
        roi_sampled = np.random.normal(posterior_mu, posterior_sigma)

        budget_ratio = self._softmax(roi_sampled)
        allocated = budget_ratio * self.total_budget

        df_alloc = self.df.copy()
        df_alloc["roi_sampled"] = roi_sampled
        df_alloc["budget_ratio"] = budget_ratio
        df_alloc["allocated_budget"] = allocated
        df_alloc = self._apply_group_constraints(df_alloc)

        return df_alloc[["segment", "platform", "geo", "roi_sampled", "allocated_budget"]]

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def _apply_group_constraints(self, df_alloc):
        for group, limits in self.constraints.items():
            for key, limit in limits.items():
                df_alloc.loc[df_alloc[group] == key, "allocated_budget"] = \
                    df_alloc.loc[df_alloc[group] == key, "allocated_budget"].clip(0, limit * self.total_budget)
        return df_alloc




###########
###############  假设我们有如下的广告 segment 数据：

# 假设你有如下的数据框：segment_df，包含各个segment的历史 ROI 数据
segment_df = pd.DataFrame({
    "segment": ["seg1", "seg2", "seg3"],
    "roi_p10": [0.8, 0.9, 0.7],
    "roi_p50": [1.2, 1.3, 1.0],
    "roi_p90": [1.5, 1.6, 1.3]
})

# Platform 和 Geo 的映射
platform_map = {"seg1": "FB", "seg2": "Google", "seg3": "Twitter"}
geo_map = {"seg1": "US", "seg2": "EU", "seg3": "APAC"}

# 创建一个 ThompsonBudgetAllocator 对象
allocator = ThompsonBudgetAllocator(
    segment_df=segment_df,
    platform_map=platform_map,
    geo_map=geo_map,
    dist_type="normal",  # 或 "lognormal" / "beta"
    tau=0.5
)

# 假设历史 ROI 数据和 segment_id 映射
roi_data = [1.2, 1.0, 0.9, 1.3, 1.1, 1.0]
segment_ids = [0, 0, 1, 1, 2, 2]  # 对应seg1, seg2, seg3

# 执行采样，得到预算分配
allocation_result = allocator.sample_once(roi_data, segment_ids)

print(allocation_result)
