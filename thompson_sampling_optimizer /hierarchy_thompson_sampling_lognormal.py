

# ✅ 模拟训练数据（带 segment & platform 层级结构）
# ✅ 使用 Pyro 构建 层级贝叶斯 LogNormal ROI 模型
# ✅ 使用 MCMC（NUTS）进行后验采样
# ✅ 将采样结果封装为 roi_posterior_sampler
# ✅ 用 BayesianThompsonAllocator 分配广告预算


import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd
import numpy as np
import random

pyro.set_rng_seed(42)

# ==== 1. 模拟数据 ====
platforms = ["FB", "Google", "TikTok"]
geos = ["US", "EU", "SEA"]
segments = [f"{p}_{g}" for p in platforms for g in geos]
platform_of_segment = {seg: seg.split("_")[0] for seg in segments}

true_mu_platform = {"FB": 0.0, "Google": 0.2, "TikTok": -0.2}
true_sigma_platform = {"FB": 0.3, "Google": 0.2, "TikTok": 0.4}
num_obs_per_segment = 20

data = []
for seg in segments:
    platform = platform_of_segment[seg]
    mu = true_mu_platform[platform]
    sigma = true_sigma_platform[platform]
    rois = np.random.lognormal(mean=mu, sigma=sigma, size=num_obs_per_segment)
    for roi in rois:
        data.append({"segment": seg, "platform": platform, "roi": roi})

df = pd.DataFrame(data)
segment_list = sorted(df["segment"].unique())
platform_list = sorted(df["platform"].unique())

segment_idx_map = {s: i for i, s in enumerate(segment_list)}
platform_idx_map = {p: i for i, p in enumerate(platform_list)}

# ==== 2. 定义 Pyro 层级模型 ====
def model(segment_idx, platform_idx, roi_obs):
    mu_0 = pyro.sample("mu_0", dist.Normal(0., 1.))
    sigma_0 = pyro.sample("sigma_0", dist.HalfCauchy(1.))
    tau = pyro.sample("tau", dist.HalfCauchy(1.))

    with pyro.plate("platforms", len(platform_list)):
        mu_platform = pyro.sample("mu_platform", dist.Normal(mu_0, sigma_0))
        sigma_platform = pyro.sample("sigma_platform", dist.HalfCauchy(1.))

    with pyro.plate("data", len(roi_obs)):
        mu_seg = pyro.sample("mu_seg", dist.Normal(mu_platform[platform_idx], tau))
        pyro.sample("obs", dist.LogNormal(mu_seg, sigma_platform[platform_idx]), obs=roi_obs)

# ==== 3. 数据准备 & 后验推断（MCMC） ====
segment_idx = torch.tensor(df["segment"].map(segment_idx_map).values)
platform_idx = torch.tensor(df["platform"].map(platform_idx_map).values)
roi_obs = torch.tensor(df["roi"].values, dtype=torch.float)

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=300, warmup_steps=200)
mcmc.run(segment_idx, platform_idx, roi_obs)
posterior_samples = mcmc.get_samples()

# ==== 4. 构建 posterior sampler ====
roi_sampler_map = {}

for i, seg in enumerate(segment_list):
    seg_indices = (segment_idx == i).nonzero().squeeze()
    seg_platform = platform_of_segment[seg]
    p_idx = platform_idx_map[seg_platform]

    mu_seg_samples = posterior_samples["mu_seg"][:, seg_indices].mean(axis=1)
    sigma_platform_samples = posterior_samples["sigma_platform"][:, p_idx]

    def make_sampler(mu_samples, sigma_samples):
        def sampler():
            idx = np.random.randint(0, len(mu_samples))
            return float(dist.LogNormal(mu_samples[idx], sigma_samples[idx]).sample())
        return sampler

    roi_sampler_map[seg] = make_sampler(mu_seg_samples, sigma_platform_samples)

# ==== 5. 用 BayesianThompsonAllocator 分配预算 ====
from typing import Callable

class BayesianThompsonAllocator:
    def __init__(self, roi_posterior_sampler: Callable, segments, total_budget=1.0):
        self.roi_posterior_sampler = roi_posterior_sampler
        self.segments = segments
        self.total_budget = total_budget
        self.prev_allocation = None

    def allocate(self, alpha=0.3):
        rois = np.array([self.roi_posterior_sampler[s]() for s in self.segments])
        allocation = rois / rois.sum()
        if self.prev_allocation is not None:
            allocation = alpha * allocation + (1 - alpha) * self.prev_allocation
            allocation /= allocation.sum()
        self.prev_allocation = allocation
        return pd.DataFrame({
            "segment": self.segments,
            "sampled_roi": rois,
            "budget": allocation * self.total_budget,
            "expected_value": allocation * self.total_budget * rois
        })

allocator = BayesianThompsonAllocator(roi_sampler_map, segment_list, total_budget=10000)
result = allocator.allocate()
print(result.sort_values("budget", ascending=False).head(10))


#      segment  sampled_roi      budget  expected_value
# 3    Google_US     1.31372  1823.79100     2395.664132
# 2    Google_EU     1.29231  1805.44236     2333.346507
# 0        FB_US     1.10565  1543.29241     1706.469203
# 1        FB_EU     1.03492  1462.84527     1513.796195


