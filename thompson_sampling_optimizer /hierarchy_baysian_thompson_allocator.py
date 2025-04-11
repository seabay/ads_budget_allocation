
# 假设已经使用模型预测 segment 的 ROI

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np
import pandas as pd
from typing import Callable, Dict

pyro.set_rng_seed(42)

class HierarchicalBayesianROIModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.segment_list = sorted(df["segment"].unique())
        self.platform_list = sorted(df["platform"].unique())
        self.segment_idx_map = {s: i for i, s in enumerate(self.segment_list)}
        self.platform_idx_map = {p: i for i, p in enumerate(self.platform_list)}

        self.segment_idx = torch.tensor(df["segment"].map(self.segment_idx_map).values)
        self.platform_idx = torch.tensor(df["platform"].map(self.platform_idx_map).values)
        self.roi_obs = torch.tensor(df["roi"].values, dtype=torch.float)

    def model(self, segment_idx, platform_idx, roi_obs):
        mu_0 = pyro.sample("mu_0", dist.Normal(0., 1.))
        sigma_0 = pyro.sample("sigma_0", dist.HalfCauchy(1.))
        tau = pyro.sample("tau", dist.HalfCauchy(1.))

        with pyro.plate("platforms", len(self.platform_list)):
            mu_platform = pyro.sample("mu_platform", dist.Normal(mu_0, sigma_0))
            sigma_platform = pyro.sample("sigma_platform", dist.HalfCauchy(1.))

        with pyro.plate("data", len(roi_obs)):
            mu_seg = pyro.sample("mu_seg", dist.Normal(mu_platform[platform_idx], tau))
            pyro.sample("obs", dist.LogNormal(mu_seg, sigma_platform[platform_idx]), obs=roi_obs)

    def run_mcmc(self, num_samples=300, warmup_steps=200):
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(self.segment_idx, self.platform_idx, self.roi_obs)
        self.posterior_samples = mcmc.get_samples()

    def build_roi_samplers(self) -> Dict[str, Callable[[], float]]:
        roi_sampler_map = {}
        platform_of_segment = {seg: seg.split("_")[0] for seg in self.segment_list}

        for i, seg in enumerate(self.segment_list):
            p = platform_of_segment[seg]
            p_idx = self.platform_idx_map[p]
            seg_indices = (self.segment_idx == i).nonzero().squeeze()

            mu_seg_samples = self.posterior_samples["mu_seg"][:, seg_indices].mean(axis=1)
            sigma_platform_samples = self.posterior_samples["sigma_platform"][:, p_idx]

            def make_sampler(mu_samples, sigma_samples):
                def sampler():
                    idx = np.random.randint(0, len(mu_samples))
                    return float(dist.LogNormal(mu_samples[idx], sigma_samples[idx]).sample())
                return sampler

            roi_sampler_map[seg] = make_sampler(mu_seg_samples, sigma_platform_samples)

        return roi_sampler_map

class BayesianThompsonAllocator:
    def __init__(self, roi_posterior_sampler: Dict[str, Callable[[], float]], segments, total_budget=1.0):
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

# 示例使用:
# df = pd.DataFrame(...)  # 包含 segment, platform, roi 三列
# model = HierarchicalBayesianROIModel(df)
# model.run_mcmc()
# roi_samplers = model.build_roi_samplers()
# allocator = BayesianThompsonAllocator(roi_samplers, model.segment_list, total_budget=10000)
# result = allocator.allocate()
# print(result.sort_values("budget", ascending=False))
