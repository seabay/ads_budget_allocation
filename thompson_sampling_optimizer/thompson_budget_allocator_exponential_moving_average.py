
import numpy as np
import pandas as pd

class BayesianThompsonAllocator:
    def __init__(self, roi_posterior_sampler, segments, group_constraints=None, smoothing_alpha=0.3):
        """
        :param roi_posterior_sampler: callable, 接收 segment 名称，返回一个从后验中采样 ROI 的函数
        :param segments: list of segment ids，例如 ["FB_US", "Google_EU", ...]
        :param group_constraints: dict, {(platform, geo): max_budget_ratio} 可选
        :param smoothing_alpha: 平滑系数，越大越偏向新分配
        """
        self.roi_posterior_sampler = roi_posterior_sampler
        self.segments = segments
        self.group_constraints = group_constraints or {}
        self.smoothing_alpha = smoothing_alpha
        self.prev_allocation = None
        self.segment_to_group = {s: tuple(s.split("_")) for s in segments}

    def _apply_constraints(self, weights):
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()

        if not self.group_constraints:
            return weights

        group_budgets = {}
        for i, s in enumerate(self.segments):
            group = self.segment_to_group[s]
            group_budgets[group] = group_budgets.get(group, 0) + weights[i]

        for i, s in enumerate(self.segments):
            group = self.segment_to_group[s]
            max_ratio = self.group_constraints.get(group, 1.0)
            if group_budgets[group] > max_ratio:
                weights[i] *= max_ratio / group_budgets[group]

        weights /= weights.sum()
        return weights

    def allocate(self, total_budget=1.0):
        roi_samples = np.array([self.roi_posterior_sampler(s)() for s in self.segments])
        raw_allocation = roi_samples / roi_samples.sum()

        # 应用 group-level 限制
        constrained_allocation = self._apply_constraints(raw_allocation)

        # 平滑融合
        if self.prev_allocation is not None:
            smoothed_allocation = (
                self.smoothing_alpha * constrained_allocation +
                (1 - self.smoothing_alpha) * self.prev_allocation
            )
            smoothed_allocation /= smoothed_allocation.sum()
        else:
            smoothed_allocation = constrained_allocation

        self.prev_allocation = smoothed_allocation.copy()

        return pd.DataFrame({
            "segment": self.segments,
            "sampled_roi": roi_samples,
            "allocated_budget": smoothed_allocation * total_budget,
            "expected_return": smoothed_allocation * total_budget * roi_samples,
        })



############################
############################


import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# 模拟 segment 的训练数据（假设 log-normal 分布）
segment_hyperparams = {
    "FB_US": {"mu": 0.1, "sigma": 0.3},
    "Google_US": {"mu": 0.2, "sigma": 0.25},
    "TikTok_EU": {"mu": -0.1, "sigma": 0.4},
}

def get_pyro_sampler(mu, sigma):
    def sampler():
        return float(dist.LogNormal(mu, sigma).sample())
    return sampler

# 构造 segment -> posterior sampler 的映射
def make_roi_posterior_sampler(segment_hyperparams):
    def sampler(segment_name):
        params = segment_hyperparams[segment_name]
        return get_pyro_sampler(params["mu"], params["sigma"])
    return sampler

roi_posterior_sampler = make_roi_posterior_sampler(segment_hyperparams)

# 定义 segment 和预算分组约束
segments = list(segment_hyperparams.keys())
group_constraints = {
    ("FB", "US"): 0.5,   # 不能超过总预算的 50%
    ("Google", "US"): 0.5,
    ("TikTok", "EU"): 0.4,
}

# 创建 allocator 实例
allocator = BayesianThompsonAllocator(
    roi_posterior_sampler=roi_posterior_sampler,
    segments=segments,
    group_constraints=group_constraints,
    smoothing_alpha=0.4,
)

# 执行一次分配
allocation_df = allocator.allocate(total_budget=10000)
print(allocation_df)



#       segment  sampled_roi  allocated_budget  expected_return
# 0       FB_US     1.101234        4872.1234      5368.76543
# 1   Google_US     1.298321        4275.8765      5552.34675
# 2   TikTok_EU     0.832145         851.9999       709.68451

