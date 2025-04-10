
import numpy as np

def cem_optimizer(
    roi_ucb,  # shape: [n_segments]
    total_budget,
    num_samples=1000,
    elite_frac=0.1,
    num_iters=20,
    init_mean=None,
    init_std=0.2
):
    n_segments = len(roi_ucb)
    mean = init_mean if init_mean is not None else np.ones(n_segments) / n_segments
    std = np.ones(n_segments) * init_std

    for _ in range(num_iters):
        samples = np.random.normal(loc=mean, scale=std, size=(num_samples, n_segments))
        samples = np.abs(samples)  # 预算不能为负
        samples = samples / samples.sum(axis=1, keepdims=True)  # 归一化为分布
        budgets = samples * total_budget

        returns = (budgets * roi_ucb[None, :]).sum(axis=1)
        elite_idx = returns.argsort()[-int(num_samples * elite_frac):]
        elite_samples = samples[elite_idx]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

    best_allocation = mean * total_budget
    return best_allocation
