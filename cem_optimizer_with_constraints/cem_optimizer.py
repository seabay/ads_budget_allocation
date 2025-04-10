
import numpy as np

def cem_optimizer_with_constraints(
    roi_ucb,  # shape: [n_segments]
    total_budget,
    group_constraints,  # {group_name: (min_percentage, max_percentage)} 每个分组的预算约束
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

        # 应用分组约束
        for group, (min_pct, max_pct) in group_constraints.items():
            group_idx = [i for i, label in enumerate(df["platform"]) if label == group]
            group_budget = budgets[:, group_idx].sum(axis=1)
            min_budget = min_pct * total_budget
            max_budget = max_pct * total_budget
            group_budget = np.clip(group_budget, min_budget, max_budget)

            # 更新预算：确保该分组的预算在指定范围内
            budgets[:, group_idx] = (budgets[:, group_idx].sum(axis=1, keepdims=True) / group_budget[:, None]) * group_budget[:, None]

        returns = (budgets * roi_ucb[None, :]).sum(axis=1)
        elite_idx = returns.argsort()[-int(num_samples * elite_frac):]
        elite_samples = samples[elite_idx]

        mean = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)

    best_allocation = mean * total_budget
    return best_allocation
