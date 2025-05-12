

import numpy as np

def cem_with_dirichlet_budget(
    simulator,                   # 模拟器函数：接收 budget 向量，返回 reward
    n_segments=45,
    n_geos=15,
    segment_to_geo=None,
    geo_budget_limit=None,       # 每个 geo 的预算比例上限，如 [0.1, ..., 0.1]
    total_budget=1_000_000,
    penalty_scale=100,
    iterations=50,
    population_size=200,
    elite_frac=0.2,
    init_alpha=10.0,
    verbose=True
):
    if segment_to_geo is None:
        segment_to_geo = [i % n_geos for i in range(n_segments)]
    if geo_budget_limit is None:
        geo_budget_limit = [1.0 / n_geos] * n_geos  # 默认平均上限

    alpha = np.full(n_segments, init_alpha)
    best_score = -np.inf
    best_budget = None

    def evaluate(real_budget):
        reward = simulator(real_budget)
        penalty = 0
        for geo_id in range(n_geos):
            geo_mask = np.array(segment_to_geo) == geo_id
            geo_budget = real_budget[geo_mask].sum()
            geo_limit = geo_budget_limit[geo_id] * total_budget
            if geo_budget > geo_limit:
                penalty += (geo_budget - geo_limit) ** 2
        return reward - penalty_scale * penalty

    for it in range(iterations):
        proportions = np.random.dirichlet(alpha, size=population_size)
        budgets = proportions * total_budget  # 转换为实际预算

        scores = np.array([evaluate(b) for b in budgets])
        elite_idx = scores.argsort()[-int(population_size * elite_frac):]
        elite_props = proportions[elite_idx]

        # 更新 alpha (moment-matching)
        alpha = elite_props.mean(axis=0) * init_alpha

        if scores[elite_idx[-1]] > best_score:
            best_score = scores[elite_idx[-1]]
            best_budget = budgets[elite_idx[-1]]

        if verbose:
            print(f"[Iter {it+1:02d}] Best score so far: {best_score:.4f}")

    return best_budget, best_score

def dummy_simulator(budget_vector):
    # 一个简单模拟器，假设最佳预算为均匀分布
    ideal = np.full_like(budget_vector, fill_value=total_budget / len(budget_vector))
    return -np.linalg.norm(budget_vector - ideal) + np.random.normal(0, 1.0)


best_budget, best_score = cem_with_dirichlet_budget(
    simulator=dummy_simulator,
    total_budget=1_000_000,
    geo_budget_limit=[0.08] * 15,  # 每个 geo 最多占用 8%
)
print("最佳预算分配：", best_budget)
print("总和：", best_budget.sum())


