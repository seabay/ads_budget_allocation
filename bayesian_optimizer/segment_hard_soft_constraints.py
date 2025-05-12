# 总预算 = 1
# 每个 geo 的预算上限
# TPESampler(multivariate=True)，考虑变量之间的相关性

import optuna
import numpy as np

# 设定 segment 与 geo 映射（总共 45 个 segment，每 geo 有 3 个 segment）
segment_to_geo = [i % 15 for i in range(45)]  # geo0, geo1, ..., geo14

# 每个 geo 的预算上限（例如，每个 geo 最多 8% 的预算）
geo_budget_limit = [0.08 for _ in range(15)]

# 模拟器（可以替换为你的 XGBoost/LSTM 模型）
def simulator(budget_vector):
    # 模拟返回 signup 数（这里假设最优分配在 budget ≈ 0.5 处）
    return -np.linalg.norm(budget_vector - 0.5) + np.random.normal(0, 0.01)

# 目标函数：返回 reward（signups），并确保满足 geo 限制
def objective(trial):
    # 提取 45 个 budget 分量（0 ~ 1）
    raw = [trial.suggest_float(f"x{i}", 1e-6, 1.0) for i in range(45)]
    budget = np.array(raw)
    budget /= budget.sum()  # 归一化保证总和为 1

    # geo-level 限制检查
    for geo_id in range(15):
        geo_mask = np.array(segment_to_geo) == geo_id
        geo_budget = budget[geo_mask].sum()
        if geo_budget > geo_budget_limit[geo_id]:
            raise optuna.TrialPruned()  # geo预算超限，丢弃此 trial

    # 模拟器计算 reward（如 signup 数）
    reward = simulator(budget)
    return reward

# 创建优化器（Bayesian Optimization + multivariate TPE）
sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

# 执行优化
study.optimize(objective, n_trials=200)

# 输出最优结果
print("Best reward:", study.best_value)
best_budget = np.array([study.best_params[f"x{i}"] for i in range(45)])
best_budget /= best_budget.sum()

# reshape 为 [platform x geo] 格式（如需）
budget_matrix = best_budget.reshape((3, 15))
print("Best budget allocation matrix:\n", budget_matrix)


# ======

# ✅ 1. Geo 预算上限用 soft constraint（罚分项）
# 我们不再直接 TrialPruned()，而是加一个 惩罚项 到 reward 里。

# ✅ 2. 使用 Dirichlet 初始化（自然归一化）
# Dirichlet 保证预算是正的且和为 1，很适合 budget 分配问题。

import optuna
import numpy as np

# === 基本设置 ===
n_segments = 45
n_geos = 15
segment_to_geo = [i % n_geos for i in range(n_segments)]

# geo budget soft constraint：每 geo 最多 8% 的预算
geo_budget_limit = [0.08] * n_geos
penalty_scale = 100  # 违反约束后的惩罚系数

# === 模拟器（可替换为真实 XGBoost / LSTM 模型）===
def simulator(budget_vector):
    # 模拟 signups（目标是在中间值时最好）
    return -np.linalg.norm(budget_vector - 0.5) + np.random.normal(0, 0.01)

# === 目标函数 ===
def objective(trial):
    # 使用 Dirichlet 采样 budget 向量
    alpha = np.ones(n_segments)
    budget = np.array(trial.suggest_dirichlet("budget", alpha))  # 自动归一化

    # --- 软约束惩罚 ---
    penalty = 0.0
    for geo_id in range(n_geos):
        geo_mask = np.array(segment_to_geo) == geo_id
        geo_budget = budget[geo_mask].sum()
        limit = geo_budget_limit[geo_id]
        if geo_budget > limit:
            penalty += (geo_budget - limit) ** 2  # 平方惩罚

    # 模拟器预测 signup
    reward = simulator(budget)

    # 返回 reward - penalty（惩罚项）
    return reward - penalty_scale * penalty

# === 创建 study ===
sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

# === 执行优化 ===
study.optimize(objective, n_trials=200)

# === 输出结果 ===
best_budget = np.array(study.best_params["budget"])
print("Best reward:", study.best_value)
print("Best budget vector (sum = %.3f):" % best_budget.sum(), best_budget)

# 可 reshape 为平台 x geo 格式
budget_matrix = best_budget.reshape((3, 15))
print("Best budget allocation matrix:\n", budget_matrix)
