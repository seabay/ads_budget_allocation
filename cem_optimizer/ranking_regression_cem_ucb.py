

# ranking + regression + CEM + UCB 优化框架模板

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# ---------------------------
# 数据准备
# ---------------------------
# 假设我们已经有了每个 segment 在不同 budget 下的 signup 历史数据
data = ...  # 包含 ['segment_id', 'features', 'budget', 'signup']

# 分为 ranking 用数据和 regression 用数据
X_rank = data[['segment_id', 'features']]
y_rank = data['signup']
X_reg = data[['segment_id', 'features', 'budget']]
y_reg = data['signup']

# 分割数据集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

# ---------------------------
# 回归模型：用于建模 signup = f(segment, budget)
# ---------------------------
reg_model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
reg_model.fit(X_train_reg, y_train_reg)

# ---------------------------
# 排序模型：用于 cold start / exploration 阶段（可选）
# ---------------------------
# 使用 ranking label: 高 signup 为正例
rank_group = data.groupby('segment_id').size().values
rank_model = xgb.XGBRanker(tree_method='hist')
rank_model.fit(X_rank, y_rank, group=rank_group)

# ---------------------------
# Bootstrap ensemble for UCB
# ---------------------------
def bootstrap_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0)
    return mu, sigma

# 构造 bootstrap ensemble
boot_models = []
for i in range(10):
    idx = np.random.choice(len(X_train_reg), size=len(X_train_reg), replace=True)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    model.fit(X_train_reg.iloc[idx], y_train_reg.iloc[idx])
    boot_models.append(model)

# ---------------------------
# CEM 优化器（基于 UCB 评分）
# ---------------------------
def cem_optimize(candidate_budgets, segment_features, n_iter=5, elite_frac=0.2):
    mu, sigma = bootstrap_predict(boot_models, segment_features)
    ucb_scores = mu + 1.0 * sigma  # exploration strength

    n = len(candidate_budgets)
    current_dist = np.ones(n) / n

    for _ in range(n_iter):
        sampled_idx = np.random.choice(n, size=100, p=current_dist)
        sampled_budgets = candidate_budgets[sampled_idx]

        scores = ucb_scores[sampled_idx]
        elite_idx = sampled_idx[np.argsort(scores)[-int(elite_frac * 100):]]

        elite_counts = np.bincount(elite_idx, minlength=n)
        current_dist = elite_counts / elite_counts.sum()

    final_allocation = candidate_budgets[np.argmax(current_dist)]
    return final_allocation

# ---------------------------
# 应用优化器
def allocate_budget(segments, budget_range):
    allocations = {}
    for segment_id, seg_feat in segments.items():
        segment_features = ...  # 构造该 segment 在各个 budget 下的 feature
        allocation = cem_optimize(budget_range, segment_features)
        allocations[segment_id] = allocation
    return allocations

# ---------------------------
# 执行
test_segments = {...}  # segment_id -> 特征
budget_grid = np.linspace(0, 500, 50)  # 预算候选集
alloc = allocate_budget(test_segments, budget_grid)
