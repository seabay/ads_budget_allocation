# 不追求预测数值的绝对准确，而是通过排序行为优化策略（例如是否能正确找出 top segment）
# 采样多个预算组合，在这些组合上评估预测排序表现（非总量）
# 优化目标可以改成：NDCG. Top-K Precision, 或你自己定义的排序打分函数

import numpy as np

def topk_hit_rate(pred, truth, k=10):
    pred_topk = set(np.argsort(pred)[-k:])
    truth_topk = set(np.argsort(truth)[-k:])
    return len(pred_topk & truth_topk) / k


def cem_ranking_optimizer(
    predict_fn,
    segments,
    total_budget,
    budget_bounds,
    num_iter=30,
    num_samples=100,
    elite_frac=0.2,
    seed=42
):
    np.random.seed(seed)
    N = len(segments)

    # 初始化参数（用高斯分布表示预算分布）
    mu = np.full(N, total_budget / N)
    sigma = np.full(N, total_budget / (3 * N))

    for t in range(num_iter):
        # 采样：正态分布 + 投影到 budget bounds
        samples = np.random.normal(mu, sigma, size=(num_samples, N))
        samples = np.clip(samples, budget_bounds[:, 0], budget_bounds[:, 1])

        # 归一化：每个样本预算总和 = total_budget
        samples = samples / samples.sum(axis=1, keepdims=True) * total_budget

        # 模型预测 signup，并以排序相关指标为目标
        pred_signups = np.array([predict_fn(b) for b in samples])
        # 这一行代码本质是对模型预测的一个“排名打分”的尝试，但实际并不代表 ranking 的好坏
        # ranking_scores = np.array([np.argsort(-s).argsort().mean() for s in pred_signups])
        # 替换为真实排序评估指标，以便更准确选择高潜力 segment。
        ranking_scores = np.array([topk_hit_rate(s, true_signup, k=10) for s in pred_signups])
   
        # 选择精英样本
        elite_idx = ranking_scores.argsort()[:int(elite_frac * num_samples)]
        elite_samples = samples[elite_idx]

        # 更新分布
        mu = elite_samples.mean(axis=0)
        sigma = elite_samples.std(axis=0) + 1e-6  # 避免 std 为 0

    # 输出最终预算分配方案
    return {segment: round(float(b), 2) for segment, b in zip(segments, mu)}
