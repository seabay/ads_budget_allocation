
import pandas as pd
import numpy as np

# 示例数据（可换成真实的XGBoost预测结果）
df = pd.DataFrame({
    "segment": [f"S{i}" for i in range(5)],
    "roi_p10": [0.7, 0.8, 0.5, 1.0, 0.6],
    "roi_p50": [1.0, 1.1, 0.9, 1.3, 1.0],
    "roi_p90": [1.3, 1.4, 1.3, 1.7, 1.4],
})

# 计算估计的标准差（ROI的不确定性）
df["roi_std"] = (df["roi_p90"] - df["roi_p10"]) / 2.56

# 动态 λ 的定义（探索强度）
# 你可以根据训练轮数、数据置信度、季度阶段动态调整
def get_lambda(iteration, max_iter):
    return 1.5 * (1 - iteration / max_iter)  # 随迭代下降（早期探索，后期 exploitation）

iteration = 5
max_iter = 20
lambda_ = get_lambda(iteration, max_iter)

# 计算 UCB
df["roi_ucb"] = df["roi_p50"] + lambda_ * df["roi_std"]

# 输出查看
print(df[["segment", "roi_p50", "roi_std", "roi_ucb"]])


###############
## 完整框架：XGBoost + ROI 不确定性 + UCB + CEM

import pandas as pd
import numpy as np

# 假设有 45 个 segment
segments = [f"{platform}_{geo}" for platform in ["FB", "GOOG", "TW"] for geo in range(1, 16)]

# 假设你已经用 XGBoost 预测出了 ROI 的分位值
# 这里 mock 数据，实际替换成你的模型预测
np.random.seed(42)
df = pd.DataFrame({
    "segment": segments,
    "roi_p10": np.random.uniform(0.4, 0.8, size=45),
    "roi_p50": np.random.uniform(0.8, 1.2, size=45),
    "roi_p90": np.random.uniform(1.0, 1.6, size=45),
})

# 计算估计的标准差
df["roi_std"] = (df["roi_p90"] - df["roi_p10"]) / 2.56

# 动态 λ（你可以按轮数、阶段变化）
def get_lambda(iteration, max_iter):
    return 1.5 * (1 - iteration / max_iter)

iteration = 2
max_iter = 10
lambda_ = get_lambda(iteration, max_iter)

# 计算 UCB 值
df["roi_ucb"] = df["roi_p50"] + lambda_ * df["roi_std"]

# 将 UCB 值标准化，作为 reward
rewards = df["roi_ucb"].values
rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())

# --- CEM 优化器：将预算分配到 segments ---

# 初始化参数
n_samples = 500
n_elite = 50
n_iters = 10
total_budget = 1_000_000
n_segments = len(segments)

# 初始化预算分布（均匀 + 噪声）
mu = np.ones(n_segments) / n_segments
sigma = np.ones(n_segments) * 0.05

for iter in range(n_iters):
    samples = np.random.normal(mu, sigma, size=(n_samples, n_segments))
    samples = np.abs(samples)  # 避免负值
    samples /= samples.sum(axis=1, keepdims=True)  # 每一行归一化为 1（总预算比例）

    # 计算每个 sample 的 ROI（加权平均）
    sample_rewards = samples @ rewards

    # 选出精英样本
    elite_idx = np.argsort(sample_rewards)[-n_elite:]
    elite_samples = samples[elite_idx]

    # 更新分布
    mu = elite_samples.mean(axis=0)
    sigma = elite_samples.std(axis=0)

# 最终 budget 分配结果
df["final_budget_ratio"] = mu
df["final_budget"] = df["final_budget_ratio"] * total_budget

# 输出结果
print(df[["segment", "roi_p50", "roi_ucb", "final_budget"]].sort_values("final_budget", ascending=False).head(10))
