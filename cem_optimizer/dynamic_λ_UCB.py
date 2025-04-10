
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
