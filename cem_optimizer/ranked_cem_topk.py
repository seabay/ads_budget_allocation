
import numpy as np
from ranking_utils import topk_hit_rate

class CEMBudgetOptimizer:
    def __init__(self, model, n_segments, budget_total,
                 sample_size=1000, elite_frac=0.2, n_iters=10,
                 topk=10):
        self.model = model
        self.n_segments = n_segments
        self.budget_total = budget_total
        self.sample_size = sample_size
        self.elite_frac = elite_frac
        self.n_iters = n_iters
        self.topk = topk

    def sample_dirichlet(self, mean, std):
        alpha = mean / (std + 1e-6)
        alpha = np.clip(alpha, 1e-2, 100)
        samples = np.random.dirichlet(alpha, size=self.sample_size)
        return samples * self.budget_total

    def optimize(self, true_signup_rank):
        # 初始均匀分布
        mean = np.ones(self.n_segments) / self.n_segments
        std = np.ones(self.n_segments) * 0.1

        for iter in range(self.n_iters):
            samples = self.sample_dirichlet(mean, std)
            pred_signups = np.array([self.model.predict(s) for s in samples])

            ranking_scores = np.array([
                topk_hit_rate(p, true_signup_rank, k=self.topk)
                for p in pred_signups
            ])
            elite_idx = ranking_scores.argsort()[-int(self.sample_size * self.elite_frac):]
            elite_samples = samples[elite_idx]

            mean = elite_samples.mean(axis=0)
            std = elite_samples.std(axis=0)

        final_preds = np.array([self.model.predict(s) for s in elite_samples])
        total_signups = final_preds.sum(axis=1)
        best_idx = np.argmax(total_signups)
        return elite_samples[best_idx]


import xgboost as xgb

class SignupPredictor:
    def __init__(self, model: xgb.Booster, segment_features):
        self.model = model
        self.segment_features = segment_features  # 每个 segment 的静态特征

    def predict(self, budget_vector):
        X = self.segment_features.copy()
        X['budget'] = budget_vector
        dmatrix = xgb.DMatrix(X)
        preds = self.model.predict(dmatrix)
        return preds


import numpy as np

def topk_hit_rate(pred, truth, k=10):
    pred_topk = set(np.argsort(pred)[-k:])
    truth_topk = set(np.argsort(truth)[-k:])
    return len(pred_topk & truth_topk) / k


################################

import numpy as np
import xgboost as xgb
import pandas as pd
from model_wrapper import SignupPredictor
from cem import CEMBudgetOptimizer

# 模拟 45个 segment 特征和模型加载
segment_features = pd.read_csv("segment_features.csv")  # 包含 geo、channel 等静态特征
model = xgb.Booster()
model.load_model("signup_model.json")

predictor = SignupPredictor(model, segment_features)

# 假设你统计过一个历史窗口内 segment 的 true signup 排名
true_signup = np.array(pd.read_csv("true_signup.csv")['signup'].values)

# 初始化优化器
optimizer = CEMBudgetOptimizer(
    model=predictor,
    n_segments=45,
    budget_total=1_000_000,
    sample_size=1000,
    elite_frac=0.2,
    n_iters=10,
    topk=10
)

best_budget = optimizer.optimize(true_signup_rank=true_signup)
print("最优预算分配方案:", best_budget)
