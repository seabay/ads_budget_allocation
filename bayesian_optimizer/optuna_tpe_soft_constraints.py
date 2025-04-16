

import optuna
import numpy as np
import pandas as pd

class BudgetOptimizerWithSoftConstraint:
    def __init__(self, roi_predictor, segments_df, total_budget, group_constraints=None, λ=1000):
        """
        roi_predictor: 可调用对象，输入 scaled feature df，输出 roi
        segments_df: 含有 platform_id, geo_id, ctr, cvr, prev_spend, cumulative_spend, time_index 等特征
        total_budget: 总预算
        group_constraints: dict, e.g.:
            {
                'geo_id==1': {'min_ratio': 0.1, 'max_ratio': 0.5},
                'platform_id==0': {'min_ratio': 0.2}
            }
        λ: 违反惩罚项的系数
        """
        self.predictor = roi_predictor
        self.df = segments_df.reset_index(drop=True)
        self.total_budget = total_budget
        self.group_constraints = group_constraints or {}
        self.λ = λ

    def evaluate(self, trial):
        n = len(self.df)
        weights = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(n)])
        weights = weights / weights.sum()  # normalize
        budget_alloc = weights * self.total_budget

        df_alloc = self.df.copy()
        df_alloc['allocated_budget'] = budget_alloc

        # predict ROI
        preds = self.predictor(df_alloc)
        df_alloc['roi'] = preds
        total_roi = (df_alloc['roi'] * df_alloc['allocated_budget']).sum()

        # soft penalty
        penalty = 0.0
        for cond_str, bounds in self.group_constraints.items():
            group_mask = df_alloc.eval(cond_str)
            group_budget = df_alloc.loc[group_mask, 'allocated_budget'].sum()
            ratio = group_budget / self.total_budget
            if 'min_ratio' in bounds and ratio < bounds['min_ratio']:
                penalty += self.λ * (bounds['min_ratio'] - ratio)
            if 'max_ratio' in bounds and ratio > bounds['max_ratio']:
                penalty += self.λ * (ratio - bounds['max_ratio'])

        return -(total_roi - penalty)

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(self.evaluate, n_trials=n_trials)
        best_weights = np.array([study.best_trial.params[f'w_{i}'] for i in range(len(self.df))])
        best_weights = best_weights / best_weights.sum()
        self.df['allocated_budget'] = best_weights * self.total_budget
        return self.df


# =============================================

optimizer = BudgetOptimizerWithSoftConstraint(
    roi_predictor=roi_predictor,
    segments_df=segment_df,
    total_budget=1_000_000,
    group_constraints={
        'geo_id==1': {'min_ratio': 0.1},               # geo 1 至少10%
        'platform_id==2': {'max_ratio': 0.5}           # 平台2最多50%
    },
    λ=500  # 控制惩罚强度
)

result_df = optimizer.optimize(n_trials=200)


