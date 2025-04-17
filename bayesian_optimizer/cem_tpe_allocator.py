
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials


class CEMBudgetAllocator:
    def __init__(self, roi_predictor, budget, num_iter=20, pop_size=100, elite_frac=0.2,
                 exploration_sigma=0.1, objectives=('roi',), weights=(1.0,)):
        self.roi_predictor = roi_predictor
        self.budget = budget
        self.num_iter = num_iter
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.exploration_sigma = exploration_sigma
        self.objectives = objectives
        self.weights = weights

    def _sample_population(self, mean, std, n_samples):
        population = np.random.normal(loc=mean, scale=std, size=(n_samples, len(mean)))
        population = np.clip(population, 0, None)
        population = (population.T / population.sum(axis=1) * self.budget).T
        return population

    def _evaluate_population(self, population, segment_df):
        scores = []
        for alloc in population:
            segment_df['alloc'] = alloc
            features = self._construct_features(segment_df)
            preds = self.roi_predictor.predict(features)
            if isinstance(preds, dict):
                multi_scores = [preds[obj].mean() * w for obj, w in zip(self.objectives, self.weights)]
                total_score = sum(multi_scores)
            else:
                total_score = preds.mean()
            scores.append(total_score)
        return np.array(scores)

    def _construct_features(self, df):
        return df.drop(columns=['alloc'], errors='ignore').assign(spend=df['alloc'])

    def allocate(self, segment_df):
        n_segments = len(segment_df)
        mean = np.ones(n_segments) * (self.budget / n_segments)
        std = mean * self.exploration_sigma

        for i in range(self.num_iter):
            population = self._sample_population(mean, std, self.pop_size)
            scores = self._evaluate_population(population, segment_df)
            elite_idx = scores.argsort()[-int(self.pop_size * self.elite_frac):]
            elite_samples = population[elite_idx]
            mean = elite_samples.mean(axis=0)
            std = elite_samples.std(axis=0) + 1e-6

        return pd.DataFrame({
            'segment_id': segment_df['segment_id'],
            'allocated_budget': mean
        })


class TPEBudgetAllocator:
    def __init__(self, roi_predictor, budget, max_evals=100):
        self.roi_predictor = roi_predictor
        self.budget = budget
        self.max_evals = max_evals

    def _construct_features(self, df):
        return df.drop(columns=['alloc'], errors='ignore').assign(spend=df['alloc'])

    def allocate(self, segment_df):
        n_segments = len(segment_df)
        trials = Trials()

        space = [hp.uniform(f'seg_{i}', 0, 1) for i in range(n_segments)]

        def objective(weights):
            weights = np.array(weights)
            alloc = (weights / weights.sum()) * self.budget
            segment_df['alloc'] = alloc
            features = self._construct_features(segment_df)
            preds = self.roi_predictor.predict(features)
            return -preds.mean()

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        best_weights = np.array([best[f'seg_{i}'] for i in range(n_segments)])
        final_alloc = (best_weights / best_weights.sum()) * self.budget

        return pd.DataFrame({
            'segment_id': segment_df['segment_id'],
            'allocated_budget': final_alloc
        })


# 示例调用
# cem_allocator = CEMBudgetAllocator(roi_predictor=model, budget=10000)
# tpe_allocator = TPEBudgetAllocator(roi_predictor=model, budget=10000)
# cem_alloc_result = cem_allocator.allocate(segment_df)
# tpe_alloc_result = tpe_allocator.allocate(segment_df)
