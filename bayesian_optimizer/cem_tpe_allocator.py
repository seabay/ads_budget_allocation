
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


class OptunaTPEBudgetAllocator:
    def __init__(self, roi_predictor, budget, n_trials=100):
        self.roi_predictor = roi_predictor
        self.budget = budget
        self.n_trials = n_trials

    def _construct_features(self, df):
        return df.drop(columns=['alloc'], errors='ignore').assign(spend=df['alloc'])

    def allocate(self, segment_df):
        n_segments = len(segment_df)

        def objective(trial):
            weights = np.array([trial.suggest_float(f'seg_{i}', 0.01, 1.0) for i in range(n_segments)])
            alloc = (weights / weights.sum()) * self.budget
            segment_df['alloc'] = alloc
            features = self._construct_features(segment_df)
            preds = self.roi_predictor.predict(features)
            return -preds.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        best_weights = np.array([study.best_trial.params[f'seg_{i}'] for i in range(n_segments)])
        final_alloc = (best_weights / best_weights.sum()) * self.budget

        return pd.DataFrame({
            'segment_id': segment_df['segment_id'],
            'allocated_budget': final_alloc
        })

from concurrent.futures import ThreadPoolExecutor
import optuna

class OptunaTPEBudgetAllocator:
    def __init__(self, roi_predictor, budget, n_trials=100, n_jobs=4):
        self.roi_predictor = roi_predictor
        self.budget = budget
        self.n_trials = n_trials
        self.n_jobs = n_jobs

    def _construct_features(self, df):
        return df.drop(columns=['alloc'], errors='ignore').assign(spend=df['alloc'])

    def allocate(self, segment_df):
        n_segments = len(segment_df)

        def objective(trial):
            weights = np.array([trial.suggest_float(f'seg_{i}', 0.01, 1.0) for i in range(n_segments)])
            alloc = (weights / weights.sum()) * self.budget
            segment_df['alloc'] = alloc
            features = self._construct_features(segment_df)
            preds = self.roi_predictor.predict(features)
            return -preds.mean()

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

        def run_trials(n_trials_per_job):
            study.optimize(objective, n_trials=n_trials_per_job)

        trials_per_job = self.n_trials // self.n_jobs

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(run_trials, trials_per_job) for _ in range(self.n_jobs)]
            for f in futures:
                f.result()  # wait for completion

        best_weights = np.array([study.best_trial.params[f'seg_{i}'] for i in range(n_segments)])
        final_alloc = (best_weights / best_weights.sum()) * self.budget

        return pd.DataFrame({
            'segment_id': segment_df['segment_id'],
            'allocated_budget': final_alloc
        })

# Optuna 的分布式执行（RDB backend），
# 你只需要设置一个共享的数据库（如 SQLite 或 MySQL）作为 study 的后端。
# 这样多个进程或机器就能并发运行同一个 study
class OptunaTPEBudgetAllocator:
    def __init__(self, roi_predictor, budget, n_trials=100, storage=None, study_name=None, load_if_exists=True):
        self.roi_predictor = roi_predictor
        self.budget = budget
        self.n_trials = n_trials
        self.storage = storage
        self.study_name = study_name
        self.load_if_exists = load_if_exists

    def _construct_features(self, df):
        return df.drop(columns=['alloc'], errors='ignore').assign(spend=df['alloc'])

    def allocate(self, segment_df):
        n_segments = len(segment_df)

        def objective(trial):
            weights = np.array([trial.suggest_float(f'seg_{i}', 0.01, 1.0) for i in range(n_segments)])
            alloc = (weights / weights.sum()) * self.budget
            segment_df['alloc'] = alloc
            features = self._construct_features(segment_df)
            preds = self.roi_predictor.predict(features)
            return -preds.mean()

        if self.storage:
            study = optuna.create_study(
                study_name=self.study_name,
                direction="minimize",
                storage=self.storage,
                load_if_exists=self.load_if_exists
            )
        else:
            study = optuna.create_study(direction="minimize")

        study.optimize(objective, n_trials=self.n_trials)

        best_weights = np.array([study.best_trial.params[f'seg_{i}'] for i in range(n_segments)])
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
