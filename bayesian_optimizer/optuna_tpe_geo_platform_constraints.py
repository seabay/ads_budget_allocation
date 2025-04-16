

from collections import defaultdict

class GroupConstrainedBudgetTPEOptimizer:
    def __init__(
        self, roi_model, scaler, segment_df, total_budget,
        segment_constraints=None, group_constraints=None,
        seed=42
    ):
        self.roi_model = roi_model
        self.scaler = scaler
        self.segment_df = segment_df.copy().reset_index(drop=True)
        self.total_budget = total_budget
        self.n_segments = len(segment_df)
        self.seed = seed

        # segment_constraints: {segment_idx: (min_budget, max_budget)}
        self.segment_constraints = segment_constraints or {}

        # group_constraints: dict with structure like:
        # {'geo_id': {1: (min, max), 2: (min, max)}, 'platform_id': {...}}
        self.group_constraints = group_constraints or {}

    def _predict_roi(self, alloc):
        df = self.segment_df.copy()
        df['allocated_budget'] = alloc
        X = df[['platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend', 'cumulative_spend', 'time_index']]
        X_scaled = self.scaler.transform(X)
        preds = self.roi_model.predict(X_scaled)
        df['predicted_roi'] = preds[0.5]
        df['predicted_return'] = df['allocated_budget'] * df['predicted_roi']
        return df['predicted_return'].sum()

    def _apply_segment_constraints(self, alloc):
        for i, (min_val, max_val) in self.segment_constraints.items():
            alloc[i] = np.clip(alloc[i], min_val, max_val)
        return alloc

    def _apply_group_constraints(self, alloc):
        df = self.segment_df.copy()
        df['alloc'] = alloc.copy()

        for group_key, group_lims in self.group_constraints.items():
            grouped = df.groupby(group_key)['alloc'].sum().to_dict()
            for group_value, (min_limit, max_limit) in group_lims.items():
                group_mask = df[group_key] == group_value
                group_indices = df[group_mask].index

                current_total = grouped.get(group_value, 0.0)
                if current_total < min_limit:
                    # 增加分配比例
                    diff = min_limit - current_total
                    alloc[group_indices] += diff / len(group_indices)
                elif current_total > max_limit:
                    # 减少分配比例
                    diff = current_total - max_limit
                    alloc[group_indices] -= diff / len(group_indices)
                    alloc[group_indices] = np.clip(alloc[group_indices], 0.0, None)

        return alloc

    def _objective(self, trial):
        weights = np.array([trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(self.n_segments)])
        weights = weights / weights.sum()
        alloc = weights * self.total_budget

        alloc = self._apply_segment_constraints(alloc)
        alloc = self._apply_group_constraints(alloc)

        # Renormalize if over total_budget
        if alloc.sum() > self.total_budget:
            alloc = alloc / alloc.sum() * self.total_budget

        return -self._predict_roi(alloc)

    def optimize(self, n_trials=100):
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)

        best_weights = np.array([study.best_trial.params[f'w_{i}'] for i in range(self.n_segments)])
        best_weights = best_weights / best_weights.sum()
        best_alloc = best_weights * self.total_budget

        best_alloc = self._apply_segment_constraints(best_alloc)
        best_alloc = self._apply_group_constraints(best_alloc)

        if best_alloc.sum() > self.total_budget:
            best_alloc = best_alloc / best_alloc.sum() * self.total_budget

        result_df = self.segment_df.copy()
        result_df['allocated_budget'] = best_alloc
        X = result_df[['platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend', 'cumulative_spend', 'time_index']]
        preds = self.roi_model.predict(self.scaler.transform(X))
        result_df['predicted_roi'] = preds[0.5]

        return result_df, -study.best_trial.value


# =================================================

group_constraints = {
    'geo_id': {
        1: (20000, 50000),
        3: (0, 0),       # 禁投 geo 3
    },
    'platform_id': {
        0: (100000, 200000),  # 平台 0 至少分配 10w，最多 20w
    }
}

optimizer = GroupConstrainedBudgetTPEOptimizer(
    roi_model=roi_model,
    scaler=scaler,
    segment_df=segment_df,
    total_budget=500000,
    group_constraints=group_constraints
)

result_df, best_total_return = optimizer.optimize(n_trials=100)


