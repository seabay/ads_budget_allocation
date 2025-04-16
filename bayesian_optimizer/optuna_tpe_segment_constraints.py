

class ConstrainedBudgetTPEOptimizer:
    def __init__(self, roi_model, scaler, segment_df, total_budget,
                 constraints=None, seed=42):
        self.roi_model = roi_model
        self.scaler = scaler
        self.segment_df = segment_df.copy().reset_index(drop=True)
        self.total_budget = total_budget
        self.n_segments = len(segment_df)
        self.seed = seed

        # constraints: {segment_index: (min_alloc, max_alloc)}
        self.constraints = constraints if constraints else {}

    def _predict_roi(self, alloc):
        df = self.segment_df.copy()
        df['allocated_budget'] = alloc
        X = df[['platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend', 'cumulative_spend', 'time_index']]
        X_scaled = self.scaler.transform(X)
        preds = self.roi_model.predict(X_scaled)
        roi_pred = preds[0.5]
        df['predicted_roi'] = roi_pred
        df['predicted_return'] = df['allocated_budget'] * df['predicted_roi']
        return df['predicted_return'].sum()

    def _objective(self, trial):
        weights = []
        for i in range(self.n_segments):
            w = trial.suggest_float(f'w_{i}', 0.0, 1.0)
            weights.append(w)

        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize
        alloc = weights * self.total_budget

        # Apply constraints
        for i, (min_val, max_val) in self.constraints.items():
            alloc[i] = np.clip(alloc[i], min_val, max_val)

        # Normalize again if constraint violates total_budget
        if alloc.sum() > self.total_budget:
            alloc = alloc / alloc.sum() * self.total_budget

        return -self._predict_roi(alloc)

    def optimize(self, n_trials=100):
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)
        best_trial = study.best_trial
        weights = np.array([best_trial.params[f'w_{i}'] for i in range(self.n_segments)])
        weights = weights / weights.sum()
        alloc = weights * self.total_budget

        for i, (min_val, max_val) in self.constraints.items():
            alloc[i] = np.clip(alloc[i], min_val, max_val)
        if alloc.sum() > self.total_budget:
            alloc = alloc / alloc.sum() * self.total_budget

        result_df = self.segment_df.copy()
        result_df['allocated_budget'] = alloc
        result_df['predicted_roi'] = self.roi_model.predict(self.scaler.transform(
            result_df[['platform_id', 'geo_id', 'ctr', 'cvr', 'prev_spend', 'cumulative_spend', 'time_index']]
        ))[0.5]
        return result_df, -best_trial.value


# ==============================================

# constraints: 限制第 3 个 segment 至少分配 10K，最多不超过 50K
constraints = {
    3: (10000, 50000),
    10: (0, 0),     # 禁投某个 segment
}

optimizer = ConstrainedBudgetTPEOptimizer(
    roi_model=roi_model,
    scaler=scaler,
    segment_df=segment_df,
    total_budget=500000,
    constraints=constraints
)

result_df, best_roi = optimizer.optimize(n_trials=100)

