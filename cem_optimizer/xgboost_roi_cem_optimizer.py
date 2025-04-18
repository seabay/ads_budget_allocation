
# 使用 GBDT 建模 segment + budget → ROI 的关系
# 构造 ROI 函数用于 CEM 优化
# 通过 Cross-Entropy Method 优化整体预算分配以最大化 ROI * budget 的总收益

# 时间序列支持：通过 time_col 按 segment 排序。
# 历史收益信息：增加了 roi_lag_1 到 roi_lag_n 的滚动窗口特征。
# 动态默认特征：在构造 ROI 函数时可传入默认值，如最近一次历史 ROI 等


# 季度特征（quarter）：通过 time 字段提取季度信息作为周期性特征。
# 移动平均 ROI（roi_ma_N）：按 segment 分组后添加滑动窗口平均历史 ROI

# 加入节假日特征（如是否为美国假期）；
# 加入时间差分特征（每个 segment 的时间间隔）；
# 所有时间特征、滚动窗口、季度等特征也一并整合进模型输入

# 是否为周末 (is_weekend)
# 营销活动周期 (campaign_period，占位符字段)
# 行业事件标志 (industry_event，占位符字段)


import torch
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import holidays

class ROIModelGBDT:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str] = ["segment", "budget"], time_col: str = "time", history_window: int = 3):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.history_window = history_window

        # Add rolling historical ROI features per segment
        if time_col in df.columns:
            self.df = self.df.sort_values(by=["segment", time_col])
            for i in range(1, history_window + 1):
                self.df[f"roi_lag_{i}"] = self.df.groupby("segment")["roi"].shift(i)
            self.df[f"roi_ma_{history_window}"] = self.df.groupby("segment")["roi"].transform(lambda x: x.shift(1).rolling(history_window).mean())

            # Add cyclic time features (e.g., quarter)
            self.df["quarter"] = pd.to_datetime(self.df[time_col]).dt.quarter.astype(str)

            # Add holiday flag (assumes US holidays, customize as needed)
            us_holidays = holidays.UnitedStates()
            self.df["is_holiday"] = pd.to_datetime(self.df[time_col]).isin(us_holidays).astype(int)

            # Add time difference from previous record per segment
            self.df["time_diff"] = self.df.groupby("segment")[time_col].transform(lambda x: pd.to_datetime(x).diff().dt.days.fillna(0))

            # Add weekend flag
            self.df["is_weekend"] = pd.to_datetime(self.df[time_col]).dt.weekday.isin([5, 6]).astype(int)

            # Add placeholder for campaign cycle or industry events (to be customized)
            self.df["campaign_period"] = 0  # can be replaced by real campaign mappings
            self.df["industry_event"] = 0   # can be replaced by real event flags

            self.df = self.df.dropna().reset_index(drop=True)
            self.feature_cols += [f"roi_lag_{i}" for i in range(1, history_window + 1)]
            self.feature_cols.append(f"roi_ma_{history_window}")
            self.feature_cols += ["quarter", "is_holiday", "time_diff", "is_weekend", "campaign_period", "industry_event"]

        # Handle categorical encoding
        categorical_cols = [col for col in self.feature_cols if df[col].dtype == 'object' or col == 'quarter']
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.encoder.fit(self.df[categorical_cols])

        encoded_cat = self.encoder.transform(self.df[categorical_cols])
        numeric_cols = [col for col in self.feature_cols if col not in categorical_cols]
        X_numeric = self.df[numeric_cols].values if numeric_cols else np.empty((len(self.df), 0))

        self.X = np.hstack([encoded_cat, X_numeric])
        self.y = self.df["roi"].values

        self.model = GradientBoostingRegressor(n_estimators=100)
        self.model.fit(self.X, self.y)

        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

    def predict_roi(self, feature_row: Dict[str, float]) -> float:
        cat_values = [[feature_row.get(col, "") for col in self.categorical_cols]]
        encoded_cat = self.encoder.transform(cat_values)
        numeric_values = np.array([[feature_row.get(col, 0.0) for col in self.numeric_cols]])
        X = np.hstack([encoded_cat, numeric_values])
        return self.model.predict(X)[0]

    def build_roi_function(self, segments: List[str], default_values: Dict[str, float] = {}) -> Callable[[np.ndarray], np.ndarray]:
        def roi_func(budgets: np.ndarray) -> np.ndarray:
            rois = []
            for i, b in enumerate(budgets):
                feature_row = {"segment": segments[i], "budget": b, **default_values}
                roi = self.predict_roi(feature_row)
                rois.append(roi)
            return np.array(rois)
        return roi_func

class CEMAllocator:
    def __init__(self, roi_func: Callable[[np.ndarray], np.ndarray], total_budget: float, segments: list):
        self.roi_func = roi_func
        self.total_budget = total_budget
        self.segments = segments
        self.n = len(segments)

    def optimize(self, num_samples=1000, elite_frac=0.2, max_iters=50):
        mu = np.ones(self.n) * (self.total_budget / self.n)
        sigma = np.ones(self.n) * (self.total_budget / (2 * self.n))

        for _ in range(max_iters):
            samples = np.random.normal(mu, sigma, size=(num_samples, self.n))
            samples = np.clip(samples, 0, self.total_budget)
            samples /= samples.sum(axis=1, keepdims=True)
            samples *= self.total_budget

            rewards = []
            for sample in samples:
                roi = self.roi_func(sample)
                rewards.append((roi * sample).sum())

            elite_idxs = np.argsort(rewards)[-int(num_samples * elite_frac):]
            elite_samples = samples[elite_idxs]
            mu = elite_samples.mean(axis=0)
            sigma = elite_samples.std(axis=0)

        return pd.DataFrame({
            "segment": self.segments,
            "budget": mu,
            "expected_roi": self.roi_func(mu),
            "expected_value": mu * self.roi_func(mu)
        })

# 示例使用:
# df = pd.DataFrame(...)  # 包含 segment, platform, geo, budget, roi, time 等字段
# model = ROIModelGBDT(df, feature_cols=["segment", "platform", "geo", "budget"], time_col="time")
# roi_func = model.build_roi_function(segments=model.df["segment"].unique().tolist())
# optimizer = CEMAllocator(roi_func, total_budget=10000, segments=model.df["segment"].unique().tolist())
# result = optimizer.optimize()
# print(result.sort_values("budget", ascending=False))
