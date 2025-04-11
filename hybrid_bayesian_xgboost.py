
# 1. GBDT ROI 模型
# 输入：segment + budget + 时间/平台/地理等特征

# 输出：预测 ROI 点估计 μ_gbdt

# 2. 构建 Beta 或 LogNormal 分布的先验
# 使用 μ_gbdt 构造 LogNormal / Beta 分布的先验参数（例如，μ, σ）

# 3. Bayesian 后验更新（可选 SVI / MCMC）
# 使用历史 roi 数据与 GBDT 输出更新为后验分布

# 4. 采样 ROI（Thompson Sampling）
# 从后验中采样每个 segment 的 ROI 值

# 用采样值代入 ROI 函数后进行 CEM 分配（或直接分配)


# 使用 GBDT 预测结果设定 LogNormal 先验：
# mu = log(μ_gbdt^2 / sqrt(σ^2 + μ_gbdt^2))
# sigma = sqrt(log(1 + σ^2 / μ_gbdt^2))

# 或设为 Beta 分布的先验：
# α = μ * ((μ*(1-μ)) / σ^2 - 1)
# β = (1-μ) * ((μ*(1-μ)) / σ^2 - 1)


import torch
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import holidays
from torch.distributions import LogNormal, Beta


def estimate_lognormal_params(mu, sigma):
    variance = sigma**2
    log_mu = np.log(mu**2 / np.sqrt(variance + mu**2))
    log_sigma = np.sqrt(np.log(1 + variance / mu**2))
    return log_mu, log_sigma

def estimate_beta_params(mu, sigma):
    epsilon = 1e-6
    mu = np.clip(mu, epsilon, 1 - epsilon)
    variance = sigma**2
    alpha = mu * ((mu * (1 - mu)) / variance - 1)
    beta = (1 - mu) * ((mu * (1 - mu)) / variance - 1)
    return alpha, beta


class ROIModelGBDT:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str] = ["segment", "budget"], time_col: str = "time", history_window: int = 3):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.history_window = history_window

        if time_col in df.columns:
            self.df = self.df.sort_values(by=["segment", time_col])
            for i in range(1, history_window + 1):
                self.df[f"roi_lag_{i}"] = self.df.groupby("segment")["roi"].shift(i)
            self.df[f"roi_ma_{history_window}"] = self.df.groupby("segment")["roi"].transform(lambda x: x.shift(1).rolling(history_window).mean())

            self.df["quarter"] = pd.to_datetime(self.df[time_col]).dt.quarter.astype(str)
            us_holidays = holidays.UnitedStates()
            self.df["is_holiday"] = pd.to_datetime(self.df[time_col]).isin(us_holidays).astype(int)
            self.df["time_diff"] = self.df.groupby("segment")[time_col].transform(lambda x: pd.to_datetime(x).diff().dt.days.fillna(0))

            self.df = self.df.dropna().reset_index(drop=True)
            self.feature_cols += [f"roi_lag_{i}" for i in range(1, history_window + 1)]
            self.feature_cols.append(f"roi_ma_{history_window}")
            self.feature_cols += ["quarter", "is_holiday", "time_diff"]

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

    def predict_std(self, feature_row: Dict[str, float]) -> float:
        return 0.1 * self.predict_roi(feature_row)  # placeholder

    def build_roi_distributions(self, segments: List[str], default_values: Dict[str, float], dist_type: str = 'lognormal') -> Dict[str, torch.distributions.Distribution]:
        distributions = {}
        for seg in segments:
            feature_row = {"segment": seg, **default_values}
            mu = self.predict_roi(feature_row)
            sigma = self.predict_std(feature_row)
            if dist_type == 'lognormal':
                log_mu, log_sigma = estimate_lognormal_params(mu, sigma)
                dist = LogNormal(torch.tensor(log_mu), torch.tensor(log_sigma))
            elif dist_type == 'beta':
                alpha, beta = estimate_beta_params(mu, sigma)
                dist = Beta(torch.tensor(alpha), torch.tensor(beta))
            else:
                raise ValueError("Unknown dist_type")
            distributions[seg] = dist
        return distributions


class HybridBayesianAllocator:
    def __init__(self, roi_model: ROIModelGBDT, segments: List[str], total_budget: float, default_values: Dict[str, float], dist_type: str = 'lognormal'):
        self.roi_model = roi_model
        self.segments = segments
        self.total_budget = total_budget
        self.default_values = default_values
        self.dist_type = dist_type

    def sample_allocation(self, num_samples=1000):
        dists = self.roi_model.build_roi_distributions(self.segments, self.default_values, self.dist_type)
        samples = []
        for _ in range(num_samples):
            roi_samples = np.array([dists[seg].sample().item() for seg in self.segments])
            alloc = roi_samples / roi_samples.sum() * self.total_budget
            samples.append(alloc)
        samples = np.array(samples)
        mean_alloc = samples.mean(axis=0)
        return pd.DataFrame({
            "segment": self.segments,
            "budget": mean_alloc
        })

# 示例使用:
# df = pd.DataFrame(...)  # 包含 segment, platform, geo, budget, roi, time 等字段
# model = ROIModelGBDT(df, feature_cols=["segment", "platform", "geo", "budget"], time_col="time")
# allocator = HybridBayesianAllocator(model, segments=model.df["segment"].unique().tolist(), total_budget=10000, default_values={}, dist_type='lognormal')
# result = allocator.sample_allocation()
# print(result.sort_values("budget", ascending=False))
