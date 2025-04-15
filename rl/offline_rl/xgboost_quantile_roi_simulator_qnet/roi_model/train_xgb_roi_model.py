

import xgboost as xgb
from sklearn.model_selection import train_test_split

# 训练多分位数模型（p10, p50, p90）
def train_quantile_xgb(X, y):
    models = {}
    for quantile in [0.1, 0.5, 0.9]:
        params = {
            "objective": "reg:quantileerror",
            "eval_metric": "mae",
            "alpha": quantile,
            "max_depth": 5,
            "eta": 0.1
        }
        model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=100)
        models[f"p{int(quantile * 100)}"] = model
    return models
