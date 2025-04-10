import xgboost as xgb

def train_quantile_model(X, y, alpha, params=None):
    if params is None:
        params = {
            "tree_method": "hist",
            "objective": "reg:quantileerror",
            "eval_metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.1,
            "alpha": alpha,  # 关键参数！
        }

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=300)
    return model
