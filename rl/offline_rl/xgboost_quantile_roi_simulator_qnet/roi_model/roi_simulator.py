import numpy as np
import xgboost as xgb

class ROISimulator:
    def __init__(self, model_dict, feature_columns):
        self.models = model_dict
        self.features = feature_columns

    def predict_quantiles(self, df):
        X = df[self.features].values
        dmatrix = xgb.DMatrix(X)
        p10 = self.models['p10'].predict(dmatrix)
        p50 = self.models['p50'].predict(dmatrix)
        p90 = self.models['p90'].predict(dmatrix)
        return p10, p50, p90

    def sample_reward(self, df):
        p10, p50, p90 = self.predict_quantiles(df)
        std = (p90 - p10) / 2.56  # approximate std
        return np.clip(np.random.normal(p50, std), 0, None)
    
