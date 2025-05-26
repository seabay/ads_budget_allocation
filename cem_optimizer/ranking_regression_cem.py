

# Ranking + Regression + CEM 预算优化框架模板

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

# =====================================
# Step 1: 数据准备
# =====================================
def prepare_data(df):
    # df: 包含 features, 'segment_id', 'budget', 'signup'
    df = df.copy()
    df['log1p_signup'] = np.log1p(df['signup'])
    return df

# =====================================
# Step 2: Ranking 模型训练（pairwise）
# =====================================
def train_ranking_model(df):
    # 用 log1p(signup) 的排序信息作为 ranking label
    df = df.copy()
    df['rank'] = df.groupby('batch_id')['log1p_signup'].transform(lambda x: rankdata(-x))
    X = df.drop(columns=['signup', 'log1p_signup', 'rank', 'batch_id'])
    y = df['rank']
    group = df.groupby('batch_id').size().to_list()

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)

    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'max_depth': 6,
        'eval_metric': 'ndcg'
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

# =====================================
# Step 3: Regression 模型训练（预测 Signup）
# =====================================
def train_regression_model(df):
    X = df.drop(columns=['signup', 'log1p_signup', 'batch_id'])
    y = df['signup']
    dtrain = xgb.DMatrix(X, label=y)

    params = {
        'objective': 'reg:squarederror',
        'eta': 0.05,
        'max_depth': 6,
        'eval_metric': 'rmse'
    }
    model = xgb.train(params, dtrain, num_boost_round=200)
    return model

# =====================================
# Step 4: 组合打分 + CEM 优化
# =====================================
def cem_optimize(df_candidates, ranking_model, regression_model, total_budget, n_iter=30, pop_size=100, elite_frac=0.2):
    X = df_candidates.drop(columns=['segment_id', 'budget'])
    dmatrix = xgb.DMatrix(X)

    # 组合 score（ranking + regression）
    rank_scores = ranking_model.predict(dmatrix)
    signup_preds = regression_model.predict(dmatrix)

    combined_score = 0.7 * rank_scores + 0.3 * signup_preds

    segment_ids = df_candidates['segment_id'].values
    budgets = df_candidates['budget'].values
    score_lookup = dict(zip(zip(segment_ids, budgets), combined_score))

    # 初始化分布（均匀）
    segment_list = sorted(df_candidates['segment_id'].unique())
    n_segments = len(segment_list)

    mu = np.ones(n_segments) * total_budget / n_segments
    sigma = np.ones(n_segments) * (total_budget / (2 * n_segments))

    for _ in range(n_iter):
        samples = np.random.normal(mu, sigma, size=(pop_size, n_segments))
        samples = np.clip(samples, 0, total_budget)
        samples = samples / samples.sum(axis=1, keepdims=True) * total_budget

        scores = []
        for alloc in samples:
            score = 0
            for i, segment in enumerate(segment_list):
                b = alloc[i]
                b_rounded = int(np.round(b))
                key = (segment, b_rounded)
                if key in score_lookup:
                    score += score_lookup[key]
            scores.append(score)

        elite_idxs = np.argsort(scores)[-int(pop_size * elite_frac):]
        elite_samples = samples[elite_idxs]

        mu = elite_samples.mean(axis=0)
        sigma = elite_samples.std(axis=0)

    final_alloc = dict(zip(segment_list, mu))
    return final_alloc

# =====================================
# 用法示例
# =====================================
# df = pd.read_csv("your_training_data.csv")
# df = prepare_data(df)
# ranking_model = train_ranking_model(df)
# regression_model = train_regression_model(df)

# df_candidates = pd.read_csv("candidate_segment_budget_pairs.csv")
# alloc = cem_optimize(df_candidates, ranking_model, regression_model, total_budget=100000)
# print(alloc)
