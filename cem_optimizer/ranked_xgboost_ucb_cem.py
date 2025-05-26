

import numpy as np
import pandas as pd
from xgboost import XGBRanker
from scipy.stats import norm

# -------------------------------
# Step 1: 训练 XGBoost Ranking 模型
# -------------------------------
def train_ranking_model(train_df, feature_cols, group_col, label_col):
    model = XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6
    )
    X = train_df[feature_cols]
    y = train_df[label_col]
    group = train_df.groupby(group_col).size().values
    model.fit(X, y, group=group)
    return model

# -------------------------------
# Step 2: 构造评分表 (segment + budget)
# -------------------------------
def build_candidate_table(segments, budgets, feature_generator):
    rows = []
    for segment in segments:
        for budget in budgets:
            features = feature_generator(segment, budget)
            rows.append({**features, 'segment': segment, 'budget': budget})
    return pd.DataFrame(rows)

# -------------------------------
# Step 3: CEM 优化器
# -------------------------------
def cem_optimize(score_func, segments, budget_space, total_budget, n_iter=10, n_sample=500, elite_frac=0.2):
    dim = len(segments)
    mu = np.array([np.mean(budget_space)] * dim)
    sigma = np.array([np.std(budget_space)] * dim)

    for iteration in range(n_iter):
        samples = np.random.normal(loc=mu, scale=sigma, size=(n_sample, dim))
        samples = np.clip(samples, min(budget_space), max(budget_space))

        # 归一化到总预算
        samples = samples / samples.sum(axis=1, keepdims=True) * total_budget

        scores = np.array([score_func(dict(zip(segments, s))) for s in samples])
        elite_idx = scores.argsort()[-int(n_sample * elite_frac):]
        elite_samples = samples[elite_idx]

        mu = elite_samples.mean(axis=0)
        sigma = elite_samples.std(axis=0)

    return dict(zip(segments, mu))

# -------------------------------
# Step 4: 综合 Pipeline
# -------------------------------
def run_allocation_pipeline(train_df, feature_cols, group_col, label_col, 
                            segments, budgets, total_budget, feature_generator):
    # Step 1: 训练 Ranking 模型
    model = train_ranking_model(train_df, feature_cols, group_col, label_col)

    # Step 2: 构造候选评分表
    df = build_candidate_table(segments, budgets, feature_generator)
    df['score'] = model.predict(df[feature_cols])

    # Step 3: 选出 top-K 组合作为候选
    topk = df.groupby('segment').apply(lambda x: x.nlargest(1, 'score')).reset_index(drop=True)

    # Step 4: 构造 score_func（给定 budget dict 返回总 score）
    score_map = topk.set_index(['segment', 'budget'])['score'].to_dict()
    def score_func(budget_dict):
        return sum([score_map.get((seg, round(bud, -2)), -1e6) for seg, bud in budget_dict.items()])

    # Step 5: CEM 优化器
    allocation = cem_optimize(score_func, segments, budgets, total_budget)
    return allocation

# -------------------------------
# 示例 Feature 构造器（用户可自定义）
# -------------------------------
def example_feature_generator(segment, budget):
    geo, channel = segment.split('_')
    return {
        'budget': budget,
        'geo_id': hash(geo) % 100,
        'channel_id': hash(channel) % 10,
        'geo_channel_interact': (hash(geo) % 100) * (hash(channel) % 10)
    }

# -------------------------------
# 示例调用
# -------------------------------
if __name__ == '__main__':
    # 示例数据（替换成你的训练数据）
    train_df = pd.read_csv('training_data.csv')  # 必须包含 feature_cols + label + group_col
    feature_cols = ['budget', 'geo_id', 'channel_id', 'geo_channel_interact']
    group_col = 'group_id'  # 表示每个 segment + budget 一组
    label_col = 'signup'

    segments = [f'geo{i}_chan{j}' for i in range(10) for j in range(20)]  # 高维 segment
    budgets = list(range(100, 2000, 100))
    total_budget = 50000

    result = run_allocation_pipeline(train_df, feature_cols, group_col, label_col,
                                     segments, budgets, total_budget, example_feature_generator)
    print("Final allocation:", result)
