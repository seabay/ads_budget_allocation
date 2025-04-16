import pandas as pd
import numpy as np
np.random.seed(42)

n_segments = 5
feature_df = pd.DataFrame({
    'platform_id': np.random.randint(0, 3, size=n_segments),
    'geo_id': np.random.randint(0, 15, size=n_segments),
    'ctr': np.random.uniform(0.01, 0.2, size=n_segments),
    'cvr': np.random.uniform(0.01, 0.1, size=n_segments),
    'prev_spend': np.random.uniform(10000, 50000, size=n_segments),
    'cumulative_spend': np.random.uniform(100000, 500000, size=n_segments),
    'time_index': [4]*n_segments  # 代表第4季度
})
com_df=pd.DataFrame({'platform_id': range(3)}).merge(pd.DataFrame({'geo_id': range(15)}), how='cross').merge(pd.DataFrame({'channel': range(2)}), how='cross')
segment_df = com_df.merge(feature_df, on=['platform_id', 'geo_id'], how='left').fillna(0)
