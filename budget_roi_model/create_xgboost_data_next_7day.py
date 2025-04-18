

def construct_xgb_dataset(df, window_size=7, horizon=7, min_spend_threshold=1.0):
    """
    将时序 df 转换为适用于 XGBoost 的监督学习格式，包含窗口、未来 ROI（基于未来注册数），
    并编码时间和 segment_id 特征。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 创建未来注册数 ROI 目标列（未来 horizon 天注册和）
    # 创建 未来注册 和未来 spend 滚动和
    df['future_registers'] = (
        df.groupby('segment_id')['registers']
          .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
    )
    df['future_spend'] = (
        df.groupby('segment_id')['spend']
          .transform(lambda x: x.shift(-horizon + 1).rolling(window=horizon).sum())
    )
    df['roi'] = df['future_registers'] / (df['future_spend'] + 1e-6)

    # 时间特征
    df['dayofweek'] = df['date'].dt.dayofweek / 6.0
    df['weekofyear'] = df['date'].dt.isocalendar().week / 52.0
    df['month'] = df['date'].dt.month / 12.0

    # 滑动窗口构造样本
    samples = []
    segment_ids = df['segment_id'].unique()
    for seg_id in segment_ids:
        df_seg = df[df['segment_id'] == seg_id].sort_values('date')
        for i in range(len(df_seg) - window_size - horizon):
            window_df = df_seg.iloc[i:i + window_size]
            future_roi = df_seg.iloc[i + window_size + horizon - 1]['roi']

            # 跳过没有 target 的行
            if pd.isna(future_roi):
                continue

            # 过滤低 spend 的样本
            total_spend = window_df['spend'].sum()
            if total_spend < min_spend_threshold:
                continue

            # 拼接特征：窗口内的均值、最大值、最后一天的特征等
            feature_vector = {
                'segment_id': seg_id,
                'spend_mean': window_df['spend'].mean(),
                'ctr_mean': window_df['ctr'].mean(),
                'cvr_mean': window_df['cvr'].mean(),
                'spend_last': window_df.iloc[-1]['spend'],
                'ctr_last': window_df.iloc[-1]['ctr'],
                'cvr_last': window_df.iloc[-1]['cvr'],
                'dayofweek_last': window_df.iloc[-1]['dayofweek'],
                'weekofyear_last': window_df.iloc[-1]['weekofyear'],
                'month_last': window_df.iloc[-1]['month'],
            }

            samples.append((feature_vector, future_roi))

    feature_df = pd.DataFrame([x[0] for x in samples])
    target = np.array([x[1] for x in samples])

    # One-hot encode segment_id
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    seg_encoded = ohe.fit_transform(feature_df[['segment_id']])
    seg_feature_names = ohe.get_feature_names_out(['segment_id'])

    feature_df = feature_df.drop(columns=['segment_id'])
    feature_df = pd.concat([feature_df, pd.DataFrame(seg_encoded, columns=seg_feature_names)], axis=1)

    return feature_df, target, ohe
