

# platform_id: 平台 ID（例如，1, 2, 3）

# geo_id: 地理区域 ID（例如，101, 102, 103）

# cvr: 转化率

# ctr: 点击率

# prev_spend: 上期的广告支出

# roi_p10: 预测的 P10 ROI

# roi_p50: 预测的 P50 ROI（用于作为奖励）

# roi_p90: 预测的 P90 ROI

# action: 动作，表示广告预算分配（例如，0: 低预算，1: 中预算，2: 高预算）

# reward: 奖励，基于 roi_p50 和延迟的影响计算得到的 ROI 回报

# s_next: 下一状态，用于强化学习训练，包含当前时刻的数据与稍微变化的特征


import pandas as pd
import numpy as np

# 生成随机数据函数，加入时间相关特征
def generate_sample_data_with_time(num_samples=1000):
    np.random.seed(42)

    # 广告平台和地理区域
    platform_ids = [1, 2, 3]  # 三个平台
    geo_ids = [101, 102, 103, 104, 105]  # 五个地理区域

    # 时间特征
    start_date = pd.to_datetime("2020-01-01")
    time_stamps = pd.date_range(start=start_date, periods=num_samples, freq="D")

    # 生成特征数据
    platform = np.random.choice(platform_ids, num_samples)
    geo = np.random.choice(geo_ids, num_samples)

    cvr = np.random.uniform(0.01, 0.1, num_samples)  # 转化率
    ctr = np.random.uniform(0.02, 0.15, num_samples)  # 点击率
    prev_spend = np.random.uniform(100, 1000, num_samples)  # 上期支出
    roi_p10 = np.random.uniform(0.5, 2.0, num_samples)  # 预测的 ROI p10
    roi_p50 = np.random.uniform(1.0, 3.0, num_samples)  # 预测的 ROI p50
    roi_p90 = np.random.uniform(1.5, 4.0, num_samples)  # 预测的 ROI p90

    # 生成动作：预算分配（动作可以是某个特定的比例或数值）
    actions = np.random.choice([0, 1, 2], num_samples)  # 例如 0:低预算，1:中预算，2:高预算

    # 计算奖励：ROI 这里我们考虑延迟，可以使用 `roi_p50` 作为奖励的基准
    # 假设奖励为 ROI * 上期支出，其中会引入一个延迟效应
    roi_delay = np.random.uniform(0.8, 1.2, num_samples)  # 延迟影响
    reward = roi_p50 * prev_spend * roi_delay

    # 时间特征：提取周期性特征
    day_of_week = time_stamps.dayofweek  # 一周中的哪一天（0是周一，6是周日）
    month = time_stamps.month  # 月份
    quarter = time_stamps.quarter  # 季度
    is_weekend = (day_of_week >= 5).astype(int)  # 是否为周末
    is_holiday = time_stamps.isin(pd.to_datetime(['2020-12-25', '2021-01-01']))  # 假设的假期

    # 时间差：当前时间与前一条数据的时间差
    time_diff = np.diff(np.concatenate([[0], time_stamps.view(np.int64)]))  # 计算时间戳之间的差异

    # 创建时间相关特征
    time_features = pd.DataFrame({
        'time_stamp': time_stamps,
        'day_of_week': day_of_week,
        'month': month,
        'quarter': quarter,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'time_diff': time_diff
    })

    # 生成“历史累计”特征，如累计广告支出和累计 ROI
    cumulative_spend = np.cumsum(prev_spend)  # 累计广告支出
    cumulative_roi_p50 = np.cumsum(roi_p50)  # 累计 ROI p50

    # 生成下一个状态：s_next，简单地模拟下一步的状态
    s_next = pd.DataFrame({
        'platform_id': platform + np.random.choice([-1, 0, 1], num_samples),  # 稍微变化平台
        'geo_id': geo + np.random.choice([-1, 0, 1], num_samples),  # 稍微变化地理区域
        'cvr': cvr + np.random.uniform(-0.01, 0.01, num_samples),  # 稍微变化转化率
        'ctr': ctr + np.random.uniform(-0.01, 0.01, num_samples),  # 稍微变化点击率
        'prev_spend': prev_spend + np.random.uniform(-50, 50, num_samples),  # 上期支出的轻微变化
        'roi_p10': roi_p10 + np.random.uniform(-0.1, 0.1, num_samples),  # ROI p10 变化
        'roi_p50': roi_p50 + np.random.uniform(-0.1, 0.1, num_samples),  # ROI p50 变化
        'roi_p90': roi_p90 + np.random.uniform(-0.1, 0.1, num_samples)  # ROI p90 变化
    })

    # 创建数据集
    data = pd.DataFrame({
        'platform_id': platform,
        'geo_id': geo,
        'cvr': cvr,
        'ctr': ctr,
        'prev_spend': prev_spend,
        'roi_p10': roi_p10,
        'roi_p50': roi_p50,
        'roi_p90': roi_p90,
        'action': actions,
        'reward': reward,
        'cumulative_spend': cumulative_spend,
        'cumulative_roi_p50': cumulative_roi_p50
    })

    # 合并时间相关特征
    data = pd.concat([data, time_features], axis=1)

    # 添加下一个状态 (s_next)
    data['s_next'] = s_next.apply(lambda row: row.tolist(), axis=1)

    return data

# 生成 1000 条样本数据
offline_data_with_time = generate_sample_data_with_time(num_samples=1000)

# 打印前几行查看数据
print(offline_data_with_time.head())


# =============================

# 使用滑动窗口方法来计算cumulative_spend和cumulative_roi_p50可以避免它们无限增大，并且更能反映近期的广告支出和ROI

import numpy as np
import pandas as pd

# 示例数据
data = {
    'prev_spend': np.random.rand(100) * 1000,  # 假设每个时间点的广告支出是随机的
    'roi_p50': np.random.rand(100) * 5,  # 假设每个时间点的 ROI p50 是随机的
}

df = pd.DataFrame(data)

# 定义滑动窗口大小
window_size = 30  # 例如：30天

# 计算滑动窗口内的累计广告支出
df['cumulative_spend'] = np.array([np.sum(df['prev_spend'][max(0, i-window_size+1):i+1]) for i in range(len(df))])

# 计算滑动窗口内的累计 ROI p50
df['cumulative_roi_p50'] = np.array([np.sum(df['roi_p50'][max(0, i-window_size+1):i+1]) for i in range(len(df))])

# 查看结果
print(df.head(40))


