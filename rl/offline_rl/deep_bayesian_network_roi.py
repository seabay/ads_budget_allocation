

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 1. 定义贝叶斯神经网络
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(BayesianNN, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, 1)
        self.fc_log_std = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)  # 保证标准差是正数
        return mean, std

# 2. 定义 Pyro 的概率模型
def model(x_data, y_data):
    input_dim = x_data.shape[1]
    nn = BayesianNN(input_dim)

    # 对每个数据点进行建模
    with pyro.plate("data", x_data.shape[0]):
        mean, std = nn(x_data)
        # 假设 ROI 是从正态分布中生成
        obs = pyro.sample("obs", dist.Normal(mean, std), obs=y_data)
        
# 3. 定义推理模型（变分推理）
def guide(x_data, y_data):
    input_dim = x_data.shape[1]
    nn = BayesianNN(input_dim)

    # 通过变分分布来近似 posterior
    mean, std = nn(x_data)
    # 使用均值和标准差作为变分分布的参数
    pyro.sample("obs", dist.Normal(mean, std))

# 4. 训练模型
def train(model, guide, x_data, y_data, num_steps=5000):
    # 使用 Adam 优化器和变分推理
    adam = optim.Adam({"lr": 1e-3})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for step in range(num_steps):
        loss = svi.step(x_data, y_data)
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss}")

# 5. 计算 P10, P50, P90
def calculate_percentiles(model, x_data, percentiles=[10, 50, 90]):
    with torch.no_grad():
        mean, std = model(x_data)
        roi_distribution = dist.Normal(mean, std)
        
        # 计算给定百分位的值
        percentiles_values = {}
        for p in percentiles:
            percentile_value = roi_distribution.icdf(torch.tensor(p / 100.0))
            percentiles_values[f'roi_p{p}'] = percentile_value.item()
            
    return percentiles_values

# 6. 数据加载和预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaler.fit(df[['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']])
    return df, scaler

# 7. 主程序
if __name__ == "__main__":
    # 加载数据并进行标准化
    file_path = "ad_budget_data.csv"
    df, scaler = load_data(file_path)
    
    # 选择特征列（假设使用所有特征）
    features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']
    X = df[features].values
    y = df['roi_p50'].values  # 使用 P50 作为目标变量

    # 将数据转换为 PyTorch 张量
    x_data = torch.tensor(scaler.transform(X), dtype=torch.float32)
    y_data = torch.tensor(y, dtype=torch.float32)

    # 训练模型
    train(model, guide, x_data, y_data)

    # 计算 P10, P50, P90
    percentiles = calculate_percentiles(model, x_data)
    print("Calculated Percentiles:", percentiles)
