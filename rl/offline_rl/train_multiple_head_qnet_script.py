

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 1. 创建数据集
class BudgetAllocationDataset(Dataset):
    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler
        self.features = [
            'roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id'
        ]
        self.labels = 'allocated_budget'  # 假设目标是优化预算分配

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[self.features].values
        label = row[self.labels]
        features_scaled = self.scaler.transform([features])  # 归一化
        return torch.tensor(features_scaled, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# 2. 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaler.fit(df[['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']])
    return df, scaler

# 3. 定义训练函数
def train(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # 获取模型输出
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets.unsqueeze(1))  # 目标是一个单一值，因此需要增加维度
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

# 4. 主程序
if __name__ == "__main__":
    # 假设数据文件路径为 "ad_budget_data.csv"
    file_path = "ad_budget_data.csv"

    # 加载数据并进行标准化
    df, scaler = load_data(file_path)

    # 创建数据集和数据加载器
    dataset = BudgetAllocationDataset(df, scaler)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 初始化多头 Q 网络模型
    model = MultiHeadQNetwork(input_dim=8, output_dim=1, num_heads=10, hidden_dim=64)
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # 使用均方误差损失函数

    # 训练模型
    train(model, train_loader, optimizer, criterion, epochs=20)

    # 保存训练后的模型
    torch.save(model.state_dict(), "trained_multihead_q_model.pth")
