

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class QTrainer:
    def __init__(self, q_net, target_net, scaler, device=None):
        self.q_net = q_net
        self.target_net = target_net
        self.scaler = scaler
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_net.to(self.device)
        self.target_net.to(self.device)

        # 初始化 optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        # 将 target_net 初始化为 q_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    def compute_td_target(self, r, done, s_next, gamma):
        with torch.no_grad():
            next_q_values = self.target_net(s_next)
            max_next_q = torch.max(next_q_values, dim=1, keepdim=True)[0]
            return r + gamma * (1 - done) * max_next_q

    def compute_td_loss(self, s, a, r, s_next, done, gamma=0.99):
        q_values = self.q_net(s)
        q_value = q_values.gather(1, a)  # shape (batch, 1)
        td_target = self.compute_td_target(r, done, s_next, gamma)
        return F.mse_loss(q_value, td_target)

    def train(self, dataset, num_epochs=100, batch_size=64, gamma=0.99, target_update_freq=10, log_every=10):
        # Prepare training data
        X = self.scaler.transform(dataset[['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']].astype(float))
        s = torch.tensor(X, dtype=torch.float32).to(self.device)
        a = torch.tensor(dataset['action'].values, dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(dataset['reward'].values, dtype=torch.float32).unsqueeze(1).to(self.device)
        done = torch.tensor(dataset['done'].values, dtype=torch.float32).unsqueeze(1).to(self.device)

        if 's_next' in dataset.columns:
            X_next = self.scaler.transform(dataset['s_next'].tolist())
            s_next = torch.tensor(X_next, dtype=torch.float32).to(self.device)
        else:
            s_next = s

        dataset_tensor = TensorDataset(s, a, r, s_next, done)
        loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            self.q_net.train()
            total_loss = 0
            for batch in loader:
                s_batch, a_batch, r_batch, s_next_batch, done_batch = [x.to(self.device) for x in batch]
                loss = self.compute_td_loss(s_batch, a_batch, r_batch, s_next_batch, done_batch, gamma)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Update target network
            if epoch % target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if epoch % log_every == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss / len(loader):.4f}")

        print("Training complete.")
    
    def save_model(self, filepath="trained_q_model.pt"):
        torch.save(self.q_net.state_dict(), filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath="trained_q_model.pt"):
        self.q_net.load_state_dict(torch.load(filepath))
        self.q_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {filepath}.")

    def predict(self, df):
        state_tensor = self.prepare_state(df)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            actions = torch.argmax(q_values, dim=1).numpy()
        return actions

    def prepare_state(self, df):
        features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']
        X = df[features].copy()
        X[['platform_id', 'geo_id']] = X[['platform_id', 'geo_id']].astype(float)
        X_scaled = self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32).to(self.device)


==========================================

# 初始化 Q 网络和 Target 网络
q_net = QNetwork(input_dim=8, output_dim=10)
target_net = QNetwork(input_dim=8, output_dim=10)

# 创建 QTrainer 实例
trainer = QTrainer(q_net=q_net, target_net=target_net, scaler=my_scaler)

# 训练模型
trainer.train(dataset=my_dataset, num_epochs=100, batch_size=64)

# 保存训练后的模型
trainer.save_model(filepath="trained_q_model.pt")

# 加载模型
trainer.load_model(filepath="trained_q_model.pt")

# 使用模型进行预测
predictions = trainer.predict(my_segment_df)


