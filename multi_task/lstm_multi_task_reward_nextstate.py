
# 使用 PyTorch 实现的多任务 LSTM 模拟器代码框架：它接收一段 (state, action) 序列作为输入，输出两个结果：
# reward: 预测未来 7 天 signup（单值）
# next_state: 预测下一个状态向量（向量）

import torch
import torch.nn as nn

class MultiTaskLSTMEnvModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.hidden_dim = hidden_dim

        # Shared LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Head for reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # scalar reward
        )

        # Head for next state prediction (state delta)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)  # vector next_state
        )

    def forward(self, state_seq, action_seq):
        """
        Args:
            state_seq: [batch, seq_len, state_dim]
            action_seq: [batch, seq_len, action_dim]
        Returns:
            reward_pred: [batch, 1]
            next_state_pred: [batch, state_dim]
        """
        x = torch.cat([state_seq, action_seq], dim=-1)  # [batch, seq_len, state+action]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        last_hidden = lstm_out[:, -1]  # use last timestep output

        reward_pred = self.reward_head(last_hidden)
        next_state_pred = self.state_head(last_hidden)

        return reward_pred, next_state_pred



# Example dimensions
state_dim = 45
action_dim = 45
seq_len = 5
batch_size = 32

# Fake input
state_seq = torch.randn(batch_size, seq_len, state_dim)
action_seq = torch.randn(batch_size, seq_len, action_dim)

# Model
model = MultiTaskLSTMEnvModel(state_dim, action_dim)
reward_pred, next_state_pred = model(state_seq, action_seq)

print("reward_pred shape:", reward_pred.shape)         # [batch_size, 1]
print("next_state_pred shape:", next_state_pred.shape) # [batch_size, state_dim]


###########################

# MultiTaskLSTMEnvModel 的 训练代码框架，用于训练该模型预测 reward 和 next_state。
# 我们假设你已有日志数据 (state_seq, action_seq, reward, next_state)

def multitask_loss(reward_true, reward_pred, state_next_true, state_next_pred, λ=1.0):
    loss_r = nn.MSELoss()(reward_pred, reward_true)
    loss_s = nn.MSELoss()(state_next_pred, state_next_true)
    return loss_r + λ * loss_s


from torch.utils.data import Dataset, DataLoader

class BudgetSequenceDataset(Dataset):
    def __init__(self, state_seqs, action_seqs, rewards, next_states):
        """
        state_seqs: [N, seq_len, state_dim]
        action_seqs: [N, seq_len, action_dim]
        rewards: [N, 1]
        next_states: [N, state_dim]
        """
        self.state_seqs = state_seqs
        self.action_seqs = action_seqs
        self.rewards = rewards
        self.next_states = next_states

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return (
            self.state_seqs[idx],
            self.action_seqs[idx],
            self.rewards[idx],
            self.next_states[idx]
        )


def train_lstm_env_model(
    model,
    dataloader,
    num_epochs=20,
    lr=1e-3,
    lambda_weight=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for state_seq, action_seq, reward, next_state in dataloader:
            state_seq = state_seq.to(device).float()
            action_seq = action_seq.to(device).float()
            reward = reward.to(device).float()
            next_state = next_state.to(device).float()

            pred_reward, pred_next_state = model(state_seq, action_seq)

            loss_r = loss_fn(pred_reward, reward)
            loss_s = loss_fn(pred_next_state, next_state)
            loss = loss_r + lambda_weight * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")


# 使用示例（带模拟数据）
# Hyperparams
state_dim = 45
action_dim = 45
seq_len = 5
n_samples = 1000
batch_size = 32

# Fake dataset
state_seqs = torch.randn(n_samples, seq_len, state_dim)
action_seqs = torch.randn(n_samples, seq_len, action_dim)
rewards = torch.randn(n_samples, 1)
next_states = torch.randn(n_samples, state_dim)

# Dataloader
dataset = BudgetSequenceDataset(state_seqs, action_seqs, rewards, next_states)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train
model = MultiTaskLSTMEnvModel(state_dim, action_dim)
train_lstm_env_model(model, loader, num_epochs=10)


###########################

# 如何用 训练好的 LSTM 模拟器 来做 offline RL 中的 rollout —— 即：从一个当前状态 s_t 和策略生成的动作 a_t 出发，
# 模拟得到 (s_t, a_t, r_t, s_{t+1}) 四元组。这个过程常用于：
# * 训练 Q 网络
# * 蒙特卡洛评估一个策略
# * 生成额外合成数据（model-based offline RL）



def rollout_simulated_transition(
    model,         # trained LSTM environment model
    policy,        # a policy: takes state_seq and returns action
    state_seq,     # [seq_len, state_dim]
    action_seq,    # [seq_len, action_dim]
    device="cpu"
):
    """
    Perform a 1-step rollout from LSTM-based environment.
    Returns: (s_t, a_t, r_t, s_{t+1})
    """
    model.eval()
    with torch.no_grad():
        # Convert to batch size = 1
        state_seq_tensor = torch.tensor(state_seq).unsqueeze(0).float().to(device)  # [1, seq_len, state_dim]
        action_seq_tensor = torch.tensor(action_seq).unsqueeze(0).float().to(device)  # [1, seq_len, action_dim]

        # Generate new action from current state (e.g. latest state in sequence)
        current_state = state_seq[-1]  # [state_dim]
        a_t = policy(current_state)  # action at t (should be [action_dim])

        # Append a_t to action_seq, and update state_seq (e.g. assume current state is repeated)
        new_state_seq = torch.cat([state_seq_tensor[:, 1:], state_seq_tensor[:, -1:].clone()], dim=1)
        new_action_seq = torch.cat([action_seq_tensor[:, 1:], torch.tensor(a_t).view(1, 1, -1).to(device)], dim=1)

        # Predict reward and next state using simulator
        r_pred, s_next_pred = model(new_state_seq, new_action_seq)

        return current_state, a_t, r_pred.squeeze(0).cpu().numpy(), s_next_pred.squeeze(0).cpu().numpy()


# 假设当前历史窗口长度为 5
seq_len = 5
state_dim = 45
action_dim = 45

# 假历史轨迹
state_seq = torch.randn(seq_len, state_dim)
action_seq = torch.randn(seq_len, action_dim)

# 假策略（随机）
def random_policy(state):
    return torch.randn(action_dim).numpy()

# 用训练好的模型 rollout
s_t, a_t, r_t, s_tp1 = rollout_simulated_transition(
    model=model,  # 训练好的 MultiTaskLSTMEnvModel
    policy=random_policy,
    state_seq=state_seq,
    action_seq=action_seq
)

print("s_t shape:", s_t.shape)
print("a_t shape:", a_t.shape)
print("r_t:", r_t)
print("s_{t+1} shape:", s_tp1.shape)


###########################

# Rollout 数据接入 CQL 模型的训练流程

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 数据集构建：将 rollout (s, a, r, s') 转为可训练格式
# 生成模拟数据集 (可以替换为 rollout 的真实数据)
def generate_synthetic_dataset(n_samples=1000, state_dim=45, action_dim=45):
    dataset = []
    for _ in range(n_samples):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = np.random.rand(1).astype(np.float32)
        s_next = s + np.random.normal(0, 0.1, size=state_dim).astype(np.float32)
        dataset.append((s, a, r, s_next))
    return dataset

# 自定义 PyTorch Dataset
class TransitionDataset(Dataset):
    def __init__(self, data):
        self.states = torch.tensor([d[0] for d in data])
        self.actions = torch.tensor([d[1] for d in data])
        self.rewards = torch.tensor([d[2] for d in data])
        self.next_states = torch.tensor([d[3] for d in data])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
        )

# 构建数据
synthetic_data = generate_synthetic_dataset()
dataset = TransitionDataset(synthetic_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 🧠 2. 接入 CQL 或任何 Offline RL 算法
for epoch in range(num_epochs):
    for s, a, r, s_next in dataloader:
        # 输入到 Offline RL 算法
        # cql.train_step(s, a, r, s_next) 或者其他算法
        pass


###########################

