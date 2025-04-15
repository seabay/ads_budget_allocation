
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class OfflineRLTrainer:
    def __init__(self, q_net, simulator, dataset, input_dim, num_actions, gamma=0.99, lr=1e-3):
        self.q_net = q_net
        self.target_net = QNetwork(input_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.simulator = simulator
        self.dataset = dataset
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def compute_td_loss(self, s, a, r, s_next, done):
        q_values = self.q_net(s).gather(1, a)
        with torch.no_grad():
            max_next_q = self.target_net(s_next).max(dim=1, keepdim=True)[0]
            td_target = r + self.gamma * (1 - done) * max_next_q
        return F.mse_loss(q_values, td_target)

    def train(self, num_epochs=100, batch_size=64):
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for batch in loader:
                s, a, s_next, done = batch
                r = self.simulator.sample_reward(s)
                loss = self.compute_td_loss(s, a, torch.tensor(r).unsqueeze(1), s_next, done)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def select_action(self, state):
        with torch.no_grad():
            q_vals = self.q_net(state)
            return torch.argmax(q_vals, dim=1)


