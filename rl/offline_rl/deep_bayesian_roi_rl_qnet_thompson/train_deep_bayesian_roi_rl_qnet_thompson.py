

# Hybrid ROI Prediction + Uncertainty-Aware RL Pipeline (Prototype)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyro
import pyro.distributions as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


### ROI Predictor Using Bayesian Regression (Pyro)
class BayesianROINet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def model(x_data, y_data):
    fc1w_prior = dist.Normal(0, 1).expand([64, x_data.size(1)]).to_event(2)
    fc1b_prior = dist.Normal(0, 1).expand([64]).to_event(1)
    fc2w_prior = dist.Normal(0, 1).expand([1, 64]).to_event(2)
    fc2b_prior = dist.Normal(0, 1).expand([1]).to_event(1)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

    lifted_module = pyro.random_module("module", BayesianROINet(x_data.size(1)), priors)
    lifted_reg_model = lifted_module()

    with pyro.plate("map", len(x_data)):
        prediction_mean = lifted_reg_model(x_data)
        pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y_data)

def guide(x_data, y_data):
    fc1w_mu = pyro.param("fc1w_mu", torch.randn(64, x_data.size(1)))
    fc1w_sigma = pyro.param("fc1w_sigma", torch.ones(64, x_data.size(1)), constraint=dist.constraints.positive)
    fc1b_mu = pyro.param("fc1b_mu", torch.randn(64))
    fc1b_sigma = pyro.param("fc1b_sigma", torch.ones(64), constraint=dist.constraints.positive)
    fc2w_mu = pyro.param("fc2w_mu", torch.randn(1, 64))
    fc2w_sigma = pyro.param("fc2w_sigma", torch.ones(1, 64), constraint=dist.constraints.positive)
    fc2b_mu = pyro.param("fc2b_mu", torch.randn(1))
    fc2b_sigma = pyro.param("fc2b_sigma", torch.ones(1), constraint=dist.constraints.positive)

    dists = {
        "fc1.weight": dist.Normal(fc1w_mu, fc1w_sigma).to_event(2),
        "fc1.bias": dist.Normal(fc1b_mu, fc1b_sigma).to_event(1),
        "fc2.weight": dist.Normal(fc2w_mu, fc2w_sigma).to_event(2),
        "fc2.bias": dist.Normal(fc2b_mu, fc2b_sigma).to_event(1),
    }
    return pyro.random_module("module", BayesianROINet(x_data.size(1)), dists)()


### RL Agent with Thompson Sampling from Ensemble
class QNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.q(x)

class RLPolicy:
    def __init__(self, q_ensemble):
        self.q_ensemble = q_ensemble

    def select_action(self, state_tensor):
        sampled_net = np.random.choice(self.q_ensemble)
        q_vals = sampled_net(state_tensor)
        return torch.argmax(q_vals, dim=1)


### Pipeline Training Example (Mock Data)
if __name__ == "__main__":
    # Generate synthetic data
    X = np.random.rand(1000, 6)
    y = (X[:, 0] * 0.5 + X[:, 1] * 0.8 + np.random.randn(1000) * 0.05)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Train Bayesian ROI Predictor
    pyro.clear_param_store()
    svi = pyro.infer.SVI(model, guide, pyro.optim.Adam({"lr": 1e-3}), loss=pyro.infer.Trace_ELBO())
    for step in range(1000):
        loss = svi.step(X_tensor, y_tensor)
        if step % 100 == 0:
            print(f"[Step {step}] Loss: {loss:.4f}")

    # Sample ROI distributions
    posterior_samples = [guide(X_tensor, y_tensor) for _ in range(30)]
    preds = torch.stack([m(X_tensor).detach() for m in posterior_samples])
    roi_p10 = torch.quantile(preds, 0.1, dim=0)
    roi_p50 = torch.quantile(preds, 0.5, dim=0)
    roi_p90 = torch.quantile(preds, 0.9, dim=0)

    # Prepare RL state
    rl_state = torch.cat([X_tensor, roi_p10.unsqueeze(1), roi_p50.unsqueeze(1), roi_p90.unsqueeze(1)], dim=1)

    # Train Q-function ensemble
    q_ensemble = []
    for _ in range(5):
        q_net = QNet(rl_state.shape[1], action_dim=5)
        optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
        for _ in range(300):
            q_val = q_net(rl_state)
            target = torch.randn_like(q_val)  # mock TD target
            loss = nn.MSELoss()(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        q_ensemble.append(q_net)

    # Use Thompson sampling policy
    policy = RLPolicy(q_ensemble)
    actions = policy.select_action(rl_state)
    print("Sample actions:", actions[:10])

