

import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from roi_model.roi_simulator import ROISimulator
from rl_policy.q_network import QNetwork
from rl_policy.trainer import OfflineRLTrainer
import torch

# Load offline dataset
data = pd.read_csv("data/offline_logs.csv")
features = ['ctr', 'cvr', 'prev_spend', 'platform_id', 'geo_id']

# Load ROI model and build simulator
models = {f"p{q}": xgb.Booster()
          for q in [10, 50, 90]}
for q in models:
    models[q].load_model(f"models/roi_{q}.json")
simulator = ROISimulator(models, features)

# Prepare dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])
s = torch.tensor(X_scaled, dtype=torch.float32)
a = torch.tensor(data['action'].values, dtype=torch.int64).unsqueeze(1)
s_next = s  # Simplified
end = torch.tensor(data['done'].values, dtype=torch.float32).unsqueeze(1)
dataset = torch.utils.data.TensorDataset(s, a, s_next, end)

# Train RL
q_net = QNetwork(input_dim=s.shape[1], output_dim=data['action'].nunique())
trainer = OfflineRLTrainer(q_net, simulator, dataset, input_dim=s.shape[1], num_actions=data['action'].nunique())
trainer.train()

# Save model
torch.save(q_net.state_dict(), "models/q_net.pt")

