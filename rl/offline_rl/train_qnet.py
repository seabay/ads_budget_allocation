

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_q_network(
    q_net,
    target_net,
    dataset,  # assumed to be a Pandas DataFrame
    scaler,   # StandardScaler fitted on dataset
    num_epochs=100,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    target_update_freq=10,
    log_every=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Define optimizer
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    q_net.to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.to(device)
    target_net.eval()

    # Prepare training tensors
    features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']
    X = scaler.transform(dataset[features].astype(float))
    s = torch.tensor(X, dtype=torch.float32)
    a = torch.tensor(dataset['action'].values, dtype=torch.int64).unsqueeze(1)
    r = torch.tensor(dataset['reward'].values, dtype=torch.float32).unsqueeze(1)
    done = torch.tensor(dataset['done'].values, dtype=torch.float32).unsqueeze(1)
    
    # s_next 默认使用下一时间步的状态，offline 模型可用相邻 sample 模拟（或真实 next_state）
    if 's_next' in dataset.columns:
        X_next = scaler.transform(dataset['s_next'].tolist())  # shape: (N, feature_dim)
        s_next = torch.tensor(X_next, dtype=torch.float32)
    else:
        s_next = s  # fallback
    
    dataset_tensor = TensorDataset(s, a, r, s_next, done)
    loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        q_net.train()
        total_loss = 0
        for batch in loader:
            s_batch, a_batch, r_batch, s_next_batch, done_batch = [x.to(device) for x in batch]
            loss = compute_td_loss(q_net, target_net, s_batch, a_batch, r_batch, s_next_batch, done_batch, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epoch % log_every == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss / len(loader):.4f}")

    # Save model
    torch.save(q_net, "trained_q_model.pt")
    print("Training complete and model saved.")


====================================================

import numpy as np
import pandas as pd

def generate_offline_rl_data(num_samples=1000, num_actions=5, seed=42):
    np.random.seed(seed)
    data = []

    for _ in range(num_samples):
        roi_p10 = np.random.uniform(0.2, 0.5)
        roi_p90 = np.random.uniform(1.0, 2.0)
        roi_p50 = np.random.uniform(roi_p10, roi_p90)
        cvr = np.random.uniform(0.01, 0.2)
        ctr = np.random.uniform(0.01, 0.5)
        prev_spend = np.random.uniform(100, 10000)
        platform_id = np.random.randint(0, 3)
        geo_id = np.random.randint(0, 15)

        # Simulate reward and action
        true_q = roi_p50 + 0.5 * ctr + 0.3 * cvr
        best_action = np.random.randint(0, num_actions)  # You can set policy here
        reward = true_q * prev_spend  # e.g., ROI × spend

        data.append({
            'roi_p10': roi_p10,
            'roi_p50': roi_p50,
            'roi_p90': roi_p90,
            'cvr': cvr,
            'ctr': ctr,
            'prev_spend': prev_spend,
            'platform_id': platform_id,
            'geo_id': geo_id,
            'action': best_action,
            'reward': reward,
            'done': 0
        })

    df = pd.DataFrame(data)

    # Optional: add simulated next state (copy with noise)
    next_states = df[['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']].copy()
    next_states += np.random.normal(0, 0.01, size=next_states.shape)
    df['s_next'] = next_states.values.tolist()

    return df


df = generate_offline_rl_data()


