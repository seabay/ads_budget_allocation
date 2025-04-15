

# Project structure for Supervised Learning -> Offline RL in ads budget allocation

# ------------------------------------
# 1. ROI Prediction (Supervised Learning)
# ------------------------------------

class ROIPredictor:
    def __init__(self, model):
        self.model = model

    def train(self, train_df, features, target):
        X = train_df[features]
        y = train_df[target]
        self.model.fit(X, y)

    def predict(self, df, features):
        return self.model.predict(df[features])

# Example usage
# roi_model = ROIPredictor(XGBRegressor())
# roi_model.train(train_df, features, target='roi')
# train_df['predicted_roi'] = roi_model.predict(train_df, features)


# ------------------------------------
# 2. Budget Allocation Optimizer (Rule-Based)
# ------------------------------------

def rule_based_budget_allocation(df, score_col, total_budget):
    df = df.copy()
    df['score'] = df[score_col]
    df['allocated_budget'] = (df['score'] / df['score'].sum()) * total_budget
    return df


# ------------------------------------
# 3. Construct Offline RL Dataset
# ------------------------------------

def construct_offline_rl_dataset(df):
    # Input: df with state features, predicted ROI, historical spend
    df = df.copy()
    df['action'] = df['historical_allocated_budget']
    df['reward'] = df['actual_roi'] * df['historical_allocated_budget']
    df['done'] = False  # episodic definition optional
    return df


# ------------------------------------
# 4. ROI Model as Environment (Model-Based RL)
# ------------------------------------

class SimulatedROIEnv:
    def __init__(self, roi_model, features):
        self.roi_model = roi_model
        self.features = features

    def step(self, state_df, action_budget):
        # Use action to simulate new state (optional)
        inputs = state_df[self.features].copy()
        inputs['allocated_budget'] = action_budget
        predicted_roi = self.roi_model.predict(inputs)
        reward = predicted_roi * action_budget
        return reward


# ------------------------------------
# 5. Offline RL Training (DQN or CQL)
# ------------------------------------

def train_offline_rl_model(dataset, q_net, target_net, optimizer, scaler, num_epochs=100):
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    features = ['roi_p10', 'roi_p50', 'roi_p90', 'cvr', 'ctr', 'prev_spend', 'platform_id', 'geo_id']
    X = scaler.transform(dataset[features].astype(float))
    s = torch.tensor(X, dtype=torch.float32)
    a = torch.tensor(dataset['action'].values, dtype=torch.int64).unsqueeze(1)
    r = torch.tensor(dataset['reward'].values, dtype=torch.float32).unsqueeze(1)
    done = torch.tensor(dataset['done'].values, dtype=torch.float32).unsqueeze(1)
    s_next = s  # or use next-state if available

    data = TensorDataset(s, a, r, s_next, done)
    loader = DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for s_b, a_b, r_b, s_n_b, d_b in loader:
            q_values = q_net(s_b).gather(1, a_b)
            with torch.no_grad():
                next_q = target_net(s_n_b).max(1, keepdim=True)[0]
                target = r_b + 0.99 * (1 - d_b) * next_q
            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ------------------------------------
# 6. Inference & Evaluation
# ------------------------------------

def evaluate_policy(policy_model, test_df, features, total_budget):
    state_tensor = torch.tensor(test_df[features].values, dtype=torch.float32)
    with torch.no_grad():
        q_values = policy_model(state_tensor)
        actions = torch.argmax(q_values, dim=1).numpy()
    test_df['score'] = actions + 1
    test_df['allocated_budget'] = (test_df['score'] / test_df['score'].sum()) * total_budget
    test_df['simulated_reward'] = test_df['allocated_budget'] * test_df['roi']
    return test_df[['platform_id', 'geo_id', 'allocated_budget', 'simulated_reward']]


# ------------------------------------
# 7. Main Pipeline
# ------------------------------------
# - Train ROI predictor
# - Simulate allocation
# - Train Offline RL
# - Compare performance



