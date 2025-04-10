

import pandas as pd
from cem_optimizer import cem_optimizer_with_constraints

def allocate_budget_with_uncertainty_and_constraints(csv_path, total_budget):
    df = pd.read_csv(csv_path)
    roi_ucb = df["roi_p90"].values  # 使用P90作为UCB策略
    group_constraints = {
        "Facebook": (0.2, 0.4),  # Facebook平台预算占比在 20% 到 40% 之间
        "Google": (0.2, 0.4),    # Google平台预算占比在 20% 到 40% 之间
        "Twitter": (0.2, 0.4),   # Twitter平台预算占比在 20% 到 40% 之间
    }

    allocation = cem_optimizer_with_constraints(roi_ucb, total_budget, group_constraints)

    df["allocated_budget"] = allocation
    df["expected_return"] = df["allocated_budget"] * df["roi_p50"]
    df["ucb_return"] = df["allocated_budget"] * df["roi_p90"]
    return df
