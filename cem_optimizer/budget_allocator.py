
import pandas as pd
import numpy as np
from cem_optimizer import cem_optimizer

def allocate_budget_with_uncertainty(csv_path, total_budget):
    df = pd.read_csv(csv_path)
    roi_ucb = df["roi_p90"].values  # UCB策略：使用P90作为目标
    allocation = cem_optimizer(roi_ucb, total_budget)

    df["allocated_budget"] = allocation
    df["expected_return"] = df["allocated_budget"] * df["roi_p50"]
    df["ucb_return"] = df["allocated_budget"] * df["roi_p90"]
    return df
