
from budget_allocator import allocate_budget_with_uncertainty_and_constraints

if __name__ == "__main__":
    csv_path = "roi_with_uncertainty.csv"
    total_budget = 1_000_000  # 100万

    result_df = allocate_budget_with_uncertainty_and_constraints(csv_path, total_budget)
    result_df.to_csv("budget_allocation_result_with_constraints.csv", index=False)
    print(result_df[["platform", "geo", "allocated_budget", "roi_p50", "roi_p90"]].head())
