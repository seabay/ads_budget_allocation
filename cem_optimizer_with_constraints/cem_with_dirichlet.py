

import numpy as np

def cem_with_dirichlet_budget(
    simulator,                   # 模拟器函数：接收 budget 向量，返回 reward
    n_segments=45,
    n_geos=15,
    segment_to_geo=None,
    geo_budget_limit=None,       # 每个 geo 的预算比例上限，如 [0.1, ..., 0.1]
    total_budget=1_000_000,
    penalty_scale=100,
    iterations=50,
    population_size=200,
    elite_frac=0.2,
    init_alpha=10.0,
    verbose=True
):
    if segment_to_geo is None:
        segment_to_geo = [i % n_geos for i in range(n_segments)]
    if geo_budget_limit is None:
        geo_budget_limit = [1.0 / n_geos] * n_geos  # 默认平均上限

    alpha = np.full(n_segments, init_alpha)
    best_score = -np.inf
    best_budget = None

    def evaluate(real_budget):
        reward = simulator(real_budget)
        penalty = 0
        for geo_id in range(n_geos):
            geo_mask = np.array(segment_to_geo) == geo_id
            geo_budget = real_budget[geo_mask].sum()
            geo_limit = geo_budget_limit[geo_id] * total_budget
            if geo_budget > geo_limit:
                penalty += (geo_budget - geo_limit) ** 2
        return reward - penalty_scale * penalty

    for it in range(iterations):
        proportions = np.random.dirichlet(alpha, size=population_size)
        budgets = proportions * total_budget  # 转换为实际预算

        scores = np.array([evaluate(b) for b in budgets])
        elite_idx = scores.argsort()[-int(population_size * elite_frac):]
        elite_props = proportions[elite_idx]

        # 更新 alpha (moment-matching)
        alpha = elite_props.mean(axis=0) * init_alpha

        if scores[elite_idx[-1]] > best_score:
            best_score = scores[elite_idx[-1]]
            best_budget = budgets[elite_idx[-1]]

        if verbose:
            print(f"[Iter {it+1:02d}] Best score so far: {best_score:.4f}")

    return best_budget, best_score

def dummy_simulator(budget_vector):
    # 一个简单模拟器，假设最佳预算为均匀分布
    ideal = np.full_like(budget_vector, fill_value=total_budget / len(budget_vector))
    return -np.linalg.norm(budget_vector - ideal) + np.random.normal(0, 1.0)


best_budget, best_score = cem_with_dirichlet_budget(
    simulator=dummy_simulator,
    total_budget=1_000_000,
    geo_budget_limit=[0.08] * 15,  # 每个 geo 最多占用 8%
)
print("最佳预算分配：", best_budget)
print("总和：", best_budget.sum())

# ==================================================

import numpy as np

def objective_function_proportions(proportions, total_budget, return_coeffs_a, return_coeffs_b):
    """
    Calculates the total return for a given budget allocation based on proportions.

    Args:
        proportions (np.ndarray): A 1D numpy array representing the budget
                                  proportion allocated to each project. Sums to 1.
        total_budget (float): The total budget available.
        return_coeffs_a (np.ndarray): Coefficients 'a' for the quadratic
                                      return functions.
        return_coeffs_b (np.ndarray): Coefficients 'b' for the quadratic
                                      return functions.

    Returns:
        float: The total return. Returns -infinity if proportions are invalid
               (though Dirichlet sampling should prevent sum != 1).
    """
    # Convert proportions to actual budget allocation
    allocation = proportions * total_budget

    # While Dirichlet sampling ensures sum=1 and non-negativity in theory,
    # numerical issues or potential edge cases might occur depending on implementation details or constraints.
    # A basic check can still be included, but is less critical than with Gaussian sampling.
    # If adding other constraints (e.g., minimum/maximum proportion per project),
    # those checks would go here and return -np.inf if violated.
    if np.any(proportions < 0) or not np.isclose(np.sum(proportions), 1.0):
         return -np.inf # Should ideally not be hit with correct Dirichlet sampling

    # Calculate total return using the actual allocation
    returns = return_coeffs_a * allocation - return_coeffs_b * (allocation ** 2)

    # We might want to penalize negative returns from individual projects if that's undesirable,
    # or just let the optimization find the best overall sum.
    # For this example, we just sum the returns.
    return np.sum(returns)


def cem_budget_allocation_dirichlet(
    num_projects,
    total_budget,
    return_coeffs_a,
    return_coeffs_b,
    n_samples=500,      # Number of proportion samples per iteration
    n_elite=50,         # Number of elite samples
    n_iterations=100,   # Total CEM iterations
    smoothing=0.9,      # Smoothing parameter for alpha update
    initial_alpha_value=1.0 # Initial value for all elements in alpha vector
):
    """
    Optimizes budget allocation proportions using CEM with Dirichlet distribution.

    Args:
        num_projects (int): The number of projects.
        total_budget (float): The total budget.
        return_coeffs_a (np.ndarray): Coefficients 'a'.
        return_coeffs_b (np.ndarray): Coefficients 'b'.
        n_samples (int): Number of samples per iteration.
        n_elite (int): Number of elite samples.
        n_iterations (int): Number of iterations.
        smoothing (float): Smoothing parameter for alpha update.
        initial_alpha_value (float): Initial value for each element of the alpha vector.

    Returns:
        np.ndarray: The best budget proportion allocation found.
        float: The objective function value for the best proportion allocation.
    """
    if n_elite >= n_samples:
        raise ValueError("n_elite must be less than n_samples")
    if initial_alpha_value <= 0:
         raise ValueError("initial_alpha_value must be positive")

    # Initialize the parameter vector alpha for the Dirichlet distribution
    # A common initialization is a vector of ones, corresponding to a uniform
    # distribution over the simplex (all proportions equally likely initially).
    alpha = np.ones(num_projects) * initial_alpha_value

    best_proportions = np.ones(num_projects) / num_projects # Initialize with equal proportion
    best_return = -np.inf

    print("Starting CEM with Dirichlet for budget allocation proportions...")

    for iteration in range(n_iterations):
        # Sample budget proportions from the current Dirichlet distribution
        # np.random.dirichlet returns samples where each row is a proportion vector
        proportion_samples = np.random.dirichlet(alpha, size=n_samples)

        # Evaluate the objective function for each sample
        returns = np.array([
            objective_function_proportions(
                prop_sample, total_budget, return_coeffs_a, return_coeffs_b
            )
            for prop_sample in proportion_samples
        ])

        # Identify elite samples (proportion vectors with the highest returns)
        valid_indices = np.where(returns > -np.inf)[0]
        if len(valid_indices) < n_elite:
             print(f"Warning: Only {len(valid_indices)} valid samples in iteration {iteration}. Using all valid samples as elite.")
             elite_indices = valid_indices
        else:
            elite_indices = valid_indices[np.argsort(returns[valid_indices])[-n_elite:]]

        elite_proportion_samples = proportion_samples[elite_indices]

        # Update the parameters of the Dirichlet distribution (alpha) based on elite samples
        if len(elite_proportion_samples) > 0:
            # Calculate the mean of the elite proportion samples
            mean_elite_proportions = np.mean(elite_proportion_samples, axis=0)

            # Update alpha based on the mean elite proportions.
            # A common update strategy is to set new alpha proportional to the mean,
            # scaled by the sum of the current alpha (related to concentration).
            # alpha_sum = np.sum(alpha) # Sum of current alpha parameters
            # new_alpha = alpha_sum * mean_elite_proportions # Simple proportional update maintaining sum

            # A more robust approach related to MLE fitting of Dirichlet to samples
            # often involves log means, but can be complex.
            # For a CEM-like update, we can use a simpler rule that pushes alpha
            # towards being proportional to the mean of elite samples,
            # while controlling the concentration (sum of alpha).
            # Let's use a pragmatic update: scale mean_elite_proportions by a factor
            # derived from the current total alpha, and apply smoothing.
            # The factor `np.sum(alpha) / np.sum(mean_elite_proportions)` would theoretically maintain the sum of alpha,
            # but mean_elite_proportions sums to 1. So we can scale by sum(alpha).
            # Or, control the concentration directly. Let's use a factor that, when multiplied
            # by the mean proportions, gives a reasonable scale for alpha, and apply smoothing.
            # A simple update: alpha_new_i = concentration_factor * mean_elite_proportion_i
            # Let's try scaling by the total number of parameters (num_projects) for simplicity,
            # or the sum of the current alpha vector. Scaling by sum of current alpha
            # helps maintain a similar level of concentration, while steering the distribution.
            current_alpha_sum = np.sum(alpha)
            new_alpha = current_alpha_sum * mean_elite_proportions # Update aiming to maintain alpha sum

            # Apply smoothing to the alpha vector
            alpha = smoothing * alpha + (1 - smoothing) * new_alpha

            # Ensure all alpha parameters remain positive (critical for Dirichlet)
            alpha = np.maximum(alpha, 1e-6) # Add a small floor to prevent zero/negative alpha

            # Find the best allocation among the current elite samples
            current_best_proportion_idx = np.argmax(returns[elite_indices])
            current_best_proportions = elite_proportion_samples[current_best_proportion_idx]
            current_best_return = returns[elite_indices][current_best_proportion_idx]


            if current_best_return > best_return:
                best_return = current_best_return
                best_proportions = current_best_proportions.copy()

        print(f"Iteration {iteration+1}/{n_iterations}, Best Return So Far: {best_return:.2f}")
        # Optional: Print current alpha to see how the distribution is changing
        # print(f"Current Alpha: {alpha}")


    # The best_proportions found is the elite sample that yielded the highest return.
    # The mean of the final alpha distribution is another candidate solution,
    # but the best sample encountered is often a good choice.
    # The expected value of a Dirichlet(alpha) distribution is alpha / sum(alpha).
    # final_mean_proportions = alpha / np.sum(alpha)


    return best_proportions, best_return

# --- Example Usage ---
if __name__ == "__main__":
    num_projects = 5
    total_budget = 1000.0

    # Define return function coefficients for each project
    # Example: f_i(budget_i) = a_i * budget_i - b_i * budget_i^2
    return_coeffs_a = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
    return_coeffs_b = np.array([0.01, 0.008, 0.012, 0.009, 0.011])

    # Run the CEM with Dirichlet
    optimal_proportions, max_return = cem_budget_allocation_dirichlet(
        num_projects=num_projects,
        total_budget=total_budget,
        return_coeffs_a=return_coeffs_a,
        return_coeffs_b=return_coeffs_b,
        n_samples=1000,      # Increased samples
        n_elite=100,         # 10% elite samples
        n_iterations=200,    # More iterations
        smoothing=0.9,       # Smoothing parameter
        initial_alpha_value=1.0 # Start with uniform-like distribution of proportions
    )

    print("\n--- Optimization Results ---")
    print(f"Optimal Budget Proportions: {optimal_proportions}")
    print(f"Sum of Proportions: {np.sum(optimal_proportions):.4f}") # Check sum is close to 1

    # Convert optimal proportions back to actual budget allocation
    optimal_allocation = optimal_proportions * total_budget
    print(f"Optimal Budget Allocation: {optimal_allocation}")
    print(f"Sum of Allocation: {np.sum(optimal_allocation):.2f}") # Check sum is close to total budget


    print(f"Maximum Estimated Return: {max_return:.2f}")

    # Verify the objective function value for the found allocation
    final_check_return = objective_function_proportions(
        optimal_proportions,
        total_budget,
        return_coeffs_a,
        return_coeffs_b
    )
    print(f"Objective function value at final proportions: {final_check_return:.2f}")
