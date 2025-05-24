
import numpy as np
from sklearn.cluster import KMeans

class CEMBudgetAllocator:
    def __init__(self, n_segments=45, n_components=3, n_samples_per_component=100,
                 elite_frac=0.2, n_iterations=5, seed=42):
        np.random.seed(seed)
        self.n_segments = n_segments
        self.n_components = n_components
        self.n_samples_per_component = n_samples_per_component
        self.elite_frac = elite_frac
        self.n_iterations = n_iterations
        self.components = []

    def _init_components(self):
        self.components = []
        for _ in range(self.n_components):
            mean = np.random.dirichlet(np.ones(self.n_segments))
            cov = np.diag(np.ones(self.n_segments) * 0.01)
            self.components.append({"mean": mean, "cov": cov})

    def _normalize_sample(self, sample):
        sample = np.abs(sample)
        return sample / sample.sum()

    def fit(self, roi_predictor):
        self._init_components()

        for _ in range(self.n_iterations):
            all_samples = []
            all_scores = []

            for comp in self.components:
                raw_samples = np.random.multivariate_normal(comp["mean"], comp["cov"],
                                                            size=self.n_samples_per_component)
                samples = np.array([self._normalize_sample(s) for s in raw_samples])
                scores = np.array([roi_predictor(s) for s in samples])
                all_samples.append(samples)
                all_scores.append(scores)

            all_samples = np.vstack(all_samples)
            all_scores = np.hstack(all_scores)

            n_elite = int(self.elite_frac * len(all_samples))
            elite_indices = np.argsort(all_scores)[-n_elite:]
            elite_samples = all_samples[elite_indices]

            kmeans = KMeans(n_clusters=self.n_components, n_init=3).fit(elite_samples)
            labels = kmeans.labels_

            for i in range(self.n_components):
                cluster_points = elite_samples[labels == i]
                if len(cluster_points) > 1:
                    self.components[i]["mean"] = np.mean(cluster_points, axis=0)
                    self.components[i]["cov"] = np.cov(cluster_points.T) + np.eye(self.n_segments) * 1e-4

    def get_best_allocation(self, roi_predictor):
        scores = [roi_predictor(comp["mean"]) for comp in self.components]
        best_idx = np.argmax(scores)
        return self.components[best_idx]["mean"]


# Example usage:
if __name__ == "__main__":
    def dummy_roi_predictor(x):
        weights = np.linspace(1.0, 2.0, len(x))
        return np.sum(np.sqrt(np.clip(x, 0, None)) * weights)

    allocator = CEMBudgetAllocator(n_segments=45, n_iterations=5)
    allocator.fit(dummy_roi_predictor)
    best_allocation = allocator.get_best_allocation(dummy_roi_predictor)

    print("Top 10 allocated segments:")
    top_indices = np.argsort(best_allocation)[-10:][::-1]
    for i in top_indices:
        print(f"Segment {i}: {best_allocation[i]:.4f}")
