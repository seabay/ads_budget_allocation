
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskROIModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.prob_head = nn.Linear(64, 1)  # 转化概率
        # 使用 value_mean_head 和 value_std_head 输出均值与标准差（log1p 后
        self.value_mean_head = nn.Linear(64, 1)  # 转化价值均值（log1p 后）
        self.value_std_head = nn.Linear(64, 1)   # 转化价值标准差（log1p 后）

        # 不确定性加权参数（log sigma）
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # for BCE
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # for MSE

    def forward(self, x):
        h = self.shared(x)
        prob = torch.sigmoid(self.prob_head(h))
        value_mu = self.value_mean_head(h)
        value_log_std = self.value_std_head(h)
        value_std = torch.exp(value_log_std)  # 保证 std > 0
        return prob, value_mu, value_std

    def compute_loss(self, prob_pred, val_mu_pred, val_std_pred, y_cls, y_val_log1p):
        # BCE Loss (classification)
        bce_loss = F.binary_cross_entropy(prob_pred, y_cls)

        # Gaussian NLL Loss only for positive samples
        mask = y_cls == 1
        if mask.sum() > 0:
            val_mu_pred_pos = val_mu_pred[mask]
            val_std_pred_pos = val_std_pred[mask]
            y_val_log1p_pos = y_val_log1p[mask]

            # 使用 Gaussian NLL 作为回归 loss（适用于不确定性建模）
            nll_loss = 0.5 * torch.log(2 * torch.pi * val_std_pred_pos**2) + \
                       0.5 * ((y_val_log1p_pos - val_mu_pred_pos)**2 / (val_std_pred_pos**2))
            nll_loss = nll_loss.mean()
        else:
            nll_loss = torch.tensor(0.0, device=prob_pred.device)

        # Total Loss
        loss = (
            torch.exp(-2 * self.log_sigma1) * bce_loss +
            torch.exp(-2 * self.log_sigma2) * nll_loss +
            self.log_sigma1 + self.log_sigma2
        )
        return loss, bce_loss.item(), nll_loss.item()

    # predict_value_distribution() 方法，输出如 P10 / P50 / P90 等分位数估计值（已还原为原始 scale）
    def predict_value_distribution(self, x, quantiles=[0.1, 0.5, 0.9]):
        with torch.no_grad():
            _, mu, std = self.forward(x)
            results = {}
            for q in quantiles:
                z = torch.tensor(norm_ppf(q), device=mu.device)
                quantile_log1p = mu + std * z
                quantile = torch.expm1(quantile_log1p)  # 还原 log1p 后的真实 value
                results[f"P{int(q*100)}"] = quantile
            return results

# 辅助函数：正态分布分位点（Z 分数）
def norm_ppf(q):
    # 常见分位点的 Z 值
    z_map = {
        0.1: -1.2816,
        0.25: -0.6745,
        0.5: 0.0,
        0.75: 0.6745,
        0.9: 1.2816,
        0.95: 1.6449
    }
    return z_map.get(q, 0.0)  # 默认中位数
