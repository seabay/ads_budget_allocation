
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
        self.value_head = nn.Linear(64, 1)  # 转化价值（log1p 后）

        # 不确定性加权参数（log sigma）
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))  # for BCE
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))  # for MSE

    def forward(self, x):
        h = self.shared(x)
        prob = torch.sigmoid(self.prob_head(h))
        value_log1p = self.value_head(h)
        return prob, value_log1p

    def compute_loss(self, prob_pred, val_pred_log1p, y_cls, y_val):
        # BCE Loss (classification)
        bce_loss = F.binary_cross_entropy(prob_pred, y_cls)

        # MSE Loss only for positive samples (converted)
        mask = y_cls == 1
        if mask.sum() > 0:
            mse_loss = F.mse_loss(val_pred_log1p[mask], y_val[mask])
        else:
            mse_loss = torch.tensor(0.0, device=prob_pred.device)

        # Uncertainty-weighted total loss
        loss = (
            torch.exp(-2 * self.log_sigma1) * bce_loss +
            torch.exp(-2 * self.log_sigma2) * mse_loss +
            self.log_sigma1 + self.log_sigma2
        )
        return loss, bce_loss.item(), mse_loss.item()

# Example usage:
# model = MultiTaskROIModel(input_dim=feature_dim)
# prob_pred, val_pred_log1p = model(x_batch)
# loss, bce_val, mse_val = model.compute_loss(prob_pred, val_pred_log1p, y_cls, y_val_log1p)
