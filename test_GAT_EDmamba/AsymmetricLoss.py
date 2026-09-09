import torch
from torch import nn


class AdaptiveAsymmetricMSELoss(nn.Module):
    """
    Cost-sensitive asymmetric loss for RUL prediction.

    The squared error is computed in the same normalized RUL space as the
    existing MSE training loss. Stage masks are computed after restoring the
    target RUL to the original scale with max_rul.
    """

    def __init__(self, sub_dataset, alpha=2.0, gamma=5.0, delta=0.9, focus_threshold=35.0, cap_threshold=125.0, max_rul=125.0):
        super().__init__()
        if delta < 0:
            raise ValueError("delta must be non-negative.")
        self.sub_dataset = sub_dataset
        if sub_dataset == "FD001":
            alpha = 1.5
            gamma = 2.0
        elif sub_dataset == "FD002":
            alpha = 3.0
            gamma = 5.0
        elif sub_dataset == "FD003":
            alpha = 3.5
            gamma = 5.0
        elif sub_dataset == "FD004":
            alpha = 2.0
            gamma = 5.0
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.focus_threshold = float(focus_threshold)
        self.cap_threshold = float(cap_threshold)
        self.max_rul = float(max_rul)

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        over = (error > 0).to(dtype=error.dtype)

        true_rul = y_true * self.max_rul
        focus = (true_rul < self.focus_threshold).to(dtype=error.dtype)
        cap = (true_rul >= self.cap_threshold).to(dtype=error.dtype)

        weight = (1.0 + self.alpha * over)
        weight = weight * (1.0 + self.gamma * focus)
        weight = weight * (1.0 - (1.0 - self.delta) * cap)

        return torch.mean(error.pow(2) * weight)
