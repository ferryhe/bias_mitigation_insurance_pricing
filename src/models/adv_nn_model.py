from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv: float = 1.0):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_adv)


class AdvPredictor(nn.Module):
    """
    Predictor with a shared representation h and one prediction head.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.pred_head = nn.Linear(8, 1)
        self.repr_dim = 8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_extractor(x)
        logits = self.pred_head(h).squeeze(-1)
        return logits, h


class AdversaryHead(nn.Module):
    """
    Takes representation h and predicts A (0/1).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)


def train_adv_nn(
    model: AdvPredictor,
    train_loader: DataLoader,
    config: Any,
    device: torch.device,
) -> AdvPredictor:
    criterion_pred = nn.BCEWithLogitsLoss()
    criterion_adv = nn.CrossEntropyLoss()

    adversary_y0 = AdversaryHead(model.repr_dim).to(device)
    adversary_y1 = AdversaryHead(model.repr_dim).to(device)

    model.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(adversary_y0.parameters())
        + list(adversary_y1.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.n_epochs_adv):
        model.train()
        adversary_y0.train()
        adversary_y1.train()

        warmup_phase = epoch < getattr(config, "warmup_epochs_adv", 0)

        for X_batch, y_batch, A_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            A_batch = A_batch.to(device).long()

            optimizer.zero_grad()
            logits_y, h = model(X_batch)
            loss_pred = criterion_pred(logits_y, y_batch)

            y_batch_binary = y_batch.long()
            batch_size = y_batch_binary.shape[0]

            mask_y0 = y_batch_binary == 0
            mask_y1 = y_batch_binary == 1

            if warmup_phase:
                total_loss = loss_pred
            else:
                loss_adv_y0 = torch.tensor(0.0, device=device)
                if mask_y0.any():
                    h0 = grad_reverse(h[mask_y0], 1.0)
                    logits_a0 = adversary_y0(h0)
                    loss_adv_y0 = criterion_adv(logits_a0, A_batch[mask_y0])
                    loss_adv_y0 = loss_adv_y0 * (mask_y0.sum() / batch_size)

                loss_adv_y1 = torch.tensor(0.0, device=device)
                if mask_y1.any():
                    h1 = grad_reverse(h[mask_y1], 1.0)
                    logits_a1 = adversary_y1(h1)
                    loss_adv_y1 = criterion_adv(logits_a1, A_batch[mask_y1])
                    loss_adv_y1 = loss_adv_y1 * (mask_y1.sum() / batch_size)

                adv_penalty = config.lambda_adv * (loss_adv_y0 + loss_adv_y1)
                total_loss = loss_pred + adv_penalty

            total_loss.backward()
            optimizer.step()

        phase = "warmup" if warmup_phase else "adversarial"
        print(f"Epoch {epoch} [{phase}]")

    return model


def predict_proba_adv_nn(
    model: AdvPredictor, X: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor_X = torch.from_numpy(X.astype(np.float32)).to(device)
        logits, _ = model(tensor_X)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
