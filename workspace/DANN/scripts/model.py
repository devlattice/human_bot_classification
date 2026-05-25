"""DANN: shared encoder, task head (logits), domain head (logits) with GRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from grl import GradientReversal


@dataclass
class DANNOutput:
    task_logits: torch.Tensor
    domain_logits: torch.Tensor
    features: torch.Tensor
    nuisance_logits: Optional[torch.Tensor] = None


class MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANN(nn.Module):
    """
    Task head uses encoder features directly.
    Domain head uses GRL(encoder features); encoder receives reversed grads from domain loss.
    Optional nuisance head (e.g. seat bucket) shares the same GRL path for hybrid invariance.
    """

    def __init__(
        self,
        in_dim: int,
        feat_dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        use_nuisance_seat: bool = False,
        n_seat_buckets: int = 9,
    ) -> None:
        super().__init__()
        self.use_nuisance_seat = bool(use_nuisance_seat)
        self.n_seat_buckets = int(n_seat_buckets)
        self.encoder = MLPBackbone(in_dim, hidden_dim, feat_dim, dropout=dropout)
        self.task_head = nn.Linear(feat_dim, 1)
        self.grl = GradientReversal(lambda_=grl_lambda)
        self.domain_head = nn.Linear(feat_dim, 1)
        self.nuisance_head: nn.Module | None
        if self.use_nuisance_seat:
            self.nuisance_head = nn.Linear(feat_dim, self.n_seat_buckets)
        else:
            self.nuisance_head = None

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def forward(self, x: torch.Tensor) -> DANNOutput:
        z = self.encoder(x)
        task_logits = self.task_head(z).squeeze(-1)
        z_rev = self.grl(z)
        domain_logits = self.domain_head(z_rev).squeeze(-1)
        nuis: torch.Tensor | None = None
        if self.nuisance_head is not None:
            nuis = self.nuisance_head(z_rev)
        return DANNOutput(
            task_logits=task_logits,
            domain_logits=domain_logits,
            features=z,
            nuisance_logits=nuis,
        )
