"""Gradient Reversal Layer (GRL) for DANN (Ganin et al., 2016)."""

from __future__ import annotations

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lam = ctx.lambda_
        return -lam * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambda_))


class GradientReversal(nn.Module):
    """Identity forward; multiplies backward gradient by ``-lambda_``."""

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)
