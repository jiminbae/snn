"""LIF neuron primitives for the ChronoSkip prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class SurrogateSpike(torch.autograd.Function):
    """Hard step in the forward pass with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        grad = 1.0 / (scale * x.abs() + 1.0).pow(2)
        return grad_output * grad, None


def spike_fn(x: Tensor, scale: float = 10.0) -> Tensor:
    return SurrogateSpike.apply(x, scale)


@dataclass(frozen=True)
class LIFConfig:
    v_th: float = 1.0
    tau: float = 2.0

    @property
    def beta(self) -> float:
        return math.exp(-1.0 / self.tau)


def lif_step(
    x_t: Tensor,
    u_prev: Tensor,
    s_prev: Tensor,
    *,
    v_th: float,
    beta: float,
    surrogate_scale: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """One LIF update: membrane leak, synaptic input, reset by previous spike."""

    u_t = beta * u_prev + x_t - v_th * s_prev
    s_t = spike_fn(u_t - v_th, surrogate_scale)
    return u_t, s_t
