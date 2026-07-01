"""Fixed LIF baseline and SpikeGate model variants."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import ConvSNNBackbone
from .lif import DEFAULT_CANDIDATES, LIFConfig, lif_step


def _zeros_like(x: Tensor, num_candidates: int | None = None) -> Tensor:
    if num_candidates is None:
        return torch.zeros_like(x)
    return torch.zeros((num_candidates, *x.shape), device=x.device, dtype=x.dtype)


class FixedLIFSNN(nn.Module):
    """Fixed Conv-SNN baseline with one LIF setting for all spiking layers."""

    def __init__(
        self,
        dataset: str,
        tmax: int = 8,
        v_th: float = 1.0,
        tau: float = 2.0,
        surrogate_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.backbone = ConvSNNBackbone(dataset)
        self.tmax = tmax
        self.config = LIFConfig("fixed_lif", v_th=v_th, tau=tau)
        self.surrogate_scale = surrogate_scale

    def forward(self, x: Tensor, **_: object) -> dict[str, Tensor | list[str] | list[int]]:
        x1 = self.backbone.conv1(x)
        u1 = _zeros_like(x1)
        s1_prev = _zeros_like(x1)

        pooled1 = self.backbone.pool(x1)
        x2_template = self.backbone.conv2(torch.zeros_like(pooled1))
        u2 = _zeros_like(x2_template)
        s2_prev = _zeros_like(x2_template)

        logits_sum = x.new_zeros((x.shape[0], 10))
        spike_costs: list[Tensor] = []

        for _ in range(self.tmax):
            # Static image input is presented at every timestep.
            u1, s1 = lif_step(
                x1,
                u1,
                s1_prev,
                v_th=self.config.v_th,
                beta=self.config.beta,
                surrogate_scale=self.surrogate_scale,
            )
            x2 = self.backbone.conv2(self.backbone.pool(s1))
            u2, s2 = lif_step(
                x2,
                u2,
                s2_prev,
                v_th=self.config.v_th,
                beta=self.config.beta,
                surrogate_scale=self.surrogate_scale,
            )
            logits_sum = logits_sum + self.backbone.classify(s2)
            spike_costs.extend([s1.mean(), s2.mean()])
            s1_prev, s2_prev = s1, s2

        spike_rate = torch.stack(spike_costs).mean()
        effective_timestep = x.new_tensor(float(self.tmax))
        return {
            "logits": logits_sum / self.tmax,
            "spike_rate": spike_rate,
            "raw_spike_rate": spike_rate,
            "gated_spike_rate": spike_rate,
            "gates": torch.ones(self.tmax, device=x.device, dtype=x.dtype),
            "effective_timestep": effective_timestep,
            "hard_effective_timestep": effective_timestep,
            "prefix_spike_rate": spike_rate,
            "selected_indices": [],
            "selected_names": ["fixed_lif", "fixed_lif"],
            "candidate_probs": [],
        }


class SpikeGateSNN(nn.Module):
    """SpikeGate model with layer-wise neuron search and optional timestep gate."""

    def __init__(
        self,
        dataset: str,
        tmax: int = 8,
        model_type: str = "spikegate",
        monotonic_gate: bool = False,
        gate_threshold: float = 0.5,
        surrogate_scale: float = 10.0,
        candidates: tuple[LIFConfig, ...] = DEFAULT_CANDIDATES,
    ) -> None:
        super().__init__()
        self.backbone = ConvSNNBackbone(dataset)
        self.tmax = tmax
        self.model_type = model_type
        self.monotonic_gate = monotonic_gate
        self.gate_threshold = gate_threshold
        self.surrogate_scale = surrogate_scale
        self.candidates = candidates

        self.use_gate = model_type in {"spikegate", "gate_only", "softmax_spikegate"}
        self.use_search = model_type in {"spikegate", "neuron_only", "softmax_spikegate"}
        self.use_softmax_mixture = model_type == "softmax_spikegate"

        if self.use_search:
            self.alpha1 = nn.Parameter(torch.zeros(len(candidates)))
            self.alpha2 = nn.Parameter(torch.zeros(len(candidates)))
        else:
            self.register_buffer("alpha1", torch.zeros(len(candidates)))
            self.register_buffer("alpha2", torch.zeros(len(candidates)))

        if self.use_gate:
            self.theta = nn.Parameter(torch.ones(tmax) * 5.0)
        else:
            self.register_buffer("theta", torch.ones(tmax) * 30.0)

    def timestep_gates(self) -> Tensor:
        # A soft gate scales spike output at each timestep; the monotonic option
        # forms a prefix-like non-increasing schedule by cumulative products.
        raw = torch.sigmoid(self.theta)
        if self.monotonic_gate:
            return torch.cumprod(raw, dim=0)
        return raw

    def architecture_weights(self, alpha: Tensor, gumbel_tau: float) -> Tensor:
        if not self.use_search:
            z = torch.zeros_like(alpha)
            z[1] = 1.0
            return z
        if self.use_softmax_mixture:
            return torch.softmax(alpha / gumbel_tau, dim=0)
        if self.training:
            # Hard Gumbel-Softmax selects one LIF candidate in the forward pass
            # while preserving a differentiable path to the architecture logits.
            return F.gumbel_softmax(alpha, tau=gumbel_tau, hard=True)
        idx = torch.argmax(alpha)
        return F.one_hot(idx, num_classes=alpha.numel()).to(alpha.dtype)

    def _candidate_step(
        self,
        x_t: Tensor,
        u_prev: Tensor,
        s_prev: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        u_next: list[Tensor] = []
        s_next: list[Tensor] = []
        for idx, candidate in enumerate(self.candidates):
            u_i, s_i = lif_step(
                x_t,
                u_prev[idx],
                s_prev[idx],
                v_th=candidate.v_th,
                beta=candidate.beta,
                surrogate_scale=self.surrogate_scale,
            )
            u_next.append(u_i)
            s_next.append(s_i)
        u_stack = torch.stack(u_next, dim=0)
        s_stack = torch.stack(s_next, dim=0)
        view_shape = (weights.numel(),) + (1,) * (s_stack.dim() - 1)
        mixed_spikes = (weights.view(view_shape) * s_stack).sum(dim=0)
        return u_stack, s_stack, mixed_spikes

    def forward(
        self,
        x: Tensor,
        *,
        gumbel_tau: float = 1.0,
        gate_threshold: float | None = None,
        hard_prefix_steps: int | None = None,
        hard_prefix_unscaled: bool = False,
    ) -> dict[str, Tensor | list[Tensor] | list[str] | list[int]]:
        x1 = self.backbone.conv1(x)
        num_candidates = len(self.candidates)
        u1 = _zeros_like(x1, num_candidates)
        s1_prev = _zeros_like(x1, num_candidates)

        pooled1 = self.backbone.pool(x1)
        x2_template = self.backbone.conv2(torch.zeros_like(pooled1))
        u2 = _zeros_like(x2_template, num_candidates)
        s2_prev = _zeros_like(x2_template, num_candidates)

        z1 = self.architecture_weights(self.alpha1, gumbel_tau)
        z2 = self.architecture_weights(self.alpha2, gumbel_tau)
        gates = self.timestep_gates()

        logits_sum = x.new_zeros((x.shape[0], 10))
        raw_spike_costs: list[Tensor] = []
        gated_spike_costs: list[Tensor] = []
        steps_to_run = self.tmax if hard_prefix_steps is None else max(0, min(self.tmax, int(hard_prefix_steps)))

        for t in range(steps_to_run):
            gate_t = x.new_tensor(1.0) if hard_prefix_steps is not None and hard_prefix_unscaled else gates[t]
            u1, s1_prev, s1 = self._candidate_step(x1, u1, s1_prev, z1)
            s1_gated = gate_t * s1
            x2 = self.backbone.conv2(self.backbone.pool(s1_gated))
            u2, s2_prev, s2 = self._candidate_step(x2, u2, s2_prev, z2)
            s2_gated = gate_t * s2
            logits_sum = logits_sum + self.backbone.classify(s2_gated)
            raw_spike_costs.extend([s1.mean(), s2.mean()])
            gated_spike_costs.extend([s1_gated.mean(), s2_gated.mean()])

        raw_spike_rate = torch.stack(raw_spike_costs).mean() if raw_spike_costs else x.new_tensor(0.0)
        gated_spike_rate = torch.stack(gated_spike_costs).mean() if gated_spike_costs else x.new_tensor(0.0)
        effective_timestep = gates.sum()
        threshold = self.gate_threshold if gate_threshold is None else gate_threshold
        active_prefix = torch.cumprod((gates > threshold).to(x.dtype), dim=0)
        hard_effective = active_prefix.sum()
        normalizer = torch.clamp(
            effective_timestep if hard_prefix_steps is None else x.new_tensor(float(max(steps_to_run, 1))),
            min=1e-6,
        )

        return {
            "logits": logits_sum / normalizer,
            "spike_rate": gated_spike_rate,
            "raw_spike_rate": raw_spike_rate,
            "gated_spike_rate": gated_spike_rate,
            "gates": gates,
            "effective_timestep": effective_timestep,
            "hard_effective_timestep": hard_effective,
            "prefix_spike_rate": gated_spike_rate,
            "selected_indices": self.selected_indices(),
            "selected_names": self.selected_names(),
            "candidate_probs": self.candidate_probabilities(),
        }

    def candidate_probabilities(self) -> list[Tensor]:
        return [torch.softmax(self.alpha1, dim=0), torch.softmax(self.alpha2, dim=0)]

    def selected_indices(self) -> list[int]:
        return [int(torch.argmax(self.alpha1).item()), int(torch.argmax(self.alpha2).item())]

    def selected_names(self) -> list[str]:
        return [self.candidates[idx].name for idx in self.selected_indices()]

    def candidate_metadata(self) -> list[dict[str, float | str]]:
        return [asdict(candidate) for candidate in self.candidates]


def build_model(
    model_type: str,
    dataset: str,
    tmax: int,
    monotonic_gate: bool = False,
    gate_threshold: float = 0.5,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "fixed_lif":
        return FixedLIFSNN(dataset=dataset, tmax=tmax)
    if model_type in {"spikegate", "gate_only", "neuron_only", "softmax_spikegate"}:
        return SpikeGateSNN(
            dataset=dataset,
            tmax=tmax,
            model_type=model_type,
            monotonic_gate=monotonic_gate,
            gate_threshold=gate_threshold,
        )
    raise ValueError(f"Unknown model type: {model_type}")
