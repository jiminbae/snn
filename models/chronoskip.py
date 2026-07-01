"""ChronoSkip SNN models with soft gates and deployable hard-prefix inference."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbone import ConvSNNBackbone
from .lif import LIFConfig, lif_step


SUPPORTED_MODELS = {
    "fixed_lif",
    "soft_gate",
    "global_chronoskip",
    "global_chronoskip_s2h",
    "layerwise_chronoskip",
    "layerwise_chronoskip_s2h",
}


def _zeros_like(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


def _stack_mean(values: list[Tensor], fallback: Tensor) -> Tensor:
    return torch.stack(values).mean() if values else fallback.new_tensor(0.0)


class FixedLIFSNN(nn.Module):
    """Fixed-timestep Conv-SNN baseline with one LIF setting."""

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
        self.config = LIFConfig(v_th=v_th, tau=tau)
        self.surrogate_scale = surrogate_scale

    def forward(self, x: Tensor, **_: object) -> dict[str, Tensor | dict[str, Tensor]]:
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
        timestep = x.new_tensor(float(self.tmax))
        gates = torch.ones(self.tmax, device=x.device, dtype=x.dtype)
        layer_steps = {"layer1": timestep, "layer2": timestep}
        return {
            "logits": logits_sum / float(self.tmax),
            "raw_spike_rate": spike_rate,
            "gated_spike_rate": spike_rate,
            "spike_rate": spike_rate,
            "prefix_spike_rate": spike_rate,
            "effective_timestep": timestep,
            "hard_effective_timestep": timestep,
            "energy_proxy": spike_rate * timestep,
            "prefix_energy_proxy": spike_rate * timestep,
            "gates": gates,
            "layer_effective_timesteps": layer_steps,
            "layer_hard_timesteps": layer_steps,
        }


class ChronoSkipSNN(nn.Module):
    """Conv-SNN with monotonic soft gates and hard-prefix timestep skipping."""

    def __init__(
        self,
        dataset: str,
        tmax: int = 8,
        model_type: str = "global_chronoskip",
        gate_init: float = 5.0,
        v_th: float = 1.0,
        tau: float = 2.0,
        surrogate_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if model_type not in SUPPORTED_MODELS - {"fixed_lif"}:
            raise ValueError(f"Unsupported ChronoSkip model type: {model_type}")
        self.backbone = ConvSNNBackbone(dataset)
        self.tmax = tmax
        self.model_type = model_type
        self.config = LIFConfig(v_th=v_th, tau=tau)
        self.surrogate_scale = surrogate_scale
        self.is_layerwise = model_type.startswith("layerwise")

        if self.is_layerwise:
            self.theta1 = nn.Parameter(torch.ones(tmax) * gate_init)
            self.theta2 = nn.Parameter(torch.ones(tmax) * gate_init)
        else:
            self.theta = nn.Parameter(torch.ones(tmax) * gate_init)

    def _monotonic_gate(self, theta: Tensor) -> Tensor:
        return torch.cumprod(torch.sigmoid(theta), dim=0)

    def timestep_gates(self) -> Tensor | dict[str, Tensor]:
        if self.is_layerwise:
            return {
                "layer1": self._monotonic_gate(self.theta1),
                "layer2": self._monotonic_gate(self.theta2),
            }
        return self._monotonic_gate(self.theta)

    def _gate_pair(self) -> tuple[Tensor, Tensor]:
        gates = self.timestep_gates()
        if isinstance(gates, dict):
            return gates["layer1"], gates["layer2"]
        return gates, gates

    def _hard_mask(self, gates: Tensor, threshold: float, min_prefix_steps: int) -> Tensor:
        active = torch.cumprod((gates > threshold).to(gates.dtype), dim=0)
        min_steps = max(0, min(self.tmax, int(min_prefix_steps)))
        if min_steps > 0:
            forced = torch.zeros_like(active)
            forced[:min_steps] = 1.0
            active = torch.maximum(active, forced)
        return active

    def forward(
        self,
        x: Tensor,
        *,
        mode: str = "soft",
        gate_threshold: float = 0.5,
        hard_prefix_unscaled: bool = False,
        min_prefix_steps: int = 1,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if mode not in {"soft", "hard_prefix"}:
            raise ValueError(f"Unknown forward mode: {mode}")

        x1 = self.backbone.conv1(x)
        u1 = _zeros_like(x1)
        s1_prev = _zeros_like(x1)

        pooled1 = self.backbone.pool(x1)
        x2_template = self.backbone.conv2(torch.zeros_like(pooled1))
        u2 = _zeros_like(x2_template)
        s2_prev = _zeros_like(x2_template)

        g1, g2 = self._gate_pair()
        hard_mask1 = self._hard_mask(g1, gate_threshold, min_prefix_steps)
        hard_mask2 = self._hard_mask(g2, gate_threshold, min_prefix_steps)
        hard_steps1 = hard_mask1.sum()
        hard_steps2 = hard_mask2.sum()
        effective1 = g1.sum()
        effective2 = g2.sum()

        if mode == "soft":
            steps_to_run = self.tmax
        else:
            steps_to_run = int(torch.maximum(hard_steps1, hard_steps2).detach().cpu().item())

        logits_sum = x.new_zeros((x.shape[0], 10))
        raw_spike_costs: list[Tensor] = []
        gated_spike_costs: list[Tensor] = []
        prefix_spike_costs: list[Tensor] = []

        for t in range(steps_to_run):
            l1_active = mode == "soft" or bool(hard_mask1[t].detach().cpu().item())
            l2_active = mode == "soft" or bool(hard_mask2[t].detach().cpu().item())

            if l1_active:
                u1, s1 = lif_step(
                    x1,
                    u1,
                    s1_prev,
                    v_th=self.config.v_th,
                    beta=self.config.beta,
                    surrogate_scale=self.surrogate_scale,
                )
                raw_spike_costs.append(s1.mean())
                if mode == "hard_prefix" and hard_prefix_unscaled:
                    gate1_t = x.new_tensor(1.0)
                else:
                    gate1_t = g1[t]
                s1_gated = gate1_t * s1
                gated_spike_costs.append(s1_gated.mean())
                prefix_spike_costs.append(s1_gated.mean())
                s1_prev = s1
            else:
                s1_gated = torch.zeros_like(s1_prev)

            if l2_active:
                x2 = self.backbone.conv2(self.backbone.pool(s1_gated))
                u2, s2 = lif_step(
                    x2,
                    u2,
                    s2_prev,
                    v_th=self.config.v_th,
                    beta=self.config.beta,
                    surrogate_scale=self.surrogate_scale,
                )
                raw_spike_costs.append(s2.mean())
                if mode == "hard_prefix" and hard_prefix_unscaled:
                    gate2_t = x.new_tensor(1.0)
                else:
                    gate2_t = g2[t]
                s2_gated = gate2_t * s2
                gated_spike_costs.append(s2_gated.mean())
                prefix_spike_costs.append(s2_gated.mean())
                logits_sum = logits_sum + self.backbone.classify(s2_gated)
                s2_prev = s2

        raw_spike_rate = _stack_mean(raw_spike_costs, x)
        gated_spike_rate = _stack_mean(gated_spike_costs, x)
        prefix_spike_rate = _stack_mean(prefix_spike_costs, x)
        effective_timestep = (effective1 + effective2) / 2.0
        hard_effective_timestep = (hard_steps1 + hard_steps2) / 2.0
        normalizer = torch.clamp(effective_timestep if mode == "soft" else hard_effective_timestep, min=1.0)
        gates = self.timestep_gates()
        layer_effective = {"layer1": effective1, "layer2": effective2}
        layer_hard = {"layer1": hard_steps1, "layer2": hard_steps2}

        return {
            "logits": logits_sum / normalizer,
            "raw_spike_rate": raw_spike_rate,
            "gated_spike_rate": gated_spike_rate,
            "spike_rate": gated_spike_rate,
            "prefix_spike_rate": prefix_spike_rate,
            "effective_timestep": effective_timestep,
            "hard_effective_timestep": hard_effective_timestep,
            "energy_proxy": gated_spike_rate * effective_timestep,
            "prefix_energy_proxy": prefix_spike_rate * hard_effective_timestep,
            "gates": gates,
            "layer_effective_timesteps": layer_effective,
            "layer_hard_timesteps": layer_hard,
        }


def build_model(model_type: str, dataset: str, tmax: int, **_: object) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "fixed_lif":
        return FixedLIFSNN(dataset=dataset, tmax=tmax)
    if model_type in SUPPORTED_MODELS:
        return ChronoSkipSNN(dataset=dataset, tmax=tmax, model_type=model_type)
    raise ValueError(f"Unknown model type: {model_type}")
