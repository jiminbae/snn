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


def _validate_input(x: Tensor, tmax: int) -> None:
    if x.ndim == 4:
        return
    if x.ndim == 5:
        if x.shape[1] != tmax:
            raise ValueError(f"Temporal input length {x.shape[1]} must match tmax={tmax}.")
        return
    raise ValueError(f"Expected input shape [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}.")


def _frame_at(x: Tensor, t: int) -> Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        return x[:, t]
    raise ValueError(f"Expected input shape [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}.")



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

    def forward(
        self,
        x: Tensor,
        *,
        temporal_prefix_steps: int = 0,
        temporal_prefix_mode: str = "none",
        **_: object,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        _validate_input(x, self.tmax)
        if temporal_prefix_mode not in {"none", "zero", "truncate"}:
            raise ValueError(f"Unknown temporal_prefix_mode: {temporal_prefix_mode}")
        if temporal_prefix_steps < 0 or temporal_prefix_steps > self.tmax:
            raise ValueError(f"temporal_prefix_steps must be in [0, {self.tmax}], got {temporal_prefix_steps}.")
        use_temporal_prefix = x.ndim == 5 and temporal_prefix_mode != "none" and temporal_prefix_steps > 0
        steps_to_run = self.tmax
        if x.ndim == 5 and temporal_prefix_mode == "truncate" and temporal_prefix_steps > 0:
            if temporal_prefix_steps < 1:
                raise ValueError("temporal_prefix_steps must be at least 1 for truncate mode.")
            steps_to_run = int(temporal_prefix_steps)

        static_x1 = self.backbone.conv1(x) if x.ndim == 4 else None
        x1_template = static_x1 if static_x1 is not None else self.backbone.conv1(_frame_at(x, 0))
        u1 = _zeros_like(x1_template)
        s1_prev = _zeros_like(x1_template)

        pooled1 = self.backbone.pool(x1_template)
        x2_template = self.backbone.conv2(torch.zeros_like(pooled1))
        u2 = _zeros_like(x2_template)
        s2_prev = _zeros_like(x2_template)

        logits_sum = x.new_zeros((x.shape[0], self.backbone.num_classes))
        spike_costs: list[Tensor] = []

        for t in range(steps_to_run):
            frame_t = _frame_at(x, t)
            if use_temporal_prefix and temporal_prefix_mode == "zero" and t >= temporal_prefix_steps:
                frame_t = torch.zeros_like(frame_t)
            x1_t = static_x1 if static_x1 is not None else self.backbone.conv1(frame_t)
            u1, s1 = lif_step(
                x1_t,
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
        timestep = x.new_tensor(float(steps_to_run))
        gates = torch.ones(self.tmax, device=x.device, dtype=x.dtype)
        if x.ndim == 5 and temporal_prefix_mode == "truncate" and temporal_prefix_steps > 0:
            gates = torch.zeros(self.tmax, device=x.device, dtype=x.dtype)
            gates[:steps_to_run] = 1.0
        layer_steps = {"layer1": timestep, "layer2": timestep}
        return {
            "logits": logits_sum / float(max(1, steps_to_run)),
            "raw_spike_rate": spike_rate,
            "gated_spike_rate": spike_rate,
            "spike_rate": spike_rate,
            "prefix_spike_rate": spike_rate,
            "effective_timestep": timestep,
            "hard_effective_timestep": timestep,
            "executed_timestep": timestep,
            "energy_proxy": spike_rate * timestep,
            "prefix_energy_proxy": spike_rate * timestep,
            "loop_energy_proxy": spike_rate * timestep,
            "is_hard_prefix": x.new_tensor(0.0),
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

    def _hard_budget_proxy_from_gates(self, gates: Tensor, gate_threshold: float, sharpness: float) -> Tensor:
        soft_active = torch.sigmoid(float(sharpness) * (gates - float(gate_threshold)))
        return soft_active.sum()

    def hard_budget_proxy_details(
        self,
        gate_threshold: float = 0.5,
        sharpness: float = 20.0,
    ) -> dict[str, Tensor]:
        g1, g2 = self._gate_pair()
        proxy1 = self._hard_budget_proxy_from_gates(g1, gate_threshold, sharpness)
        if self.is_layerwise:
            proxy2 = self._hard_budget_proxy_from_gates(g2, gate_threshold, sharpness)
            return {
                "layer1": proxy1,
                "layer2": proxy2,
                "average": (proxy1 + proxy2) / 2.0,
            }
        return {"global": proxy1}

    def hard_budget_proxy(
        self,
        gate_threshold: float = 0.5,
        sharpness: float = 20.0,
    ) -> Tensor:
        details = self.hard_budget_proxy_details(gate_threshold=gate_threshold, sharpness=sharpness)
        if self.is_layerwise:
            return details["average"]
        return details["global"]

    def hard_prefix_masks(
        self,
        gate_threshold: float = 0.5,
        min_prefix_steps: int = 1,
        dependency_constrained_prefix: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        g1, g2 = self._gate_pair()
        mask1 = self._hard_mask(g1, gate_threshold, min_prefix_steps)
        if not self.is_layerwise:
            return mask1
        mask2 = self._hard_mask(g2, gate_threshold, min_prefix_steps)
        if dependency_constrained_prefix:
            mask2 = mask2 * mask1
        return {"layer1": mask1, "layer2": mask2}

    def hard_prefix_steps(
        self,
        gate_threshold: float = 0.5,
        min_prefix_steps: int = 1,
        dependency_constrained_prefix: bool = False,
    ) -> dict[str, Tensor]:
        masks = self.hard_prefix_masks(
            gate_threshold=gate_threshold,
            min_prefix_steps=min_prefix_steps,
            dependency_constrained_prefix=dependency_constrained_prefix,
        )
        if isinstance(masks, dict):
            steps1 = masks["layer1"].sum()
            steps2 = masks["layer2"].sum()
            return {
                "layer1": steps1,
                "layer2": steps2,
                "average": (steps1 + steps2) / 2.0,
                "executed": torch.maximum(steps1, steps2),
            }
        return {"global": masks.sum()}

    def forward(
        self,
        x: Tensor,
        *,
        mode: str = "soft",
        gate_threshold: float = 0.5,
        hard_prefix_unscaled: bool = False,
        min_prefix_steps: int = 1,
        dependency_constrained_prefix: bool = False,
        temporal_prefix_steps: int = 0,
        temporal_prefix_mode: str = "none",
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if mode not in {"soft", "hard_prefix"}:
            raise ValueError(f"Unknown forward mode: {mode}")

        _validate_input(x, self.tmax)
        static_x1 = self.backbone.conv1(x) if x.ndim == 4 else None
        x1_template = static_x1 if static_x1 is not None else self.backbone.conv1(_frame_at(x, 0))
        u1 = _zeros_like(x1_template)
        s1_prev = _zeros_like(x1_template)

        pooled1 = self.backbone.pool(x1_template)
        x2_template = self.backbone.conv2(torch.zeros_like(pooled1))
        u2 = _zeros_like(x2_template)
        s2_prev = _zeros_like(x2_template)

        g1, g2 = self._gate_pair()
        hard_mask1 = self._hard_mask(g1, gate_threshold, min_prefix_steps)
        hard_mask2 = self._hard_mask(g2, gate_threshold, min_prefix_steps)
        if mode == "hard_prefix" and self.is_layerwise and dependency_constrained_prefix:
            hard_mask2 = hard_mask2 * hard_mask1
        hard_steps1 = hard_mask1.sum()
        hard_steps2 = hard_mask2.sum()
        effective1 = g1.sum()
        effective2 = g2.sum()

        if mode == "soft":
            steps_to_run = self.tmax
        else:
            steps_to_run = int(torch.maximum(hard_steps1, hard_steps2).detach().cpu().item())

        logits_sum = x.new_zeros((x.shape[0], self.backbone.num_classes))
        raw_spike_costs: list[Tensor] = []
        gated_spike_costs: list[Tensor] = []
        prefix_spike_costs: list[Tensor] = []

        # Layer-wise hard-prefix semantics:
        # If a layer is inactive at timestep t, it produces no new spikes.
        # If layer 1 is inactive but layer 2 is still active, layer 2 receives zero
        # new input from layer 1 at this timestep. This corresponds to a deployable
        # timestep skipping interpretation where skipped upstream computation emits
        # no events, while downstream membrane dynamics can still continue.
        for t in range(steps_to_run):
            l1_active = mode == "soft" or bool(hard_mask1[t].detach().cpu().item())
            l2_active = mode == "soft" or bool(hard_mask2[t].detach().cpu().item())

            if l1_active:
                x1_t = static_x1 if static_x1 is not None else self.backbone.conv1(_frame_at(x, t))
                u1, s1 = lif_step(
                    x1_t,
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
        # In soft mode, prefix_spike_rate is kept equal to gated_spike_rate for
        # logging compatibility. It should only be interpreted as actual prefix
        # spike activity when is_hard_prefix == 1.
        if mode == "soft":
            prefix_spike_rate = gated_spike_rate
        effective_timestep = (effective1 + effective2) / 2.0
        hard_effective_timestep = (hard_steps1 + hard_steps2) / 2.0
        executed_timestep = x.new_tensor(float(self.tmax if mode == "soft" else steps_to_run))
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
            "executed_timestep": executed_timestep,
            "energy_proxy": gated_spike_rate * effective_timestep,
            "prefix_energy_proxy": prefix_spike_rate * hard_effective_timestep,
            "loop_energy_proxy": prefix_spike_rate * executed_timestep,
            "is_hard_prefix": x.new_tensor(1.0 if mode == "hard_prefix" else 0.0),
            "gates": gates,
            "layer_effective_timesteps": layer_effective,
            "layer_hard_timesteps": layer_hard,
        }


def build_model(model_type: str, dataset: str, tmax: int, gate_init: float = 5.0, **_: object) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "fixed_lif":
        return FixedLIFSNN(dataset=dataset, tmax=tmax)
    if model_type in SUPPORTED_MODELS:
        return ChronoSkipSNN(dataset=dataset, tmax=tmax, model_type=model_type, gate_init=gate_init)
    raise ValueError(f"Unknown model type: {model_type}")
