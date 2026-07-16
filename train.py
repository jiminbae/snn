from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from models import build_model
from utils.data import build_dataloaders, build_train_val_test_dataloaders
from utils.logging import append_metrics, prepare_run_dir, save_json
from utils.metrics import AverageMeter, accuracy, energy_proxy
from utils.plotting import plot_timestep_gates, plot_training_curves
from utils.prefix_evaluation import evaluate_prefix_diagnostics, save_prefix_diagnostics
from utils.temporal_reliability_loss import (all_prefix_cross_entropy, selective_regression_loss,
    combine_temporal_objective, temporal_reliability_metrics)

MODEL_CHOICES = [
    "fixed_lif",
    "soft_gate",
    "global_chronoskip",
    "global_chronoskip_s2h",
    "layerwise_chronoskip",
    "layerwise_chronoskip_s2h",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ChronoSkip SNN prototypes.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="fixed_lif")
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10", "nmnist", "dvs_gesture"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--temporal-prefix-steps", type=int, default=0)
    parser.add_argument("--temporal-prefix-mode", choices=["none", "zero", "truncate"], default="none")
    parser.add_argument("--temporal-training-mode", choices=["final_ce", "all_prefix_ce", "symmetric_kl", "selective_regression"], default="final_ce")
    parser.add_argument("--prefix-loss-weight", type=float, default=0.0)
    parser.add_argument("--temporal-loss-weight", type=float, default=1.0)
    parser.add_argument("--temporal-margin", type=float, default=0.0)
    parser.add_argument("--temporal-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--temporal-temperature", type=float, default=1.0)
    parser.add_argument("--temporal-selection-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--gate-init", type=float, default=5.0)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.02)
    parser.add_argument("--lambda-hard-budget", type=float, default=0.0)
    parser.add_argument("--hard-budget-sharpness", type=float, default=20.0)
    parser.add_argument("--target-timestep", type=float, default=0.0)
    parser.add_argument("--target-budget-weight", type=float, default=0.0)
    parser.add_argument("--target-budget-mode", choices=["upper", "two_sided", "l2"], default="upper")
    parser.add_argument("--min-target-timestep", type=float, default=0.0)
    parser.add_argument("--min-target-weight", type=float, default=0.0)
    parser.add_argument("--spike-cost-mode", choices=["raw", "gated", "mixed"], default="gated")
    parser.add_argument("--hard-prefix-eval", action="store_true")
    parser.add_argument("--hard-prefix-unscaled", action="store_true")
    parser.add_argument("--dependency-constrained-prefix", action="store_true")
    parser.add_argument("--min-prefix-steps", type=int, default=1)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--hard-ce-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--checkpoint-selection", choices=["last", "best_val"], default="last")
    parser.add_argument("--selection-metric", choices=["val_acc", "val_loss"], default="val_acc")
    parser.add_argument("--prefix-diagnostics", action="store_true")
    parser.add_argument("--save-prefix-trajectories", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def is_better_validation_checkpoint(
    *,
    candidate_acc: float,
    candidate_loss: float,
    best_acc: float | None,
    best_loss: float | None,
    selection_metric: str,
) -> bool:
    if not math.isfinite(candidate_acc) or not math.isfinite(candidate_loss):
        return False
    if best_acc is None or best_loss is None:
        return True
    if selection_metric == "val_acc":
        return candidate_acc > best_acc or (candidate_acc == best_acc and candidate_loss < best_loss)
    if selection_metric == "val_loss":
        return candidate_loss < best_loss or (candidate_loss == best_loss and candidate_acc > best_acc)
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def load_checkpoint_compat(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_model_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint_compat(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def output_to_float(output: dict[str, Any], key: str) -> float:
    value = output[key]
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def nested_output_to_float(output: dict[str, Any], group: str, key: str) -> float:
    value = output[group][key]
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def select_spike_cost(output: dict[str, Any], mode: str) -> torch.Tensor:
    raw = output["raw_spike_rate"]
    gated = output["gated_spike_rate"]
    assert isinstance(raw, torch.Tensor)
    assert isinstance(gated, torch.Tensor)
    if mode == "raw":
        return raw
    if mode == "gated":
        return gated
    if mode == "mixed":
        return 0.5 * raw + 0.5 * gated
    raise ValueError(f"Unknown spike cost mode: {mode}")


def hard_budget_terms(
    args: argparse.Namespace,
    model: nn.Module,
    output: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    zero = logits.new_tensor(0.0)
    if args.model == "fixed_lif" or not hasattr(model, "hard_budget_proxy"):
        return zero, zero, zero, zero

    hard_budget_proxy = model.hard_budget_proxy(
        gate_threshold=args.gate_threshold,
        sharpness=args.hard_budget_sharpness,
    )
    assert isinstance(hard_budget_proxy, torch.Tensor)
    hard_budget_cost = hard_budget_proxy / float(args.tmax)
    target_budget_loss = zero
    if args.target_timestep > 0.0 and args.target_budget_weight > 0.0:
        target = hard_budget_proxy.new_tensor(args.target_timestep)
        delta = hard_budget_proxy - target
        if args.target_budget_mode == "upper":
            target_budget_loss = torch.relu(delta) / float(args.tmax)
        elif args.target_budget_mode == "two_sided":
            target_budget_loss = torch.abs(delta) / float(args.tmax)
        elif args.target_budget_mode == "l2":
            target_budget_loss = delta.pow(2) / float(args.tmax)
        else:
            raise ValueError(f"Unknown target_budget_mode: {args.target_budget_mode}")
    min_target_loss = zero
    if args.min_target_timestep > 0.0 and args.min_target_weight > 0.0:
        min_target = hard_budget_proxy.new_tensor(args.min_target_timestep)
        min_target_loss = torch.relu(min_target - hard_budget_proxy) / float(args.tmax)
    return hard_budget_cost, target_budget_loss, min_target_loss, hard_budget_proxy


def regularized_loss(
    args: argparse.Namespace,
    model: nn.Module,
    output: dict[str, Any],
    target: torch.Tensor,
    reg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    ce_loss = F.cross_entropy(logits, target)
    spike_cost = select_spike_cost(output, args.spike_cost_mode)
    effective_timestep = output["effective_timestep"]
    assert isinstance(effective_timestep, torch.Tensor)
    time_cost = effective_timestep / float(args.tmax)
    hard_budget_cost, target_budget_loss, min_target_loss, hard_budget_proxy = hard_budget_terms(args, model, output)
    if args.model == "fixed_lif":
        total = ce_loss
    else:
        total = (
            ce_loss
            + reg_scale * args.lambda_spike * spike_cost
            + reg_scale * args.eta_time * time_cost
            + reg_scale * args.lambda_hard_budget * hard_budget_cost
            + reg_scale * args.target_budget_weight * target_budget_loss
            + reg_scale * args.min_target_weight * min_target_loss
        )
    return total, ce_loss, spike_cost, time_cost, hard_budget_cost, target_budget_loss, min_target_loss, hard_budget_proxy


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    meter_names = [
        "loss",
        "ce_loss",
        "ce_hard",
        "consistency_loss",
        "spike_cost",
        "time_cost",
        "hard_budget_cost",
        "target_budget_loss",
        "min_target_loss",
        "hard_budget_proxy",
        "acc",
        "acc_hard",
        "raw_spike_rate",
        "gated_spike_rate",
        "final_ce",
        "prefix_ce",
        "temporal_loss",
        "selected_transition_fraction",
        "violating_transition_fraction",
    ]
    meters = {name: AverageMeter() for name in meter_names}
    amp_enabled = args.amp and device.type == "cuda"
    reg_scale = min(1.0, epoch / max(1, args.reg_warmup_epochs))
    use_s2h = args.model.endswith("_s2h")
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)

    for batch_idx, (images, target) in enumerate(progress, start=1):
        if args.limit_train_batches and batch_idx > args.limit_train_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            soft_output = model(
                images,
                mode="soft",
                gate_threshold=args.gate_threshold,
                min_prefix_steps=args.min_prefix_steps,
                temporal_prefix_steps=args.temporal_prefix_steps,
                temporal_prefix_mode=args.temporal_prefix_mode,
                return_prefix_logits=args.temporal_training_mode != "final_ce",
            )
            if use_s2h:
                # The hard-prefix pass uses non-differentiable binary prefix decisions.
                # Therefore, hard CE and consistency mainly train the network weights to be
                # robust under hard-prefix inference. The soft gate and time/hard-budget
                # regularizers remain the differentiable path for learning timestep budgets.
                hard_output = model(
                    images,
                    mode="hard_prefix",
                    gate_threshold=args.gate_threshold,
                    hard_prefix_unscaled=True,
                    min_prefix_steps=args.min_prefix_steps,
                    dependency_constrained_prefix=args.dependency_constrained_prefix,
                    temporal_prefix_steps=args.temporal_prefix_steps,
                    temporal_prefix_mode=args.temporal_prefix_mode,
                )
                soft_logits = soft_output["logits"]
                hard_logits = hard_output["logits"]
                assert isinstance(soft_logits, torch.Tensor)
                assert isinstance(hard_logits, torch.Tensor)
                ce_soft = F.cross_entropy(soft_logits, target)
                ce_hard = F.cross_entropy(hard_logits, target)
                temp = 1.0
                consistency = F.kl_div(
                    F.log_softmax(hard_logits / temp, dim=1),
                    F.softmax(soft_logits.detach() / temp, dim=1),
                    reduction="batchmean",
                ) * temp * temp
                spike_cost = select_spike_cost(soft_output, args.spike_cost_mode)
                effective_timestep = soft_output["effective_timestep"]
                assert isinstance(effective_timestep, torch.Tensor)
                time_cost = effective_timestep / float(args.tmax)
                hard_budget_cost, target_budget_loss, min_target_loss, hard_budget_proxy = hard_budget_terms(args, model, soft_output)
                total = (
                    ce_soft
                    + args.hard_ce_weight * ce_hard
                    + args.consistency_weight * consistency
                    + reg_scale * args.lambda_spike * spike_cost
                    + reg_scale * args.eta_time * time_cost
                    + reg_scale * args.lambda_hard_budget * hard_budget_cost
                    + reg_scale * args.target_budget_weight * target_budget_loss
                    + reg_scale * args.min_target_weight * min_target_loss
                )
            else:
                hard_output = None
                total, ce_soft, spike_cost, time_cost, hard_budget_cost, target_budget_loss, min_target_loss, hard_budget_proxy = regularized_loss(
                    args, model, soft_output, target, reg_scale
                )
                ce_hard = images.new_tensor(0.0)
                consistency = images.new_tensor(0.0)

            prefix_logits = soft_output.get("prefix_logits")
            assert prefix_logits is None or isinstance(prefix_logits, torch.Tensor)
            total, prefix_ce, temporal_loss, temporal_diagnostics = combine_temporal_objective(
                args.temporal_training_mode, total, ce_soft, prefix_logits, target,
                prefix_loss_weight=args.prefix_loss_weight, temporal_loss_weight=args.temporal_loss_weight,
                margin=args.temporal_margin, confidence_threshold=args.temporal_confidence_threshold,
                temperature=args.temporal_temperature, selection_mode=args.temporal_selection_mode,
            )

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        soft_logits = soft_output["logits"]
        assert isinstance(soft_logits, torch.Tensor)
        meters["loss"].update(total.detach().item(), batch_size)
        meters["ce_loss"].update(ce_soft.detach().item(), batch_size)
        meters["ce_hard"].update(ce_hard.detach().item(), batch_size)
        meters["consistency_loss"].update(consistency.detach().item(), batch_size)
        meters["spike_cost"].update(spike_cost.detach().item(), batch_size)
        meters["time_cost"].update(time_cost.detach().item(), batch_size)
        meters["hard_budget_cost"].update(hard_budget_cost.detach().item(), batch_size)
        meters["target_budget_loss"].update(target_budget_loss.detach().item(), batch_size)
        meters["min_target_loss"].update(min_target_loss.detach().item(), batch_size)
        meters["hard_budget_proxy"].update(hard_budget_proxy.detach().item(), batch_size)
        meters["acc"].update(accuracy(soft_logits.detach(), target), batch_size)
        if hard_output is not None:
            hard_logits = hard_output["logits"]
            assert isinstance(hard_logits, torch.Tensor)
            meters["acc_hard"].update(accuracy(hard_logits.detach(), target), batch_size)
        meters["raw_spike_rate"].update(output_to_float(soft_output, "raw_spike_rate"), batch_size)
        meters["gated_spike_rate"].update(output_to_float(soft_output, "gated_spike_rate"), batch_size)
        meters["final_ce"].update(ce_soft.detach().item(), batch_size)
        meters["prefix_ce"].update(prefix_ce.detach().item(), batch_size)
        meters["temporal_loss"].update(temporal_loss.detach().item(), batch_size)
        meters["selected_transition_fraction"].update(temporal_diagnostics["selected_transition_fraction"].item(), batch_size)
        meters["violating_transition_fraction"].update(temporal_diagnostics["violating_transition_fraction"].item(), batch_size)
        progress.set_postfix(loss=f"{meters['loss'].avg:.4f}", acc=f"{meters['acc'].avg:.2f}")

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    split_name: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    meter_names = [
        "loss",
        "acc",
        "soft_acc",
        "hard_acc",
        "raw_spike_rate",
        "gated_spike_rate",
        "prefix_spike_rate",
        "effective_timestep",
        "hard_effective_timestep",
        "layer1_effective_timestep",
        "layer2_effective_timestep",
        "layer1_hard_timestep",
        "layer2_hard_timestep",
        "energy_proxy",
        "prefix_energy_proxy",
        "executed_timestep",
        "loop_energy_proxy",
        "final_accuracy",
        "mean_prefix_accuracy",
        "ever_regressed_fraction",
        "mean_population_regression",
        "mean_conditional_regression",
        "beneficial_transition_fraction",
        "destructive_transition_fraction",
        "stable_correct_fraction",
    ]
    meters = {name: AverageMeter() for name in meter_names}
    all_prefix_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    progress = tqdm(loader, desc=split_name, leave=False)

    for batch_idx, (images, target) in enumerate(progress, start=1):
        if max_batches and batch_idx > max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        soft_output = model(
            images,
            mode="soft",
            gate_threshold=args.gate_threshold,
            min_prefix_steps=args.min_prefix_steps,
            temporal_prefix_steps=args.temporal_prefix_steps,
            temporal_prefix_mode=args.temporal_prefix_mode,
            return_prefix_logits=True,
        )
        eval_output = soft_output
        hard_acc_value = 0.0
        if args.hard_prefix_eval:
            eval_output = model(
                images,
                mode="hard_prefix",
                gate_threshold=args.gate_threshold,
                hard_prefix_unscaled=args.hard_prefix_unscaled,
                min_prefix_steps=args.min_prefix_steps,
                dependency_constrained_prefix=args.dependency_constrained_prefix,
                temporal_prefix_steps=args.temporal_prefix_steps,
                temporal_prefix_mode=args.temporal_prefix_mode,
            )
            hard_logits = eval_output["logits"]
            assert isinstance(hard_logits, torch.Tensor)
            hard_acc_value = accuracy(hard_logits, target)

        soft_logits = soft_output["logits"]
        eval_logits = eval_output["logits"]
        assert isinstance(soft_logits, torch.Tensor)
        assert isinstance(eval_logits, torch.Tensor)
        batch_size = images.shape[0]

        meters["loss"].update(F.cross_entropy(eval_logits, target).item(), batch_size)
        meters["acc"].update(accuracy(eval_logits, target), batch_size)
        meters["soft_acc"].update(accuracy(soft_logits, target), batch_size)
        meters["hard_acc"].update(hard_acc_value, batch_size)
        meters["raw_spike_rate"].update(output_to_float(soft_output, "raw_spike_rate"), batch_size)
        meters["gated_spike_rate"].update(output_to_float(soft_output, "gated_spike_rate"), batch_size)
        metric_output = eval_output if args.hard_prefix_eval else soft_output
        prefix_spike = output_to_float(metric_output, "prefix_spike_rate")
        prefix_timestep = output_to_float(metric_output, "hard_effective_timestep")
        executed_timestep = output_to_float(metric_output, "executed_timestep")
        meters["prefix_spike_rate"].update(prefix_spike, batch_size)
        meters["effective_timestep"].update(output_to_float(soft_output, "effective_timestep"), batch_size)
        meters["hard_effective_timestep"].update(prefix_timestep, batch_size)
        meters["executed_timestep"].update(executed_timestep, batch_size)
        meters["layer1_effective_timestep"].update(nested_output_to_float(soft_output, "layer_effective_timesteps", "layer1"), batch_size)
        meters["layer2_effective_timestep"].update(nested_output_to_float(soft_output, "layer_effective_timesteps", "layer2"), batch_size)
        meters["layer1_hard_timestep"].update(nested_output_to_float(metric_output, "layer_hard_timesteps", "layer1"), batch_size)
        meters["layer2_hard_timestep"].update(nested_output_to_float(metric_output, "layer_hard_timesteps", "layer2"), batch_size)
        meters["energy_proxy"].update(energy_proxy(output_to_float(soft_output, "gated_spike_rate"), output_to_float(soft_output, "effective_timestep")), batch_size)
        meters["prefix_energy_proxy"].update(energy_proxy(prefix_spike, prefix_timestep), batch_size)
        meters["loop_energy_proxy"].update(energy_proxy(prefix_spike, executed_timestep), batch_size)
        reliability = temporal_reliability_metrics(soft_output["prefix_logits"], target)
        for key in ("final_accuracy", "mean_prefix_accuracy", "ever_regressed_fraction",
                    "mean_population_regression", "mean_conditional_regression",
                    "beneficial_transition_fraction", "destructive_transition_fraction",
                    "stable_correct_fraction"):
            meters[key].update(float(reliability[key].item()), batch_size)
        all_prefix_logits.append(soft_output["prefix_logits"].detach().cpu())
        all_targets.append(target.detach().cpu())
        progress.set_postfix(acc=f"{meters['acc'].avg:.2f}")

    result = {key: meter.avg for key, meter in meters.items()}
    if all_prefix_logits:
        dataset_reliability = temporal_reliability_metrics(torch.cat(all_prefix_logits), torch.cat(all_targets))
        for key in ("final_accuracy", "mean_prefix_accuracy", "ever_regressed_fraction",
                    "mean_population_regression", "mean_conditional_regression",
                    "beneficial_transition_fraction", "destructive_transition_fraction",
                    "stable_correct_fraction"):
            result[key] = float(dataset_reliability[key].item())
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    """Backward-compatible test evaluation API."""
    metrics = evaluate_loader(
        model, loader, device, args, split_name="test", max_batches=args.limit_test_batches
    )
    return {"test_acc": metrics.pop("acc"), **metrics}


def tensor_tree_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return [float(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {key: tensor_tree_to_python(item) for key, item in value.items()}
    return value


def snapshot_model_state(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    state: dict[str, Any] = {"timestep_gates": []}
    if hasattr(model, "timestep_gates"):
        state["timestep_gates"] = tensor_tree_to_python(model.timestep_gates())
    if hasattr(model, "hard_prefix_masks"):
        state["hard_prefix_masks"] = tensor_tree_to_python(
            model.hard_prefix_masks(
                gate_threshold=args.gate_threshold,
                min_prefix_steps=args.min_prefix_steps,
                dependency_constrained_prefix=args.dependency_constrained_prefix,
            )
        )
    else:
        state["hard_prefix_masks"] = []
    if hasattr(model, "hard_prefix_steps"):
        state["hard_prefix_steps"] = tensor_tree_to_python(
            model.hard_prefix_steps(
                gate_threshold=args.gate_threshold,
                min_prefix_steps=args.min_prefix_steps,
                dependency_constrained_prefix=args.dependency_constrained_prefix,
            )
        )
    else:
        state["hard_prefix_steps"] = {}
    if hasattr(model, "hard_budget_proxy_details"):
        state["hard_budget_proxy"] = tensor_tree_to_python(
            model.hard_budget_proxy_details(
                gate_threshold=args.gate_threshold,
                sharpness=args.hard_budget_sharpness,
            )
        )
    else:
        state["hard_budget_proxy"] = 0.0
    return state


def format_tree(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={format_tree(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(item):.4f}" for item in value) + "]"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    if args.checkpoint_selection == "best_val" and not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--checkpoint-selection best_val requires --val-ratio strictly between 0 and 1.")
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        requested = torch.device("cpu")
    device = requested
    set_seed(args.seed, device)

    run_name = args.run_name or f"{args.model}_{args.dataset}_T{args.tmax}_seed{args.seed}_{int(time.time())}"
    run_dir = prepare_run_dir(args.results_dir, run_name)
    event_downsample_size = args.event_downsample_size
    if args.dataset == "dvs_gesture" and event_downsample_size is None:
        event_downsample_size = 64
    config = vars(args).copy()
    config["resolved_device"] = str(device)
    config["resolved_event_downsample_size"] = event_downsample_size
    save_json(run_dir / "config.json", config)

    val_loader = None
    if args.checkpoint_selection == "best_val":
        train_loader, val_loader, test_loader, split_indices = build_train_val_test_dataloaders(
            args.dataset,
            args.data_dir,
            args.batch_size,
            tmax=args.tmax,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
            event_frame_mode=args.event_frame_mode,
            event_downsample_size=event_downsample_size,
        )
        torch.save(split_indices, run_dir / "split_indices.pt")
    else:
        train_loader, test_loader = build_dataloaders(
            args.dataset,
            args.data_dir,
            args.batch_size,
            tmax=args.tmax,
            num_workers=args.num_workers,
            event_frame_mode=args.event_frame_mode,
            event_downsample_size=event_downsample_size,
        )
    model = build_model(args.model, dataset=args.dataset, tmax=args.tmax, gate_init=args.gate_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    metrics_path = run_dir / "metrics.csv"
    last_eval: dict[str, float] = {}
    best_epoch: int | None = None
    best_val_acc: float | None = None
    best_val_loss: float | None = None
    checkpoint_path = run_dir / "best_checkpoint.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_ce_hard": train_metrics["ce_hard"],
            "train_consistency_loss": train_metrics["consistency_loss"],
            "train_spike_cost": train_metrics["spike_cost"],
            "train_time_cost": train_metrics["time_cost"],
            "train_hard_budget_cost": train_metrics["hard_budget_cost"],
            "train_target_budget_loss": train_metrics["target_budget_loss"],
            "train_min_target_loss": train_metrics["min_target_loss"],
            "train_hard_budget_proxy": train_metrics["hard_budget_proxy"],
            "train_acc_soft": train_metrics["acc"],
            "train_acc_hard": train_metrics["acc_hard"],
            "train_final_ce": train_metrics.get("final_ce", train_metrics["ce_loss"]),
            "train_prefix_ce": train_metrics.get("prefix_ce", 0.0),
            "train_temporal_loss": train_metrics.get("temporal_loss", 0.0),
            "train_total_loss": train_metrics["loss"],
            "train_selected_transition_fraction": train_metrics.get("selected_transition_fraction", 0.0),
            "train_violating_transition_fraction": train_metrics.get("violating_transition_fraction", 0.0),
        }
        if args.checkpoint_selection == "best_val":
            assert val_loader is not None
            val_metrics = evaluate_loader(
                model, val_loader, device, args, split_name="val", max_batches=args.limit_val_batches
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if is_better_validation_checkpoint(
                candidate_acc=val_metrics["acc"],
                candidate_loss=val_metrics["loss"],
                best_acc=best_val_acc,
                best_loss=best_val_loss,
                selection_metric=args.selection_metric,
            ):
                best_epoch = epoch
                best_val_acc = val_metrics["acc"]
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_metric": args.selection_metric,
                        "val_acc": best_val_acc,
                        "val_loss": best_val_loss,
                        "split_seed": args.split_seed,
                        "val_ratio": args.val_ratio,
                        "config": config,
                    },
                    checkpoint_path,
                )
            print(
                f"epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | val {val_metrics['acc']:.2f}% | "
                f"soft {val_metrics['soft_acc']:.2f}% | hard {val_metrics['hard_acc']:.2f}% | "
                f"best epoch {best_epoch}"
            )
        else:
            last_eval = evaluate(model, test_loader, device, args)
            row.update(last_eval)
            print(
                f"epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
                f"test {last_eval['test_acc']:.2f}% | soft {last_eval['soft_acc']:.2f}% | hard {last_eval['hard_acc']:.2f}% | "
                f"raw {last_eval['raw_spike_rate']:.4f} | gated {last_eval['gated_spike_rate']:.4f} | "
                f"T {last_eval['effective_timestep']:.2f}/hard {last_eval['hard_effective_timestep']:.2f}/exec {last_eval['executed_timestep']:.2f} | "
                f"energy proxy {last_eval['energy_proxy']:.4f} | loop proxy {last_eval['loop_energy_proxy']:.4f}"
            )
        append_metrics(metrics_path, row)

    if args.checkpoint_selection == "best_val":
        if best_epoch is None:
            raise RuntimeError("No finite validation checkpoint was produced.")
        restore_model_checkpoint(model, checkpoint_path, device)
        print(f"Loaded validation-selected checkpoint from epoch {best_epoch}.")
        last_eval = evaluate(model, test_loader, device, args)
        print(f"Final test accuracy: {last_eval['test_acc']:.2f}%")
        save_json(
            run_dir / "selection_summary.json",
            {
                "checkpoint_selection": args.checkpoint_selection,
                "selection_metric": args.selection_metric,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_val_acc,
                "best_validation_loss": best_val_loss,
                "total_epochs": args.epochs,
                "test_accuracy_at_selected_checkpoint": last_eval["test_acc"],
                "val_ratio": args.val_ratio,
                "split_seed": args.split_seed,
                "train_size": len(train_loader.dataset),
                "validation_size": len(val_loader.dataset),
                "test_size": len(test_loader.dataset),
                "checkpoint_path": checkpoint_path.name,
            },
        )

    model_state = snapshot_model_state(model, args)
    prefix_metrics: dict[str, Any] = {}
    if args.prefix_diagnostics:
        trajectory_path = run_dir / "prefix_trajectories.pt" if args.save_prefix_trajectories else None
        prefix_metrics = evaluate_prefix_diagnostics(
            model,
            test_loader,
            device,
            args,
            trajectory_path=trajectory_path,
        )
        final_prefix_key = f"prefix_accuracy_t{args.tmax}"
        if (
            args.checkpoint_selection == "best_val"
            and final_prefix_key in prefix_metrics
            and abs(prefix_metrics[final_prefix_key] - last_eval["test_acc"]) > 1e-4
        ):
            raise RuntimeError("Final prefix accuracy does not match final test accuracy.")
        save_prefix_diagnostics(run_dir, prefix_metrics)
        temporal_summary = {
            "temporal_training_mode": args.temporal_training_mode,
            "prefix_loss_weight": args.prefix_loss_weight,
            "temporal_loss_weight": args.temporal_loss_weight,
            "temporal_margin": args.temporal_margin,
            "temporal_confidence_threshold": args.temporal_confidence_threshold,
            "temporal_temperature": args.temporal_temperature,
            "temporal_selection_mode": args.temporal_selection_mode,
            "prefix_accuracy_curve": prefix_metrics["prefix_accuracy_curve"],
            **{key: prefix_metrics[key] for key in (
                "final_accuracy", "mean_prefix_accuracy", "minimum_prefix_accuracy",
                "ever_regressed_fraction", "mean_population_regression", "mean_conditional_regression",
                "correct_to_wrong_transition_count", "destructive_transition_fraction",
                "ever_recovered_fraction", "wrong_to_correct_transition_count",
                "beneficial_transition_fraction", "stable_correct_fraction",
            )},
        }
        save_json(run_dir / "temporal_reliability_summary.json", temporal_summary)
    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "tmax": args.tmax,
        "checkpoint_selection": args.checkpoint_selection,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_acc,
        "best_validation_loss": best_val_loss,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "test_accuracy": last_eval.get("test_acc", 0.0),
        **last_eval,
        **model_state,
        **prefix_metrics,
    }
    save_json(run_dir / "summary.json", summary)
    plot_training_curves(metrics_path, run_dir / "plots")
    if model_state["timestep_gates"]:
        plot_timestep_gates(model_state["timestep_gates"], run_dir / "plots" / "final_timestep_gates.png")

    print("\nChronoSkip summary:")
    print(f"Test accuracy: {summary['test_accuracy']:.2f}%")
    print(f"Soft accuracy: {summary['soft_acc']:.2f}%")
    print(f"Hard accuracy: {summary['hard_acc']:.2f}%")
    print(f"Raw spike rate: {summary['raw_spike_rate']:.6f}")
    print(f"Gated spike rate: {summary['gated_spike_rate']:.6f}")
    print(f"Prefix spike rate: {summary['prefix_spike_rate']:.6f}")
    print(f"Effective timestep: {summary['effective_timestep']:.4f}")
    print(f"Hard effective timestep: {summary['hard_effective_timestep']:.4f}")
    print(f"Executed timestep: {summary['executed_timestep']:.4f}")
    print(f"Energy proxy: {summary['energy_proxy']:.6f}")
    print(f"Prefix energy proxy: {summary['prefix_energy_proxy']:.6f}")
    print(f"Loop energy proxy: {summary['loop_energy_proxy']:.6f}")
    print("Final timestep gates:", format_tree(model_state["timestep_gates"]))
    print("Final hard prefix masks:", format_tree(model_state["hard_prefix_masks"]))
    print("Final hard prefix steps:", format_tree(model_state["hard_prefix_steps"]))
    print("Hard budget proxy:", format_tree(model_state["hard_budget_proxy"]))
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
