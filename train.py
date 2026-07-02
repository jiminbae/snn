from __future__ import annotations

import argparse
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from models import build_model
from utils.data import build_dataloaders
from utils.logging import append_metrics, prepare_run_dir, save_json
from utils.metrics import AverageMeter, accuracy, energy_proxy
from utils.plotting import plot_timestep_gates, plot_training_curves

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
    parser.add_argument("--gate-init", type=float, default=5.0)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.02)
    parser.add_argument("--lambda-hard-budget", type=float, default=0.0)
    parser.add_argument("--hard-budget-sharpness", type=float, default=20.0)
    parser.add_argument("--target-timestep", type=float, default=0.0)
    parser.add_argument("--target-budget-weight", type=float, default=0.0)
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
    return parser.parse_args()


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    zero = logits.new_tensor(0.0)
    if args.model == "fixed_lif" or not hasattr(model, "hard_budget_proxy"):
        return zero, zero, zero

    hard_budget_proxy = model.hard_budget_proxy(
        gate_threshold=args.gate_threshold,
        sharpness=args.hard_budget_sharpness,
    )
    assert isinstance(hard_budget_proxy, torch.Tensor)
    hard_budget_cost = hard_budget_proxy / float(args.tmax)
    target_budget_loss = zero
    if args.target_timestep > 0.0 and args.target_budget_weight > 0.0:
        target = hard_budget_proxy.new_tensor(args.target_timestep)
        target_budget_loss = torch.relu(hard_budget_proxy - target) / float(args.tmax)
    return hard_budget_cost, target_budget_loss, hard_budget_proxy


def regularized_loss(
    args: argparse.Namespace,
    model: nn.Module,
    output: dict[str, Any],
    target: torch.Tensor,
    reg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    ce_loss = F.cross_entropy(logits, target)
    spike_cost = select_spike_cost(output, args.spike_cost_mode)
    effective_timestep = output["effective_timestep"]
    assert isinstance(effective_timestep, torch.Tensor)
    time_cost = effective_timestep / float(args.tmax)
    hard_budget_cost, target_budget_loss, hard_budget_proxy = hard_budget_terms(args, model, output)
    if args.model == "fixed_lif":
        total = ce_loss
    else:
        total = (
            ce_loss
            + reg_scale * args.lambda_spike * spike_cost
            + reg_scale * args.eta_time * time_cost
            + reg_scale * args.lambda_hard_budget * hard_budget_cost
            + reg_scale * args.target_budget_weight * target_budget_loss
        )
    return total, ce_loss, spike_cost, time_cost, hard_budget_cost, target_budget_loss, hard_budget_proxy


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
        "hard_budget_proxy",
        "acc",
        "acc_hard",
        "raw_spike_rate",
        "gated_spike_rate",
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
                hard_budget_cost, target_budget_loss, hard_budget_proxy = hard_budget_terms(args, model, soft_output)
                total = (
                    ce_soft
                    + args.hard_ce_weight * ce_hard
                    + args.consistency_weight * consistency
                    + reg_scale * args.lambda_spike * spike_cost
                    + reg_scale * args.eta_time * time_cost
                    + reg_scale * args.lambda_hard_budget * hard_budget_cost
                    + reg_scale * args.target_budget_weight * target_budget_loss
                )
            else:
                hard_output = None
                total, ce_soft, spike_cost, time_cost, hard_budget_cost, target_budget_loss, hard_budget_proxy = regularized_loss(
                    args, model, soft_output, target, reg_scale
                )
                ce_hard = images.new_tensor(0.0)
                consistency = images.new_tensor(0.0)

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
        meters["hard_budget_proxy"].update(hard_budget_proxy.detach().item(), batch_size)
        meters["acc"].update(accuracy(soft_logits.detach(), target), batch_size)
        if hard_output is not None:
            hard_logits = hard_output["logits"]
            assert isinstance(hard_logits, torch.Tensor)
            meters["acc_hard"].update(accuracy(hard_logits.detach(), target), batch_size)
        meters["raw_spike_rate"].update(output_to_float(soft_output, "raw_spike_rate"), batch_size)
        meters["gated_spike_rate"].update(output_to_float(soft_output, "gated_spike_rate"), batch_size)
        progress.set_postfix(loss=f"{meters['loss'].avg:.4f}", acc=f"{meters['acc'].avg:.2f}")

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    meter_names = [
        "test_acc",
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
    ]
    meters = {name: AverageMeter() for name in meter_names}
    progress = tqdm(loader, desc="test", leave=False)

    for batch_idx, (images, target) in enumerate(progress, start=1):
        if args.limit_test_batches and batch_idx > args.limit_test_batches:
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

        meters["test_acc"].update(accuracy(eval_logits, target), batch_size)
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
        progress.set_postfix(acc=f"{meters['test_acc'].avg:.2f}")

    return {key: meter.avg for key, meter in meters.items()}


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
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        requested = torch.device("cpu")
    device = requested
    set_seed(args.seed, device)

    run_name = args.run_name or f"{args.model}_{args.dataset}_T{args.tmax}_seed{args.seed}_{int(time.time())}"
    run_dir = prepare_run_dir(args.results_dir, run_name)
    config = vars(args).copy()
    config["resolved_device"] = str(device)
    save_json(run_dir / "config.json", config)

    event_downsample_size = args.event_downsample_size
    if args.dataset == "dvs_gesture" and event_downsample_size is None:
        event_downsample_size = 64
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
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        last_eval = evaluate(model, test_loader, device, args)
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
            "train_hard_budget_proxy": train_metrics["hard_budget_proxy"],
            "train_acc_soft": train_metrics["acc"],
            "train_acc_hard": train_metrics["acc_hard"],
            **last_eval,
        }
        append_metrics(metrics_path, row)
        print(
            f"epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
            f"test {last_eval['test_acc']:.2f}% | soft {last_eval['soft_acc']:.2f}% | hard {last_eval['hard_acc']:.2f}% | "
            f"raw {last_eval['raw_spike_rate']:.4f} | gated {last_eval['gated_spike_rate']:.4f} | "
            f"T {last_eval['effective_timestep']:.2f}/hard {last_eval['hard_effective_timestep']:.2f}/exec {last_eval['executed_timestep']:.2f} | "
            f"energy proxy {last_eval['energy_proxy']:.4f} | loop proxy {last_eval['loop_energy_proxy']:.4f}"
        )

    model_state = snapshot_model_state(model, args)
    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "tmax": args.tmax,
        "test_accuracy": last_eval.get("test_acc", 0.0),
        **last_eval,
        **model_state,
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
