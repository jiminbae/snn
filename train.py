from __future__ import annotations

import argparse
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
from models.lif import DEFAULT_CANDIDATES
from models.spikegate import SpikeGateSNN
from utils.data import build_dataloaders
from utils.logging import append_metrics, prepare_run_dir, save_json
from utils.metrics import AverageMeter, accuracy, energy_proxy
from utils.plotting import (
    plot_candidate_probabilities,
    plot_timestep_gates,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed LIF and SpikeGate SNN prototypes.")
    parser.add_argument("--model", choices=["fixed_lif", "gate_only", "neuron_only", "softmax_spikegate", "spikegate"], default="fixed_lif")
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.02)
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--monotonic-gate", action="store_true")
    parser.add_argument("--hard-prefix-eval", action="store_true")
    parser.add_argument("--hard-prefix-unscaled", action="store_true")
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
    parser.add_argument("--spike-cost-mode", choices=["gated", "raw", "mixed"], default="gated")
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


def move_output_to_float(output: dict[str, Any], key: str) -> float:
    value = output[key]
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


def compute_loss(
    model_name: str,
    output: dict[str, Any],
    target: torch.Tensor,
    lambda_spike: float,
    eta_time: float,
    tmax: int,
    spike_cost_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    ce_loss = F.cross_entropy(logits, target)
    spike_cost = select_spike_cost(output, spike_cost_mode)
    effective_timestep = output["effective_timestep"]
    assert isinstance(spike_cost, torch.Tensor)
    assert isinstance(effective_timestep, torch.Tensor)
    time_cost = effective_timestep / float(tmax)

    if model_name == "fixed_lif":
        total = ce_loss
    else:
        # Spike/time regularization encourages lower spike activity and shorter
        # effective temporal computation, but remains only an energy proxy.
        total = ce_loss + lambda_spike * spike_cost + eta_time * time_cost
    return total, ce_loss, spike_cost, time_cost


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
    meters = {name: AverageMeter() for name in ["loss", "ce_loss", "spike_cost", "time_cost", "acc", "raw_spike_rate", "gated_spike_rate"]}
    amp_enabled = args.amp and device.type == "cuda"
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)

    for batch_idx, (images, target) in enumerate(progress, start=1):
        if args.limit_train_batches and batch_idx > args.limit_train_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(images, gumbel_tau=args.gumbel_tau, gate_threshold=args.gate_threshold)
            total, ce_loss, spike_cost, time_cost = compute_loss(
                args.model,
                output,
                target,
                args.lambda_spike * min(1.0, epoch / max(1, args.reg_warmup_epochs)),
                args.eta_time * min(1.0, epoch / max(1, args.reg_warmup_epochs)),
                args.tmax,
                args.spike_cost_mode,
            )

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        logits = output["logits"]
        assert isinstance(logits, torch.Tensor)
        meters["loss"].update(total.detach().item(), batch_size)
        meters["ce_loss"].update(ce_loss.detach().item(), batch_size)
        meters["spike_cost"].update(spike_cost.detach().item(), batch_size)
        meters["raw_spike_rate"].update(move_output_to_float(output, "raw_spike_rate"), batch_size)
        meters["gated_spike_rate"].update(move_output_to_float(output, "gated_spike_rate"), batch_size)
        meters["time_cost"].update(time_cost.detach().item(), batch_size)
        meters["acc"].update(accuracy(logits.detach(), target), batch_size)
        progress.set_postfix(loss=f"{meters['loss'].avg:.4f}", acc=f"{meters['acc'].avg:.2f}")

    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    meters = {
        name: AverageMeter()
        for name in [
            "acc",
            "raw_spike_rate",
            "gated_spike_rate",
            "spike_rate",
            "effective_timestep",
            "hard_effective_timestep",
            "prefix_spike_rate",
        ]
    }
    progress = tqdm(loader, desc="test", leave=False)

    for batch_idx, (images, target) in enumerate(progress, start=1):
        if args.limit_test_batches and batch_idx > args.limit_test_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        full_output = model(images, gumbel_tau=args.gumbel_tau, gate_threshold=args.gate_threshold)
        eval_output = full_output

        if args.hard_prefix_eval:
            hard_steps = int(round(move_output_to_float(full_output, "hard_effective_timestep")))
            hard_steps = max(0, min(args.tmax, hard_steps))
            eval_output = model(
                images,
                gumbel_tau=args.gumbel_tau,
                gate_threshold=args.gate_threshold,
                hard_prefix_steps=hard_steps,
                hard_prefix_unscaled=args.hard_prefix_unscaled,
            )

        logits = eval_output["logits"]
        assert isinstance(logits, torch.Tensor)
        batch_size = images.shape[0]
        meters["acc"].update(accuracy(logits, target), batch_size)
        meters["raw_spike_rate"].update(move_output_to_float(full_output, "raw_spike_rate"), batch_size)
        meters["gated_spike_rate"].update(move_output_to_float(full_output, "gated_spike_rate"), batch_size)
        meters["spike_rate"].update(move_output_to_float(full_output, "spike_rate"), batch_size)
        meters["effective_timestep"].update(move_output_to_float(full_output, "effective_timestep"), batch_size)
        meters["hard_effective_timestep"].update(move_output_to_float(full_output, "hard_effective_timestep"), batch_size)
        meters["prefix_spike_rate"].update(move_output_to_float(eval_output, "prefix_spike_rate"), batch_size)
        progress.set_postfix(acc=f"{meters['acc'].avg:.2f}")

    result = {key: meter.avg for key, meter in meters.items()}
    result["energy_proxy"] = energy_proxy(result["gated_spike_rate"], result["effective_timestep"])
    result["prefix_energy_proxy"] = energy_proxy(result["prefix_spike_rate"], result["hard_effective_timestep"])
    return result


def snapshot_model_state(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, SpikeGateSNN):
        probs = [[float(v) for v in p.detach().cpu().tolist()] for p in model.candidate_probabilities()]
        gates = [float(v) for v in model.timestep_gates().detach().cpu().tolist()]
        return {
            "selected_indices": model.selected_indices(),
            "selected_names": model.selected_names(),
            "candidate_probabilities": probs,
            "candidate_metadata": model.candidate_metadata(),
            "timestep_gates": gates,
        }
    return {
        "selected_indices": [],
        "selected_names": ["fixed_lif", "fixed_lif"],
        "candidate_probabilities": [],
        "candidate_metadata": [],
        "timestep_gates": [],
    }


def main() -> None:
    args = parse_args()
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        requested = torch.device("cpu")
    device = requested
    set_seed(args.seed, device)

    run_name = args.run_name or f"{args.model}_{args.dataset}_seed{args.seed}_{int(time.time())}"
    run_dir = prepare_run_dir(args.results_dir, run_name)
    config = vars(args).copy()
    config["resolved_device"] = str(device)
    save_json(run_dir / "config.json", config)

    train_loader, test_loader = build_dataloaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_model(
        args.model,
        dataset=args.dataset,
        tmax=args.tmax,
        monotonic_gate=args.monotonic_gate,
        gate_threshold=args.gate_threshold,
    ).to(device)
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
            "train_spike_cost": train_metrics["spike_cost"],
            "train_raw_spike_rate": train_metrics["raw_spike_rate"],
            "train_gated_spike_rate": train_metrics["gated_spike_rate"],
            "train_time_cost": train_metrics["time_cost"],
            "train_acc": train_metrics["acc"],
            "test_acc": last_eval["acc"],
            "test_spike_rate": last_eval["spike_rate"],
            "raw_spike_rate": last_eval["raw_spike_rate"],
            "gated_spike_rate": last_eval["gated_spike_rate"],
            "prefix_spike_rate": last_eval["prefix_spike_rate"],
            "effective_timestep": last_eval["effective_timestep"],
            "hard_effective_timestep": last_eval["hard_effective_timestep"],
            "energy_proxy": last_eval["energy_proxy"],
            "prefix_energy_proxy": last_eval["prefix_energy_proxy"],
        }
        append_metrics(metrics_path, row)
        state = snapshot_model_state(model)
        selected = ", ".join(state["selected_names"])
        print(
            f"epoch {epoch:03d} | loss {train_metrics['loss']:.4f} | "
            f"acc {last_eval['acc']:.2f}% | raw {last_eval['raw_spike_rate']:.4f} | gated {last_eval['gated_spike_rate']:.4f} | "
            f"T_eff {last_eval['effective_timestep']:.2f}/hard {last_eval['hard_effective_timestep']:.0f} | energy proxy {last_eval['energy_proxy']:.4f} | "
            f"neurons {selected}"
        )

    model_state = snapshot_model_state(model)
    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "test_accuracy": last_eval.get("acc", 0.0),
        "average_spike_rate": last_eval.get("spike_rate", 0.0),
        "raw_spike_rate": last_eval.get("raw_spike_rate", 0.0),
        "gated_spike_rate": last_eval.get("gated_spike_rate", 0.0),
        "prefix_spike_rate": last_eval.get("prefix_spike_rate", 0.0),
        "effective_timestep": last_eval.get("effective_timestep", 0.0),
        "hard_effective_timestep": last_eval.get("hard_effective_timestep", 0.0),
        "energy_proxy": last_eval.get("energy_proxy", 0.0),
        "prefix_energy_proxy": last_eval.get("prefix_energy_proxy", 0.0),
        **model_state,
    }
    save_json(run_dir / "summary.json", summary)
    plot_training_curves(metrics_path, run_dir / "plots")
    if model_state["timestep_gates"]:
        plot_timestep_gates(model_state["timestep_gates"], run_dir / "plots" / "final_timestep_gates.png")
    if model_state["candidate_probabilities"]:
        names = [candidate.name for candidate in DEFAULT_CANDIDATES]
        plot_candidate_probabilities(names, model_state["candidate_probabilities"], run_dir / "plots")

    print("\nSelected neuron candidates:")
    names = model_state["selected_names"]
    for idx, name in enumerate(names, start=1):
        print(f"Layer {idx}: {name}")
    print("\nTimestep gates:")
    print([round(v, 4) for v in model_state["timestep_gates"]] or [1.0] * args.tmax)
    print(f"Effective timestep: {summary['effective_timestep']:.4f}")
    print(f"Raw spike rate: {summary['raw_spike_rate']:.6f}")
    print(f"Gated spike rate: {summary['gated_spike_rate']:.6f}")
    print(f"Prefix spike rate: {summary['prefix_spike_rate']:.6f}")
    print(f"Energy proxy: {summary['energy_proxy']:.6f}")
    print(f"Prefix energy proxy: {summary['prefix_energy_proxy']:.6f}")
    print(f"Test accuracy: {summary['test_accuracy']:.2f}%")
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
