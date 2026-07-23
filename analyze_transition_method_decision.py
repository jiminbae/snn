#!/usr/bin/env python3
"""Train/validation/test decision experiment for three temporal-reliability lanes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.transition_method_decision import (
    DIRECTION_MODELS,
    FALSE_VETO_BUDGETS,
    DecisionGuardrails,
    aggregate_candidate_rows,
    build_direction_examples,
    decode_hidden_trajectory,
    direction_metrics,
    direction_selector_rollout,
    evaluate_candidate,
    recommend_branch,
    regression_survival_rows,
    simple_filter_rollout,
    simple_filter_specs,
    split_payload_to_trajectory,
    threshold_at_false_veto_budget,
    train_direction_ranker,
    train_hidden_rc_ced,
    validate_split_trajectories,
)
from utils.transition_selector import FEATURE_SCHEMA, standardize_train_validation


SEEDS = (3, 4, 5)
HIDDEN_CONFIGS = (
    {
        "candidate_id": "hidden_rc_ced_a0.05_d0.5_i0.02",
        "alpha_min": 0.05,
        "destructive_weight": 0.5,
        "intervention_weight": 0.02,
    },
    {
        "candidate_id": "hidden_rc_ced_a0.10_d1_i0.02",
        "alpha_min": 0.10,
        "destructive_weight": 1.0,
        "intervention_weight": 0.02,
    },
)
COLORS = {
    "simple_filter": "#4C78A8",
    "output_only_selector": "#F58518",
    "hidden_state_rc_ced": "#54A24B",
}


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, float) and not math.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def load_split_inputs(
    results_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    input_paths: list[str] = []
    missing: list[str] = []
    for seed in SEEDS:
        trajectory_root = results_root / "final_ce" / f"seed_{seed}" / "trajectories"
        for split in by_split:
            path = trajectory_root / f"{split}_trajectories.pt"
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            by_split[split].append(split_payload_to_trajectory(payload, seed))
            input_paths.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing split trajectories. Run run_transition_method_decision.py "
            "without --skip-export first:\n" + "\n".join(missing)
        )
    validate_split_trajectories(by_split, require_hidden=True)
    return by_split, input_paths


def select_validation_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if not row.get("control_only", False)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            bool(row.get("guardrail_pass")),
            row.get("regression_reduction_pp", -math.inf),
            row.get("pooled_matched_recovery_preservation", -math.inf),
            -int(row.get("complexity_rank", 99)),
        ),
    )


def evaluate_filter_spec(
    trajectories: list[dict[str, Any]],
    spec: Any,
    split: str,
    guardrails: DecisionGuardrails,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    survival: list[dict[str, Any]] = []
    for raw in trajectories:
        candidate, _ = simple_filter_rollout(raw, spec)
        seed = int(raw["seed"])
        seed_rows.append(
            evaluate_candidate(
                candidate,
                raw,
                family="simple_filter",
                candidate_id=spec.candidate_id,
                seed=seed,
                split=split,
                filter=spec.method,
                parameters=json.dumps(spec.parameter_dict, sort_keys=True),
            )
        )
        survival.extend(
            regression_survival_rows(
                candidate,
                raw,
                family="simple_filter",
                candidate_id=spec.candidate_id,
                seed=seed,
                split=split,
            )
        )
    aggregate = aggregate_candidate_rows(
        seed_rows,
        guardrails,
        family="simple_filter",
        candidate_id=spec.candidate_id,
        split=split,
        filter=spec.method,
        parameters=json.dumps(spec.parameter_dict, sort_keys=True),
        complexity_rank=spec.complexity_rank,
        control_only=spec.control_only,
    )
    if spec.control_only:
        aggregate["guardrail_pass"] = False
        aggregate["guardrail_failures"] = "control_only"
    return seed_rows, aggregate, survival


def run_simple_filters(
    by_split: dict[str, list[dict[str, Any]]],
    guardrails: DecisionGuardrails,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    survival: list[dict[str, Any]] = []
    specs = {spec.candidate_id: spec for spec in simple_filter_specs()}
    for spec in specs.values():
        _, aggregate, curve = evaluate_filter_spec(
            by_split["val"], spec, "val", guardrails
        )
        validation_rows.append(aggregate)
        survival.extend(curve)
    selected_validation = select_validation_candidate(validation_rows)
    if selected_validation is None:
        return validation_rows, [], None, survival
    selected_spec = specs[selected_validation["candidate_id"]]
    seed_rows, test_aggregate, curve = evaluate_filter_spec(
        by_split["test"], selected_spec, "test", guardrails
    )
    survival.extend(curve)
    test_aggregate.update(
        {
            "validation_guardrail_pass": selected_validation["guardrail_pass"],
            "test_guardrail_pass": test_aggregate["guardrail_pass"],
            "guardrail_pass": (
                selected_validation["guardrail_pass"]
                and test_aggregate["guardrail_pass"]
            ),
            "selected_on": "validation",
        }
    )
    return validation_rows, seed_rows, test_aggregate, survival


def concatenate_direction_examples(
    trajectories: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    examples = [build_direction_examples(row) for row in trajectories]
    return {
        key: torch.cat([example[key] for example in examples])
        for key in examples[0]
    }


def run_direction_selectors(
    by_split: dict[str, list[dict[str, Any]]],
    guardrails: DecisionGuardrails,
    *,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    train = concatenate_direction_examples(by_split["train"])
    validation = concatenate_direction_examples(by_split["val"])
    validation_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    survival: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float], tuple[Any, torch.Tensor, torch.Tensor, float]] = {}

    for model_name in DIRECTION_MODELS:
        train_scaled, validation_scaled, mean, std = standardize_train_validation(
            train["features"], validation["features"]
        )
        model, training = train_direction_ranker(
            model_name,
            train_scaled,
            train["target"],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        training_rows.append(training)
        with torch.no_grad():
            validation_scores = torch.sigmoid(model(validation_scaled))
        for budget in FALSE_VETO_BUDGETS:
            threshold = threshold_at_false_veto_budget(
                validation_scores, validation["target"], budget
            )
            teacher_rows.append(
                {
                    "model": model_name,
                    "split": "val",
                    "false_veto_budget": budget,
                    **direction_metrics(
                        validation_scores, validation["target"], threshold
                    ),
                }
            )
            score_fn = lambda features, m=model, mu=mean, sigma=std: torch.sigmoid(
                m((features - mu) / sigma)
            )
            seed_rows: list[dict[str, Any]] = []
            candidate_id = f"{model_name}__fpr-{budget:g}"
            for raw in by_split["val"]:
                candidate, _ = direction_selector_rollout(
                    raw, score_fn, threshold, candidate_id
                )
                seed_value = int(raw["seed"])
                seed_rows.append(
                    evaluate_candidate(
                        candidate,
                        raw,
                        family="output_only_selector",
                        candidate_id=candidate_id,
                        seed=seed_value,
                        split="val",
                        model=model_name,
                        false_veto_budget=budget,
                        threshold=threshold,
                    )
                )
                survival.extend(
                    regression_survival_rows(
                        candidate,
                        raw,
                        family="output_only_selector",
                        candidate_id=candidate_id,
                        seed=seed_value,
                        split="val",
                    )
                )
            aggregate = aggregate_candidate_rows(
                seed_rows,
                guardrails,
                family="output_only_selector",
                candidate_id=candidate_id,
                split="val",
                model=model_name,
                false_veto_budget=budget,
                threshold=threshold,
                complexity_rank=1 if model_name.startswith("linear") else 2,
            )
            validation_rows.append(aggregate)
            fitted[(model_name, budget)] = (model, mean, std, threshold)

    selected_validation = select_validation_candidate(validation_rows)
    if selected_validation is None:
        return training_rows, teacher_rows, validation_rows, None, survival
    key = (
        selected_validation["model"],
        float(selected_validation["false_veto_budget"]),
    )
    model, mean, std, threshold = fitted[key]
    score_fn = lambda features: torch.sigmoid(model((features - mean) / std))
    test_seed_rows: list[dict[str, Any]] = []
    for raw in by_split["test"]:
        candidate, _ = direction_selector_rollout(
            raw, score_fn, threshold, selected_validation["candidate_id"]
        )
        seed_value = int(raw["seed"])
        test_seed_rows.append(
            evaluate_candidate(
                candidate,
                raw,
                family="output_only_selector",
                candidate_id=selected_validation["candidate_id"],
                seed=seed_value,
                split="test",
                model=selected_validation["model"],
                false_veto_budget=selected_validation["false_veto_budget"],
                threshold=threshold,
            )
        )
        survival.extend(
            regression_survival_rows(
                candidate,
                raw,
                family="output_only_selector",
                candidate_id=selected_validation["candidate_id"],
                seed=seed_value,
                split="test",
            )
        )
    test_aggregate = aggregate_candidate_rows(
        test_seed_rows,
        guardrails,
        family="output_only_selector",
        candidate_id=selected_validation["candidate_id"],
        split="test",
        model=selected_validation["model"],
        false_veto_budget=selected_validation["false_veto_budget"],
        threshold=threshold,
    )
    test_aggregate.update(
        {
            "validation_guardrail_pass": selected_validation["guardrail_pass"],
            "test_guardrail_pass": test_aggregate["guardrail_pass"],
            "guardrail_pass": (
                selected_validation["guardrail_pass"]
                and test_aggregate["guardrail_pass"]
            ),
            "selected_on": "validation",
        }
    )
    test_aggregate["_seed_rows"] = test_seed_rows
    return training_rows, teacher_rows, validation_rows, test_aggregate, survival


def evaluate_hidden_model(
    trajectories: list[dict[str, Any]],
    model: Any,
    config: dict[str, Any],
    split: str,
    guardrails: DecisionGuardrails,
    *,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    survival: list[dict[str, Any]] = []
    for raw in trajectories:
        candidate, alpha = decode_hidden_trajectory(
            raw,
            model,
            device=device,
            batch_size=batch_size,
            method=config["candidate_id"],
            metadata=config,
        )
        seed_value = int(raw["seed"])
        seed_rows.append(
            evaluate_candidate(
                candidate,
                raw,
                family="hidden_state_rc_ced",
                candidate_id=config["candidate_id"],
                seed=seed_value,
                split=split,
                alpha_mean=float(alpha[:, 1:].mean()),
                **{
                    key: value
                    for key, value in config.items()
                    if key != "candidate_id"
                },
            )
        )
        survival.extend(
            regression_survival_rows(
                candidate,
                raw,
                family="hidden_state_rc_ced",
                candidate_id=config["candidate_id"],
                seed=seed_value,
                split=split,
            )
        )
    aggregate = aggregate_candidate_rows(
        seed_rows,
        guardrails,
        family="hidden_state_rc_ced",
        candidate_id=config["candidate_id"],
        split=split,
        complexity_rank=3,
        **{key: value for key, value in config.items() if key != "candidate_id"},
    )
    return seed_rows, aggregate, survival


def run_hidden_rc_ced(
    by_split: dict[str, list[dict[str, Any]]],
    guardrails: DecisionGuardrails,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    survival: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    model_dir = output_dir / "cache" / "hidden_rc_ced_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for config in HIDDEN_CONFIGS:
        model, training = train_hidden_rc_ced(
            by_split["train"],
            alpha_min=config["alpha_min"],
            destructive_weight=config["destructive_weight"],
            intervention_weight=config["intervention_weight"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=seed,
            device=device,
        )
        training_rows.append({"candidate_id": config["candidate_id"], **training})
        models[config["candidate_id"]] = model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "training": training,
            },
            model_dir / f"{config['candidate_id']}.pt",
        )
        _, aggregate, curve = evaluate_hidden_model(
            by_split["val"],
            model,
            config,
            "val",
            guardrails,
            device=device,
            batch_size=evaluation_batch_size,
        )
        validation_rows.append(aggregate)
        survival.extend(curve)

    selected_validation = select_validation_candidate(validation_rows)
    if selected_validation is None:
        return training_rows, validation_rows, None, survival
    config = next(
        row
        for row in HIDDEN_CONFIGS
        if row["candidate_id"] == selected_validation["candidate_id"]
    )
    seed_rows, test_aggregate, curve = evaluate_hidden_model(
        by_split["test"],
        models[selected_validation["candidate_id"]],
        config,
        "test",
        guardrails,
        device=device,
        batch_size=evaluation_batch_size,
    )
    survival.extend(curve)
    test_aggregate.update(
        {
            "validation_guardrail_pass": selected_validation["guardrail_pass"],
            "test_guardrail_pass": test_aggregate["guardrail_pass"],
            "guardrail_pass": (
                selected_validation["guardrail_pass"]
                and test_aggregate["guardrail_pass"]
            ),
            "selected_on": "validation",
            "_seed_rows": seed_rows,
        }
    )
    return training_rows, validation_rows, test_aggregate, survival


def plot_tradeoff(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(8, 5))
    for family, color in COLORS.items():
        selected = [row for row in rows if row.get("family") == family]
        if not selected:
            continue
        axis.scatter(
            [row.get("pooled_matched_recovery_preservation", math.nan) for row in selected],
            [row.get("regression_reduction_pp", math.nan) for row in selected],
            label=family,
            color=color,
            alpha=0.75,
        )
    axis.axvline(0.95, color="gray", linestyle="--", label="recovery guardrail")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Validation branch candidates",
        xlabel="Pooled matched recovery preservation",
        ylabel="Matched regression reduction (percentage points)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_regression_recovery_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_test_survival(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(8, 5))
    raw_by_timestep: dict[int, list[float]] = {}
    for row in test_rows:
        raw_by_timestep.setdefault(int(row["timestep"]), []).append(
            float(row["raw_regression_free_survival"])
        )
    axis.plot(
        sorted(raw_by_timestep),
        [
            sum(raw_by_timestep[timestep]) / len(raw_by_timestep[timestep])
            for timestep in sorted(raw_by_timestep)
        ],
        color="black",
        linestyle="--",
        label="raw final_ce",
    )
    for family, color in COLORS.items():
        family_rows = [row for row in test_rows if row["family"] == family]
        by_timestep: dict[int, list[float]] = {}
        for row in family_rows:
            by_timestep.setdefault(int(row["timestep"]), []).append(
                float(row["candidate_regression_free_survival"])
            )
        if by_timestep:
            axis.plot(
                sorted(by_timestep),
                [
                    sum(by_timestep[timestep]) / len(by_timestep[timestep])
                    for timestep in sorted(by_timestep)
                ],
                marker="o",
                color=color,
                label=family,
            )
    axis.set(
        title="Selected candidates: regression-free survival on test",
        xlabel="Prefix timestep",
        ylabel="Fraction without an observed C-to-W transition",
        ylim=(0.0, 1.01),
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selected_test_regression_survival.png", dpi=180)
    plt.close(fig)


def strip_private(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/temporal_reliability_nmnist_confirmatory"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--direction-epochs", type=int, default=200)
    parser.add_argument("--direction-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-epochs", type=int, default=40)
    parser.add_argument("--hidden-batch-size", type=int, default=1024)
    parser.add_argument("--evaluation-batch-size", type=int, default=2048)
    parser.add_argument("--training-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    output_dir = args.results_root / "transition_method_decision"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split, input_paths = load_split_inputs(args.results_root)
    mechanism_path = args.results_root / "mechanism_analysis" / "mechanism_summary.json"
    if not mechanism_path.is_file():
        raise FileNotFoundError(f"Missing mechanism summary: {mechanism_path}")
    mechanism = json.loads(mechanism_path.read_text())
    required_improvement = 0.5 * abs(
        float(mechanism["matched_regression_difference_mean"])
    )
    guardrails = DecisionGuardrails(required_improvement)

    simple_validation, simple_test_seed, simple_test, simple_survival = run_simple_filters(
        by_split, guardrails
    )
    (
        direction_training,
        direction_teacher,
        direction_validation,
        direction_test,
        direction_survival,
    ) = run_direction_selectors(
        by_split,
        guardrails,
        epochs=args.direction_epochs,
        batch_size=args.direction_batch_size,
        device=args.device,
        seed=args.training_seed,
    )
    (
        hidden_training,
        hidden_validation,
        hidden_test,
        hidden_survival,
    ) = run_hidden_rc_ced(
        by_split,
        guardrails,
        output_dir,
        epochs=args.hidden_epochs,
        batch_size=args.hidden_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        device=args.device,
        seed=args.training_seed,
    )

    selected_test_seed_rows = list(simple_test_seed)
    if direction_test:
        selected_test_seed_rows.extend(direction_test.pop("_seed_rows", []))
    if hidden_test:
        selected_test_seed_rows.extend(hidden_test.pop("_seed_rows", []))
    validation_candidates = (
        simple_validation + direction_validation + hidden_validation
    )
    selected_test = [
        row for row in (simple_test, direction_test, hidden_test) if row is not None
    ]
    support_ok = all(
        row.get("pooled_raw_regression_count", 0)
        >= guardrails.minimum_pooled_raw_regression_count
        and row.get("pooled_raw_recovery_count", 0)
        >= guardrails.minimum_pooled_raw_recovery_count
        for row in selected_test
    )
    decision = recommend_branch(
        simple_test,
        direction_test,
        hidden_test,
        sufficient_support=support_ok,
    )
    survival_rows = simple_survival + direction_survival + hidden_survival

    write_csv(output_dir / "simple_filter_validation_sweep.csv", simple_validation)
    write_csv(output_dir / "direction_ranker_training.csv", direction_training)
    write_csv(output_dir / "direction_ranker_validation_metrics.csv", direction_teacher)
    write_csv(output_dir / "output_selector_validation_candidates.csv", direction_validation)
    write_csv(output_dir / "hidden_rc_ced_training.csv", hidden_training)
    write_csv(output_dir / "hidden_rc_ced_validation_candidates.csv", hidden_validation)
    write_csv(output_dir / "candidate_decision_table.csv", validation_candidates)
    write_csv(output_dir / "selected_test_seed_metrics.csv", selected_test_seed_rows)
    write_csv(output_dir / "selected_test_aggregate_metrics.csv", selected_test)
    write_csv(output_dir / "regression_survival.csv", survival_rows)
    write_json(
        output_dir / "hidden_feature_schema.json",
        by_split["train"][0]["hidden_feature_metadata"],
    )
    plot_tradeoff(output_dir, validation_candidates)
    plot_test_survival(output_dir, survival_rows)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    summary = {
        "analysis_type": "train_validation_test_transition_method_decision",
        "decision_status": "post_hoc_branch_recommendation",
        "decision": decision,
        "source_commit": commit,
        "source_results_root": str(args.results_root),
        "input_files": input_paths + [str(mechanism_path)],
        "input_validation": validate_split_trajectories(
            by_split, require_hidden=True
        ),
        "protocol": {
            "training_split": "official saved train split",
            "selection_and_calibration_split": "official saved validation split",
            "test_policy": "one validation-selected candidate per lane",
            "selection_uses_test": False,
            "seed_models": list(SEEDS),
            "shared_training_across_backbone_seeds": True,
            "direction_positive": "C_TO_W",
            "direction_negative": "W_TO_C",
            "false_veto_budgets": list(FALSE_VETO_BUDGETS),
            "simple_filter_grid_predeclared": True,
            "hidden_configs_predeclared": list(HIDDEN_CONFIGS),
        },
        "guardrails": guardrails.as_dict(),
        "selected_test_lanes": {
            "simple_filter": strip_private(simple_test),
            "output_only_selector": strip_private(direction_test),
            "hidden_state_rc_ced": strip_private(hidden_test),
        },
        "validation_candidate_count": len(validation_candidates),
        "target_free_inference": {
            "output_features": all(
                not feature["uses_target"] for feature in FEATURE_SCHEMA
            ),
            "simple_filters": True,
            "hidden_features": True,
        },
        "limitations": [
            "The method family was designed after inspecting earlier N-MNIST test trajectories, so this remains post-hoc research guidance.",
            "Validation selects one fixed candidate per lane; test metrics do not select thresholds or configurations.",
            "Three backbone seeds are reported separately, while the learned post-processors share train data across seeds.",
            "The hidden decoder is a compact adaptive evidence filter, not an unrestricted recurrent model.",
        ],
        "artifacts": [
            "simple_filter_validation_sweep.csv",
            "direction_ranker_training.csv",
            "direction_ranker_validation_metrics.csv",
            "output_selector_validation_candidates.csv",
            "hidden_rc_ced_training.csv",
            "hidden_rc_ced_validation_candidates.csv",
            "candidate_decision_table.csv",
            "selected_test_seed_metrics.csv",
            "selected_test_aggregate_metrics.csv",
            "regression_survival.csv",
            "hidden_feature_schema.json",
            "validation_regression_recovery_tradeoff.png",
            "selected_test_regression_survival.png",
        ],
    }
    write_json(output_dir / "transition_method_decision_summary.json", summary)
    print(json.dumps(safe(decision), indent=2))
    print(f"Wrote decision analysis to {output_dir}")


if __name__ == "__main__":
    main()
