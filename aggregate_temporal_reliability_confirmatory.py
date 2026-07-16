#!/usr/bin/env python3
"""Aggregate fixed-protocol N-MNIST temporal-reliability runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_METHODS = (
    "final_ce",
    "symmetric_kl",
    "selective_regression_thr0.6",
)
REQUIRED_SEEDS = (3, 4, 5)
PRIMARY_METRICS = (
    "final_accuracy",
    "mean_prefix_accuracy",
    "minimum_prefix_accuracy",
    "ever_regressed_fraction",
    "mean_population_regression",
    "mean_conditional_regression",
    "correct_to_wrong_transition_count",
    "destructive_transition_fraction",
    "ever_recovered_fraction",
    "wrong_to_correct_transition_count",
    "beneficial_transition_fraction",
)
OPTIONAL_METRICS = (
    "mean_stable_correct_timestep",
    "stable_by_t4_fraction",
    "stable_by_t6_fraction",
    "never_stable_fraction",
)
COMMON_PROTOCOL = {
    "dataset": "nmnist",
    "model": "fixed_lif",
    "epochs": 30,
    "batch_size": 32,
    "tmax": 8,
    "split_seed": 123,
    "val_ratio": 0.2,
    "checkpoint_selection": "best_val",
    "selection_metric": "val_acc",
    "event_frame_mode": "binary",
    "prefix_loss_weight": 0.0,
    "temporal_loss_weight": 1.0,
    "temporal_margin": 0.0,
    "temporal_temperature": 1.0,
    "temporal_selection_mode": "hard",
}


def summarize(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    deviation = 0.0
    if len(values) > 1:
        deviation = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
    return {
        "mean": mean,
        "sample_standard_deviation": deviation,
        "individual_seed_values": values,
        "valid_seed_count": len(values),
    }


def method_name(summary: dict[str, Any]) -> str:
    mode = str(summary["temporal_training_mode"])
    if mode == "selective_regression":
        threshold = float(summary.get("temporal_confidence_threshold", float("nan")))
        return f"selective_regression_thr{threshold:g}"
    return mode


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_confirmatory_config(
    run_dir: str | Path,
    config: dict[str, Any],
    method: str,
    *,
    allow_smoke: bool = False,
) -> None:
    expected = dict(COMMON_PROTOCOL)
    if allow_smoke:
        expected["epochs"] = 1
    mode = {
        "final_ce": "final_ce",
        "symmetric_kl": "symmetric_kl",
        "selective_regression_thr0.6": "selective_regression",
    }.get(method)
    if mode is None:
        raise ValueError(f"{run_dir}: method expected one of {REQUIRED_METHODS}, actual {method!r}")
    expected["temporal_training_mode"] = mode
    if method == "selective_regression_thr0.6":
        expected["temporal_confidence_threshold"] = 0.6
    for key, expected_value in expected.items():
        actual = config.get(key, "<missing>")
        if not _equal(actual, expected_value):
            raise ValueError(
                f"{run_dir}: config key {key!r} expected {expected_value!r}, "
                f"actual {actual!r}"
            )
    seed = config.get("seed")
    if seed not in REQUIRED_SEEDS:
        raise ValueError(
            f"{run_dir}: config key 'seed' expected one of {REQUIRED_SEEDS}, actual {seed!r}"
        )


def load_record(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir)
    summary = json.loads(
        (path / "temporal_reliability_summary.json").read_text(encoding="utf-8")
    )
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    method = method_name(summary)
    validate_confirmatory_config(path, config, method)
    missing = [metric for metric in PRIMARY_METRICS if metric not in summary]
    if missing:
        raise ValueError(f"{path}: summary missing required metrics: {missing}")
    return {
        "run_dir": str(path),
        "method": method,
        "seed": int(config["seed"]),
        "prefix_accuracy_curve": [float(value) for value in summary["prefix_accuracy_curve"]],
        **{
            metric: float(summary[metric])
            for metric in PRIMARY_METRICS + OPTIONAL_METRICS
            if metric in summary
        },
    }


def build_matched_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[int], dict[tuple[str, int], dict[str, Any]]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["method"]), int(record["seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate confirmatory result for method={key[0]} seed={key[1]}")
        indexed[key] = record
    matched = [
        seed
        for seed in REQUIRED_SEEDS
        if all((method, seed) in indexed for method in REQUIRED_METHODS)
    ]
    return matched, indexed


def compute_confirmatory_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    seeds, indexed = build_matched_records(records)
    empty = {
        "matched_seeds": seeds,
        "matched_seed_count": len(seeds),
        "regression_reduced_seed_count": 0,
        "aggregate_regression_reduction": None,
        "beneficial_transition_preservation_ratio": None,
        "ever_recovered_preservation_ratio": None,
        "final_accuracy_change_pp": None,
        "mean_prefix_accuracy_change_vs_final_ce": None,
        "minimum_prefix_accuracy_change_vs_final_ce": None,
        "selective_destructive_vs_symmetric": None,
        "selective_beneficial_vs_symmetric": None,
    }
    if not seeds:
        return empty
    rows = {
        method: [indexed[(method, seed)] for seed in seeds]
        for method in REQUIRED_METHODS
    }
    mean = lambda method, metric: summarize(
        [row[metric] for row in rows[method]]
    )["mean"]
    final = "final_ce"
    selective = "selective_regression_thr0.6"
    symmetric = "symmetric_kl"
    reduced_count = sum(
        indexed[(selective, seed)]["ever_regressed_fraction"]
        < indexed[(final, seed)]["ever_regressed_fraction"]
        for seed in seeds
    )
    empty.update(
        {
            "regression_reduced_seed_count": reduced_count,
            "aggregate_regression_reduction": (
                mean(final, "ever_regressed_fraction")
                - mean(selective, "ever_regressed_fraction")
            ),
            "beneficial_transition_preservation_ratio": (
                mean(selective, "beneficial_transition_fraction")
                / max(1e-8, mean(final, "beneficial_transition_fraction"))
            ),
            "ever_recovered_preservation_ratio": (
                mean(selective, "ever_recovered_fraction")
                / max(1e-8, mean(final, "ever_recovered_fraction"))
            ),
            "final_accuracy_change_pp": (
                mean(selective, "final_accuracy") - mean(final, "final_accuracy")
            ),
            "mean_prefix_accuracy_change_vs_final_ce": (
                mean(selective, "mean_prefix_accuracy")
                - mean(final, "mean_prefix_accuracy")
            ),
            "minimum_prefix_accuracy_change_vs_final_ce": (
                mean(selective, "minimum_prefix_accuracy")
                - mean(final, "minimum_prefix_accuracy")
            ),
            "selective_destructive_vs_symmetric": (
                mean(selective, "destructive_transition_fraction")
                - mean(symmetric, "destructive_transition_fraction")
            ),
            "selective_beneficial_vs_symmetric": (
                mean(selective, "beneficial_transition_fraction")
                - mean(symmetric, "beneficial_transition_fraction")
            ),
        }
    )
    return empty


def recommendation_from_records(
    records: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    values = compute_confirmatory_statistics(records)
    if values["matched_seed_count"] == 0:
        return "no_go", ["No complete matched confirmatory seeds were found."]
    if values["matched_seed_count"] < 3:
        return "no_go", [
            f"Matched seed count is {values['matched_seed_count']}; all 3 are required.",
            "No statistical significance is claimed.",
        ]
    strict = (
        values["regression_reduced_seed_count"] >= 2
        and values["aggregate_regression_reduction"] > 0
        and values["beneficial_transition_preservation_ratio"] >= 0.9
        and values["ever_recovered_preservation_ratio"] >= 0.9
        and values["final_accuracy_change_pp"] >= -0.5
        and values["selective_destructive_vs_symmetric"] < 0
        and values["selective_beneficial_vs_symmetric"] > 0
    )
    weak = (
        values["regression_reduced_seed_count"] >= 2
        and values["aggregate_regression_reduction"] > 0
        and values["beneficial_transition_preservation_ratio"] >= 0.8
        and values["ever_recovered_preservation_ratio"] >= 0.8
        and values["final_accuracy_change_pp"] >= -1.0
    )
    decision = "go" if strict else "weak_go" if weak else "no_go"
    return decision, [
        f"Regression reduced in {values['regression_reduced_seed_count']} of 3 seeds.",
        f"Aggregate regression reduction: {values['aggregate_regression_reduction']:.4f} pp.",
        "No statistical significance is claimed.",
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    default_fields: tuple[str, ...],
) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else list(default_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_outputs(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds, indexed = build_matched_records(records)
    stats_values = compute_confirmatory_statistics(records)
    decision, reasons = recommendation_from_records(records)
    active = [indexed[(method, seed)] for method in REQUIRED_METHODS for seed in seeds]
    metrics = list(PRIMARY_METRICS)
    metrics += [
        metric
        for metric in OPTIONAL_METRICS
        if active and all(metric in row for row in active)
    ]
    metric_rows = [
        {
            "method": method,
            "metric": metric,
            **summarize([indexed[(method, seed)][metric] for seed in seeds]),
        }
        for method in REQUIRED_METHODS
        for metric in metrics
        if seeds
    ]
    write_csv(
        output / "aggregate_temporal_metrics.csv",
        metric_rows,
        ("method", "metric", "mean", "sample_standard_deviation",
         "individual_seed_values", "valid_seed_count"),
    )
    comparisons: list[dict[str, Any]] = []
    for seed in seeds:
        final = indexed[("final_ce", seed)]
        symmetric = indexed[("symmetric_kl", seed)]
        for method in ("symmetric_kl", "selective_regression_thr0.6"):
            row = indexed[(method, seed)]
            item = {
                "method": method,
                "seed": seed,
                "regression_reduction_vs_final_ce": (
                    final["ever_regressed_fraction"] - row["ever_regressed_fraction"]
                ),
                "final_accuracy_change_vs_final_ce": (
                    row["final_accuracy"] - final["final_accuracy"]
                ),
                "mean_prefix_accuracy_change_vs_final_ce": (
                    row["mean_prefix_accuracy"] - final["mean_prefix_accuracy"]
                ),
                "minimum_prefix_accuracy_change_vs_final_ce": (
                    row["minimum_prefix_accuracy"] - final["minimum_prefix_accuracy"]
                ),
                "beneficial_transition_preservation_ratio": (
                    row["beneficial_transition_fraction"]
                    / max(1e-8, final["beneficial_transition_fraction"])
                ),
                "ever_recovered_preservation_ratio": (
                    row["ever_recovered_fraction"]
                    / max(1e-8, final["ever_recovered_fraction"])
                ),
            }
            if method == "selective_regression_thr0.6":
                item["destructive_transition_change_vs_symmetric_kl"] = (
                    row["destructive_transition_fraction"]
                    - symmetric["destructive_transition_fraction"]
                )
                item["beneficial_transition_change_vs_symmetric_kl"] = (
                    row["beneficial_transition_fraction"]
                    - symmetric["beneficial_transition_fraction"]
                )
            comparisons.append(item)
    write_csv(
        output / "seed_method_comparisons.csv",
        comparisons,
        ("method", "seed"),
    )
    aggregate_comparisons = []
    for method in ("symmetric_kl", "selective_regression_thr0.6"):
        rows = [row for row in comparisons if row["method"] == method]
        fields = sorted({key for row in rows for key in row} - {"method", "seed"})
        for metric in fields:
            aggregate_comparisons.append(
                {
                    "method": method,
                    "metric": metric,
                    **summarize([float(row[metric]) for row in rows if metric in row]),
                }
            )
    write_csv(
        output / "aggregate_method_comparisons.csv",
        aggregate_comparisons,
        ("method", "metric", "mean", "sample_standard_deviation",
         "individual_seed_values", "valid_seed_count"),
    )
    summary = {
        "recommendation": decision,
        "recommendation_reasons": reasons,
        "required_methods": list(REQUIRED_METHODS),
        "required_seeds": list(REQUIRED_SEEDS),
        "matched_methods": list(REQUIRED_METHODS) if seeds else [],
        **stats_values,
        "protocol_validation_passed": True,
    }
    (output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_plots(seeds, indexed, output)


def _write_plots(
    seeds: list[int],
    indexed: dict[tuple[str, int], dict[str, Any]],
    output: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    for method in REQUIRED_METHODS:
        curves = [indexed[(method, seed)]["prefix_accuracy_curve"] for seed in seeds]
        if curves:
            curve = [sum(values) / len(values) for values in zip(*curves)]
            plt.plot(range(1, len(curve) + 1), curve, "o-", label=method)
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    if seeds:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "prefix_accuracy_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for method in REQUIRED_METHODS:
        rows = [indexed[(method, seed)] for seed in seeds]
        if rows:
            plt.scatter(
                [row["destructive_transition_fraction"] for row in rows],
                [row["beneficial_transition_fraction"] for row in rows],
                label=method,
            )
    plt.xlabel("Destructive transitions (%)")
    plt.ylabel("Beneficial transitions (%)")
    plt.grid(alpha=0.3)
    if seeds:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output / "regression_recovery_tradeoff.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = [load_record(path) for path in args.run_dirs]
    build_matched_records(records)
    write_aggregate_outputs(records, args.output_dir)


if __name__ == "__main__":
    main()

