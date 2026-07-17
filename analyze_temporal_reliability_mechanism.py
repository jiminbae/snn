#!/usr/bin/env python3
"""Post-hoc matched-state analysis of fixed N-MNIST confirmatory runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from export_confirmatory_prefix_trajectories import validate_alignment
from utils.prefix_metrics import first_correct_timestep, stable_correct_timestep
from utils.trajectory_export import load_torch_compat

MIN_BIN_SUPPORT = 100
CONFIDENCE_BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.000001))
PRIMARY = "final_ce"
SECONDARY = "symmetric_kl"
SELECTIVE = "selective_regression_thr0.6"
SEEDS = (3, 4, 5)


def rate(count: int, support: int) -> float:
    return 100.0 * count / support if support else float("nan")


def safe_mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def sample_std(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) < 2:
        return 0.0
    mean = sum(valid) / len(valid)
    return math.sqrt(sum((value - mean) ** 2 for value in valid) / (len(valid) - 1))


def paired_state_rows(
    selective: dict[str, Any],
    comparator: dict[str, Any],
    comparator_name: str,
) -> dict[str, list[dict[str, Any]]]:
    validate_alignment([selective, comparator])
    sc = selective["correct"].bool()
    cc = comparator["correct"].bool()
    sp = selective["true_class_probability"].float()
    cp = comparator["true_class_probability"].float()
    seed = int(selective["seed"])
    regression = []
    recovery = []
    probability = []
    confidence = []
    opportunity = []
    for t in range(sc.shape[1] - 1):
        common_correct = sc[:, t] & cc[:, t]
        s_reg = common_correct & ~sc[:, t + 1]
        c_reg = common_correct & ~cc[:, t + 1]
        support = int(common_correct.sum())
        both_regress = int((s_reg & c_reg).sum())
        s_only = int((s_reg & ~c_reg).sum())
        c_only = int((~s_reg & c_reg).sum())
        regression.append({
            "comparison": comparator_name, "seed": seed, "from_t": t + 1, "to_t": t + 2,
            "common_correct_count": support,
            "both_remain_correct_count": support - both_regress - s_only - c_only,
            "selective_only_regresses_count": s_only,
            "comparator_only_regresses_count": c_only,
            "both_regress_count": both_regress,
            "selective_matched_regression_rate": rate(int(s_reg.sum()), support),
            "comparator_matched_regression_rate": rate(int(c_reg.sum()), support),
            "matched_regression_difference": rate(int(s_reg.sum()), support) - rate(int(c_reg.sum()), support),
        })
        common_wrong = ~sc[:, t] & ~cc[:, t]
        s_rec = common_wrong & sc[:, t + 1]
        c_rec = common_wrong & cc[:, t + 1]
        wrong_support = int(common_wrong.sum())
        both_recover = int((s_rec & c_rec).sum())
        s_only_rec = int((s_rec & ~c_rec).sum())
        c_only_rec = int((~s_rec & c_rec).sum())
        recovery.append({
            "comparison": comparator_name, "seed": seed, "from_t": t + 1, "to_t": t + 2,
            "common_wrong_count": wrong_support,
            "both_remain_wrong_count": wrong_support - both_recover - s_only_rec - c_only_rec,
            "selective_only_recovers_count": s_only_rec,
            "comparator_only_recovers_count": c_only_rec,
            "both_recover_count": both_recover,
            "selective_matched_recovery_rate": rate(int(s_rec.sum()), wrong_support),
            "comparator_matched_recovery_rate": rate(int(c_rec.sum()), wrong_support),
            "matched_recovery_difference": rate(int(s_rec.sum()), wrong_support) - rate(int(c_rec.sum()), wrong_support),
        })
        eligible = common_correct & (sp[:, t] >= 0.6) & (cp[:, t] >= 0.6)
        probability.append(probability_row(selective, comparator, comparator_name, seed, t, common_correct, "common_correct"))
        probability.append(probability_row(selective, comparator, comparator_name, seed, t, eligible, "eligible_common_correct"))
        eligible_support = int(eligible.sum())
        confidence.append({
            "comparison": comparator_name, "seed": seed, "from_t": t + 1, "to_t": t + 2,
            "bin": "eligible>=0.6", "support_count": eligible_support,
            "selective_regression_rate": rate(int((eligible & ~sc[:, t + 1]).sum()), eligible_support),
            "comparator_regression_rate": rate(int((eligible & ~cc[:, t + 1]).sum()), eligible_support),
            "paired_difference": rate(int((eligible & ~sc[:, t + 1]).sum()), eligible_support) - rate(int((eligible & ~cc[:, t + 1]).sum()), eligible_support),
            "interpretable": eligible_support >= MIN_BIN_SUPPORT,
        })
        for low, high in CONFIDENCE_BINS:
            mask = common_correct & (sp[:, t] >= low) & (sp[:, t] < high) & (cp[:, t] >= low) & (cp[:, t] < high)
            bin_support = int(mask.sum())
            s_value = rate(int((mask & ~sc[:, t + 1]).sum()), bin_support)
            c_value = rate(int((mask & ~cc[:, t + 1]).sum()), bin_support)
            confidence.append({
                "comparison": comparator_name, "seed": seed, "from_t": t + 1, "to_t": t + 2,
                "bin": f"[{low},{high})", "support_count": bin_support,
                "selective_regression_rate": s_value if bin_support >= MIN_BIN_SUPPORT else float("nan"),
                "comparator_regression_rate": c_value if bin_support >= MIN_BIN_SUPPORT else float("nan"),
                "paired_difference": s_value - c_value if bin_support >= MIN_BIN_SUPPORT else float("nan"),
                "interpretable": bin_support >= MIN_BIN_SUPPORT,
            })
        for method, correct in ((SELECTIVE, sc), (comparator_name, cc)):
            correct_support = int(correct[:, t].sum())
            destructive = int((correct[:, t] & ~correct[:, t + 1]).sum())
            opportunity.append({
                "comparison": comparator_name, "method": method, "seed": seed,
                "from_t": t + 1, "to_t": t + 2,
                "correct_opportunity_rate": rate(correct_support, correct.shape[0]),
                "conditional_regression_rate": rate(destructive, correct_support),
                "population_destructive_rate": rate(destructive, correct.shape[0]),
            })
    return {"regression": regression, "recovery": recovery, "probability": probability, "confidence": confidence, "opportunity": opportunity}


def probability_row(selective, comparator, comparator_name, seed, t, mask, subset):
    support = int(mask.sum())
    values = {}
    for prefix, trajectory in (("selective", selective), ("comparator", comparator)):
        probability = trajectory["true_class_probability"].float()
        delta = probability[:, t + 1] - probability[:, t]
        drops = (-delta).clamp_min(0)
        selected_delta = delta[mask]
        selected_drops = drops[mask]
        values.update({
            f"{prefix}_mean_delta_p_y": float(selected_delta.mean()) if support else float("nan"),
            f"{prefix}_median_delta_p_y": float(selected_delta.median()) if support else float("nan"),
            f"{prefix}_probability_drop_fraction": float((selected_delta < 0).float().mean()) if support else float("nan"),
            f"{prefix}_mean_drop_magnitude": float(selected_drops.mean()) if support else float("nan"),
            f"{prefix}_large_drop_fraction_0.05": float((selected_drops >= 0.05).float().mean()) if support else float("nan"),
            f"{prefix}_large_drop_fraction_0.10": float((selected_drops >= 0.10).float().mean()) if support else float("nan"),
        })
    return {
        "comparison": comparator_name, "seed": seed, "from_t": t + 1, "to_t": t + 2,
        "subset": subset, "support_count": support, **values,
        "mean_drop_magnitude_difference": values["selective_mean_drop_magnitude"] - values["comparator_mean_drop_magnitude"],
        "probability_drop_fraction_difference": values["selective_probability_drop_fraction"] - values["comparator_probability_drop_fraction"],
    }


def timing_row(selective, comparator, comparator_name):
    def logits(t):
        return t["prefix_logits"]
    targets = selective["targets"]
    sf = first_correct_timestep(logits(selective), targets).float()
    cf = first_correct_timestep(logits(comparator), targets).float()
    ss = stable_correct_timestep(logits(selective), targets).float()
    cs = stable_correct_timestep(logits(comparator), targets).float()
    result = {"comparison": comparator_name, "seed": int(selective["seed"])}
    for name, difference in (("first_correct", sf - cf), ("stable_correct", ss - cs)):
        result.update({
            f"{name}_delay_mean": float(difference.mean()),
            f"{name}_delay_median": float(difference.median()),
            f"{name}_delay_q25": float(torch.quantile(difference, 0.25)),
            f"{name}_delay_q75": float(torch.quantile(difference, 0.75)),
            f"{name}_fraction_delayed": float((difference > 0).float().mean()),
            f"{name}_fraction_accelerated": float((difference < 0).float().mean()),
            f"{name}_fraction_unchanged": float((difference == 0).float().mean()),
        })
    return result


def seed_metrics(rows):
    regression, recovery, confidence, probability = rows["regression"], rows["recovery"], rows["confidence"], rows["probability"]
    reg_support = sum(row["common_correct_count"] for row in regression)
    s_reg = sum(row["selective_only_regresses_count"] + row["both_regress_count"] for row in regression)
    c_reg = sum(row["comparator_only_regresses_count"] + row["both_regress_count"] for row in regression)
    rec_support = sum(row["common_wrong_count"] for row in recovery)
    s_rec = sum(row["selective_only_recovers_count"] + row["both_recover_count"] for row in recovery)
    c_rec = sum(row["comparator_only_recovers_count"] + row["both_recover_count"] for row in recovery)
    eligible = [row for row in confidence if row["bin"] == "eligible>=0.6"]
    eligible_support = sum(row["support_count"] for row in eligible)
    eligible_s = sum(round(row["selective_regression_rate"] * row["support_count"] / 100) for row in eligible if row["support_count"])
    eligible_c = sum(round(row["comparator_regression_rate"] * row["support_count"] / 100) for row in eligible if row["support_count"])
    common_probability = [row for row in probability if row["subset"] == "common_correct"]
    seed = regression[0]["seed"] if regression else recovery[0]["seed"]
    return {
        "comparison": regression[0]["comparison"] if regression else recovery[0]["comparison"],
        "seed": seed,
        "common_correct_support": reg_support,
        "common_wrong_support": rec_support,
        "eligible_common_correct_support": eligible_support,
        "micro_matched_regression_difference": rate(s_reg, reg_support) - rate(c_reg, reg_support),
        "macro_matched_regression_difference": safe_mean([row["matched_regression_difference"] for row in regression]),
        "valid_regression_transition_count": sum(not math.isnan(row["matched_regression_difference"]) for row in regression),
        "micro_matched_recovery_difference": rate(s_rec, rec_support) - rate(c_rec, rec_support),
        "selective_matched_recovery_rate": rate(s_rec, rec_support),
        "comparator_matched_recovery_rate": rate(c_rec, rec_support),
        "recovery_preservation_ratio": (s_rec / max(1, c_rec)),
        "eligible_micro_regression_difference": rate(eligible_s, eligible_support) - rate(eligible_c, eligible_support),
        "mean_drop_magnitude_difference": safe_mean([row["mean_drop_magnitude_difference"] for row in common_probability]),
        "probability_drop_fraction_difference": safe_mean([row["probability_drop_fraction_difference"] for row in common_probability]),
    }


def mechanism_recommendation(primary_seed_rows):
    if len(primary_seed_rows) < 3:
        return "insufficient_support", ["Three aligned training seeds are required."]
    if any(row["common_correct_support"] == 0 or row["common_wrong_support"] == 0 for row in primary_seed_rows):
        return "insufficient_support", ["Primary matched-state support is absent for at least one seed."]
    reg = [row["micro_matched_regression_difference"] for row in primary_seed_rows]
    eligible = [row["eligible_micro_regression_difference"] for row in primary_seed_rows]
    drop = [row["mean_drop_magnitude_difference"] for row in primary_seed_rows]
    drop_fraction = [row["probability_drop_fraction_difference"] for row in primary_seed_rows]
    recovery = [row["recovery_preservation_ratio"] for row in primary_seed_rows]
    reg_ok = sum(value < 0 for value in reg) >= 2 and safe_mean(reg) < 0
    eligible_ok = sum(value < 0 for value in eligible) >= 2 and safe_mean(eligible) < 0
    protection_ok = (
        (sum(value < 0 for value in drop) >= 2 and safe_mean(drop) < 0)
        or (sum(value < 0 for value in drop_fraction) >= 2 and safe_mean(drop_fraction) < 0)
    )
    recovery_ok = safe_mean(recovery) >= 0.9 and min(recovery) >= 0.8
    if reg_ok and eligible_ok and protection_ok and recovery_ok:
        decision = "mechanism_supported"
    elif not reg_ok or safe_mean(eligible) >= 0:
        decision = "mechanism_not_supported"
    else:
        decision = "partial_support"
    return decision, [
        f"Common-correct regression direction passed: {reg_ok}.",
        f"Eligibility-matched direction passed: {eligible_ok}.",
        f"Direct probability protection passed: {protection_ok}.",
        f"Matched recovery preservation passed: {recovery_ok}.",
        "This is a post-hoc mechanistic analysis, not independent confirmatory evidence.",
    ]


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def load_trajectory(root, method, seed):
    path = root / "mechanism_analysis" / "trajectories" / method / f"seed_{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_torch_compat(path)


def analyze(results_root: Path) -> dict[str, Any]:
    output = results_root / "mechanism_analysis"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = {key: [] for key in ("regression", "recovery", "probability", "confidence", "opportunity")}
    timing = []
    micro = []
    for comparator in (PRIMARY, SECONDARY):
        for seed in SEEDS:
            selective = load_trajectory(results_root, SELECTIVE, seed)
            other = load_trajectory(results_root, comparator, seed)
            rows = paired_state_rows(selective, other, comparator)
            for key in all_rows:
                all_rows[key].extend(rows[key])
            timing.append(timing_row(selective, other, comparator))
            micro.append(seed_metrics(rows))
    primary = [row for row in micro if row["comparison"] == PRIMARY]
    decision, reasons = mechanism_recommendation(primary)
    macro = [{
        "comparison": comparator,
        "metric": metric,
        "mean": safe_mean([row[metric] for row in micro if row["comparison"] == comparator]),
        "sample_standard_deviation": sample_std([row[metric] for row in micro if row["comparison"] == comparator]),
        "same_direction_seed_count": sum(row[metric] < 0 for row in micro if row["comparison"] == comparator),
    } for comparator in (PRIMARY, SECONDARY) for metric in (
        "macro_matched_regression_difference", "eligible_micro_regression_difference",
        "mean_drop_magnitude_difference", "probability_drop_fraction_difference")]
    write_csv(output / "per_seed_micro_metrics.csv", micro)
    write_csv(output / "per_seed_macro_metrics.csv", macro)
    write_csv(output / "per_transition_matched_regression.csv", all_rows["regression"])
    write_csv(output / "per_transition_matched_recovery.csv", all_rows["recovery"])
    write_csv(output / "confidence_matched_regression.csv", all_rows["confidence"])
    write_csv(output / "probability_drop_analysis.csv", all_rows["probability"])
    write_csv(output / "opportunity_decomposition.csv", all_rows["opportunity"])
    write_csv(output / "first_stable_timestep_analysis.csv", timing)
    write_csv(output / "bootstrap_intervals.csv", [])
    summary = {
        "analysis_type": "post_hoc_mechanistic_analysis",
        "primary_comparison": f"{SELECTIVE} vs {PRIMARY}",
        "secondary_comparison": f"{SELECTIVE} vs {SECONDARY}",
        "matched_seeds": list(SEEDS), "matched_seed_count": len(SEEDS),
        "common_correct_support_by_seed": [row["common_correct_support"] for row in primary],
        "common_wrong_support_by_seed": [row["common_wrong_support"] for row in primary],
        "eligible_common_correct_support_by_seed": [row["eligible_common_correct_support"] for row in primary],
        "matched_regression_difference_by_seed": [row["micro_matched_regression_difference"] for row in primary],
        "matched_regression_difference_mean": safe_mean([row["micro_matched_regression_difference"] for row in primary]),
        "matched_regression_same_direction_count": sum(row["micro_matched_regression_difference"] < 0 for row in primary),
        "eligible_regression_difference_by_seed": [row["eligible_micro_regression_difference"] for row in primary],
        "eligible_regression_difference_mean": safe_mean([row["eligible_micro_regression_difference"] for row in primary]),
        "eligible_same_direction_count": sum(row["eligible_micro_regression_difference"] < 0 for row in primary),
        "matched_recovery_preservation_by_seed": [row["recovery_preservation_ratio"] for row in primary],
        "matched_recovery_preservation_mean": safe_mean([row["recovery_preservation_ratio"] for row in primary]),
        "probability_drop_difference_by_seed": [row["mean_drop_magnitude_difference"] for row in primary],
        "first_correct_delay_by_seed": [row["first_correct_delay_mean"] for row in timing if row["comparison"] == PRIMARY],
        "stable_correct_delay_by_seed": [row["stable_correct_delay_mean"] for row in timing if row["comparison"] == PRIMARY],
        "recommendation": decision, "recommendation_reasons": reasons,
        "limitations": [
            "The analysis was designed after observing the confirmatory test results.",
            "The same N-MNIST test set is reused.",
            "The analysis supports or weakens a mechanism interpretation but does not provide independent confirmatory evidence.",
            "Early-prefix accuracy degradation remains a separate limitation even if the mechanism is supported.",
            "Per-seed micro metrics repeat a sample across transitions and do not treat those pairs as independent.",
        ],
    }
    (output / "mechanism_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    make_plots(output, all_rows, timing)
    return summary


def make_plots(output, rows, timing):
    specs = (
        ("matched_regression_by_transition.png", rows["regression"], "matched_regression_difference"),
        ("matched_recovery_by_transition.png", rows["recovery"], "matched_recovery_difference"),
        ("confidence_matched_regression.png", [r for r in rows["confidence"] if r["bin"] == "eligible>=0.6"], "paired_difference"),
        ("probability_drop_by_transition.png", [r for r in rows["probability"] if r["subset"] == "common_correct"], "mean_drop_magnitude_difference"),
    )
    for filename, selected, metric in specs:
        plt.figure(figsize=(7, 4))
        for comparison in (PRIMARY, SECONDARY):
            points = [r for r in selected if r["comparison"] == comparison]
            plt.plot(range(1, len(points) + 1), [r[metric] for r in points], "o", label=comparison)
        plt.axhline(0, color="black", linewidth=1); plt.legend(); plt.tight_layout(); plt.savefig(output / filename); plt.close()
    plt.figure(figsize=(7, 4))
    for method in (SELECTIVE, PRIMARY, SECONDARY):
        selected = [r for r in rows["opportunity"] if r["method"] == method]
        plt.scatter([r["correct_opportunity_rate"] for r in selected], [r["conditional_regression_rate"] for r in selected], label=method)
    plt.legend(); plt.tight_layout(); plt.savefig(output / "opportunity_vs_conditional_regression.png"); plt.close()
    for prefix in ("first", "stable"):
        plt.figure(figsize=(7, 4))
        values = [r[f"{prefix}_correct_delay_mean"] for r in timing if r["comparison"] == PRIMARY]
        plt.step(sorted(values), [(i + 1) / len(values) for i in range(len(values))])
        plt.tight_layout(); plt.savefig(output / f"{prefix}_correct_timestep_cdf.png"); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/temporal_reliability_nmnist_confirmatory")
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.results_root)), indent=2))


if __name__ == "__main__":
    main()

