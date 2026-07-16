"""Validation-only operating-point selection and tolerance-matched comparisons."""

from __future__ import annotations

from typing import Any, Iterable

TOLERANCES = (0.0, 0.5, 1.0, 2.0)


def unique_in_order(values: Iterable[str], label: str) -> list[str]:
    result = []
    for value in values:
        if value in result:
            raise ValueError(f"Duplicate {label}: {value}")
        result.append(value)
    if not result:
        raise ValueError(f"At least one {label} is required.")
    return result


def planned_configurations(predictors: Iterable[str], feature_modes: Iterable[str]) -> list[tuple[str, str, str]]:
    predictor_list = unique_in_order(predictors, "predictor")
    mode_list = unique_in_order(feature_modes, "feature mode")
    return [(predictor, mode, f"{predictor}__{mode}") for predictor in predictor_list for mode in mode_list]


def select_validation_operating_points(
    validation_rows: list[dict[str, Any]], final_accuracy: float, value_key: str,
    tolerances: Iterable[float] = TOLERANCES,
) -> list[dict[str, Any]]:
    selected = []
    for tolerance in tolerances:
        eligible = [row for row in validation_rows if float(row["Accuracy"]) >= final_accuracy - float(tolerance)]
        if eligible:
            best = min(eligible, key=lambda row: (float(row["Average Timestep"]), -float(row["Accuracy"]),
                                                  -float(row[value_key])))
            selected.append({**best, "Accuracy Tolerance PP": float(tolerance), "Meets Validation Tolerance": True})
    return selected


def apply_selected_parameters(
    selected_validation: list[dict[str, Any]], test_rows: list[dict[str, Any]], value_key: str,
) -> list[dict[str, Any]]:
    output = []
    for selected in selected_validation:
        match = next(row for row in test_rows if float(row[value_key]) == float(selected[value_key]))
        output.append({**match, "Accuracy Tolerance PP": selected["Accuracy Tolerance PP"],
                       "Validation Accuracy": selected["Accuracy"],
                       "Validation Average Timestep": selected["Average Timestep"],
                       "Meets Validation Tolerance": selected["Meets Validation Tolerance"],
                       "Result Type": "validation_selected"})
    return output


def tolerance_matched_comparisons(
    rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]] | None = None,
    final_test_accuracy: float | None = None,
) -> list[dict[str, Any]]:
    oracle_rows = oracle_rows or []
    by_key = {(float(row["Accuracy Tolerance PP"]), str(row["Method"])): row for row in rows}
    output = []
    for row in rows:
        tolerance = float(row["Accuracy Tolerance PP"]); method = str(row["Method"])
        predictor = str(row.get("Predictor", "")); mode = str(row.get("Feature Mode", ""))
        result = {"Accuracy Tolerance PP": tolerance, "Method": method, "Predictor": predictor,
                  "Feature Mode": mode, "Test Accuracy": row["Accuracy"],
                  "Test Average Timestep": row["Average Timestep"],
                  "Validation Accuracy": row.get("Validation Accuracy", ""),
                  "Validation Average Timestep": row.get("Validation Average Timestep", ""),
                  "Selected Threshold": row.get("Threshold", ""), "Selected Lambda": row.get("Lambda", ""),
                  "Meets Validation Tolerance": row.get("Meets Validation Tolerance", False)}
        result["Meets Test Tolerance"] = (float(row["Accuracy"]) >= final_test_accuracy - tolerance
                                           if final_test_accuracy is not None else "")
        references = {
            "Timestep Gain vs Confidence": "Confidence",
            "Timestep Gain vs Confidence Stability": "Confidence Stability",
            "Timestep Gain vs Final-Horizon Same Feature": f"final_horizon_gain__{mode}" if mode else "",
            "Timestep Gain vs Same-Predictor Current Logits": f"{predictor}__current_logits" if predictor else "",
        }
        for key, reference_method in references.items():
            reference = by_key.get((tolerance, reference_method))
            result[key] = (float(reference["Average Timestep"]) - float(row["Average Timestep"])) if reference else ""
        eligible = [oracle for oracle in oracle_rows if float(oracle["Accuracy"]) >= float(row["Accuracy"])]
        if eligible:
            oracle = min(eligible, key=lambda item: float(item["Average Timestep"]))
            result["Timestep Gap to Oracle"] = float(row["Average Timestep"]) - float(oracle["Average Timestep"])
            confidence = by_key.get((tolerance, "Confidence"))
            denominator = float(confidence["Average Timestep"]) - float(oracle["Average Timestep"]) if confidence else 0
            result["Oracle Gap Closed"] = ((float(confidence["Average Timestep"]) - float(row["Average Timestep"])) / denominator
                                            if denominator > 0 else "")
        else:
            result["Timestep Gap to Oracle"] = result["Oracle Gap Closed"] = ""
        output.append(result)
    return output


def provisional_recommendation(comparisons: list[dict[str, Any]]) -> tuple[str, dict[str, Any], list[str]]:
    tolerance_results: dict[str, Any] = {}
    successful = 0
    for tolerance in TOLERANCES:
        rows = [row for row in comparisons if float(row["Accuracy Tolerance PP"]) == tolerance
                and str(row["Predictor"]) == "multi_horizon"]
        success_rows = [row for row in rows
                        if isinstance(row["Timestep Gain vs Final-Horizon Same Feature"], (int, float))
                        and row["Timestep Gain vs Final-Horizon Same Feature"] > 0
                        and ((isinstance(row["Timestep Gain vs Confidence"], (int, float))
                              and row["Timestep Gain vs Confidence"] > 0)
                             or (isinstance(row["Timestep Gain vs Confidence Stability"], (int, float))
                                 and row["Timestep Gain vs Confidence Stability"] > 0))
                        and bool(row["Meets Validation Tolerance"])
                        and bool(row.get("Meets Test Tolerance", True))]
        beats_final = any(isinstance(row["Timestep Gain vs Final-Horizon Same Feature"], (int, float))
                          and row["Timestep Gain vs Final-Horizon Same Feature"] > 0 for row in rows)
        beats_baseline = any((isinstance(row["Timestep Gain vs Confidence"], (int, float)) and row["Timestep Gain vs Confidence"] > 0)
                             or (isinstance(row["Timestep Gain vs Confidence Stability"], (int, float))
                                 and row["Timestep Gain vs Confidence Stability"] > 0) for row in rows)
        history = next((row for row in rows if row["Feature Mode"] == "logit_history"), None)
        history_beats = bool(history and isinstance(history["Timestep Gain vs Same-Predictor Current Logits"], (int, float))
                             and history["Timestep Gain vs Same-Predictor Current Logits"] > 0)
        success = bool(success_rows)
        successful += int(success)
        tolerance_results[str(tolerance)] = {"multi_horizon_beats_confidence_or_stability": beats_baseline,
                                             "multi_horizon_beats_final_horizon": beats_final,
                                             "logit_history_beats_current_logits": history_beats,
                                             "provisional_success": success}
    recommendation = "provisional_go" if successful >= 2 else "provisional_weak_go" if successful == 1 else "provisional_no_go"
    reasons = [f"Multi-horizon met target and baseline criteria at {successful} of {len(TOLERANCES)} tolerances.",
               "Final go/no-go requires the backbone-seed aggregate."]
    return recommendation, tolerance_results, reasons
