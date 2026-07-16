#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-results/temporal_reliability_dvs}"
RUN_DIRS=()

for seed in 0 1 2; do
  for spec in "final_ce:0.8" "all_prefix_ce:0.8" "symmetric_kl:0.8" \
              "selective_regression:0.8" "selective_regression:0.6" "selective_regression:0.9"; do
    mode="${spec%%:*}"
    threshold="${spec##*:}"
    if [[ "${mode}" == "selective_regression" ]]; then
      label="${mode}_thr${threshold}"
    else
      label="${mode}"
    fi
    run_dir="${RESULTS_DIR}/${label}/seed_${seed}"
    RUN_DIRS+=("${run_dir}")
    if [[ -f "${run_dir}/temporal_reliability_summary.json" ]]; then
      echo "Skipping completed ${label} seed ${seed}"
      continue
    fi
    python train.py \
      --model fixed_lif --dataset dvs_gesture --epochs 30 --batch-size 32 --tmax 8 \
      --device cuda --amp --seed "${seed}" --event-frame-mode binary --event-downsample-size 64 \
      --val-ratio 0.2 --split-seed 42 --checkpoint-selection best_val --selection-metric val_acc \
      --temporal-training-mode "${mode}" --temporal-loss-weight 1.0 \
      --temporal-confidence-threshold "${threshold}" --temporal-margin 0.0 \
      --temporal-selection-mode hard --prefix-diagnostics \
      --results-dir "${RESULTS_DIR}/${label}" --run-name "seed_${seed}"
  done
done

python aggregate_temporal_reliability_results.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "${RESULTS_DIR}/aggregate"
