#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-results/temporal_reliability_nmnist_confirmatory}"
MODE="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIRM_DEVICE="${CONFIRM_DEVICE:-cuda}"

if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="python"; fi
if [[ -n "$MODE" && "$MODE" != "--smoke" ]]; then
  echo "Usage: $0 [results-dir] [--smoke]" >&2
  exit 2
fi
mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$RESULTS_DIR/commands.log") 2>&1
set -x

COMMON_ARGS=(
  --dataset nmnist --model fixed_lif --batch-size 32 --tmax 8
  --event-frame-mode binary --val-ratio 0.2 --split-seed 123
  --checkpoint-selection best_val --selection-metric val_acc
  --prefix-diagnostics --prefix-loss-weight 0.0 --temporal-loss-weight 1.0
  --temporal-margin 0.0 --temporal-temperature 1.0
  --temporal-selection-mode hard --device "$CONFIRM_DEVICE" --data-dir data
)

if [[ "$MODE" == "--smoke" ]]; then
  SMOKE_RUN="$RESULTS_DIR/final_ce/seed_3_smoke"
  if [[ ! -f "$SMOKE_RUN/temporal_reliability_summary.json" ]]; then
    "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" --epochs 1 --seed 3 \
      --temporal-training-mode final_ce --limit-train-batches 2 \
      --limit-val-batches 2 --limit-test-batches 2 --num-workers 0 \
      --results-dir "$RESULTS_DIR/final_ce" --run-name seed_3_smoke
  fi
  test -f "$SMOKE_RUN/best_checkpoint.pt"
  test -f "$SMOKE_RUN/prefix_metrics.json"
  test -f "$SMOKE_RUN/temporal_reliability_summary.json"
  exit 0
fi

RUN_DIRS=()
for SEED in 3 4 5; do
  for METHOD in final_ce symmetric_kl selective_regression; do
    LABEL="$METHOD"
    METHOD_ARGS=(--temporal-training-mode "$METHOD")
    if [[ "$METHOD" == "selective_regression" ]]; then
      LABEL="selective_regression_thr0.6"
      METHOD_ARGS+=(--temporal-confidence-threshold 0.6)
    fi
    RUN_DIR="$RESULTS_DIR/$LABEL/seed_$SEED"
    RUN_DIRS+=("$RUN_DIR")
    if [[ -f "$RUN_DIR/temporal_reliability_summary.json" ]]; then
      echo "Skipping completed $LABEL seed $SEED"
      continue
    fi
    "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}" \
      --epochs 30 --seed "$SEED" --results-dir "$RESULTS_DIR/$LABEL" \
      --run-name "seed_$SEED"
  done
done
"$PYTHON_BIN" aggregate_temporal_reliability_confirmatory.py \
  --run-dirs "${RUN_DIRS[@]}" --output-dir "$RESULTS_DIR/aggregate"

