#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
EXP="${EXP:-results/dvs_gesture_phase_a_e30_val20_binary_bs32}"
DATA_DIR="${DATA_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

BACKBONE_EPOCHS="${BACKBONE_EPOCHS:-30}"
BACKBONE_BATCH_SIZE="${BACKBONE_BATCH_SIZE:-32}"

PREDICTOR_EPOCHS="${PREDICTOR_EPOCHS:-100}"
PREDICTOR_BATCH_SIZE="${PREDICTOR_BATCH_SIZE:-512}"
PREDICTOR_PATIENCE="${PREDICTOR_PATIENCE:-10}"

SPLIT_SEED="${SPLIT_SEED:-42}"
SEEDS=(0 1 2)

mkdir -p "$EXP/logs"

echo "========================================"
echo "DVS Gesture Phase A kill test"
echo "Experiment directory: $EXP"
echo "Device: $DEVICE"
echo "Backbone seeds: ${SEEDS[*]}"
echo "Split seed: $SPLIT_SEED"
echo "========================================"

# =========================
# Optional unit tests
# =========================
echo
echo "[0/4] Running unit tests..."

python -m unittest tests.test_stopping_policy_evaluation
python -m unittest tests.test_kill_test_selection
python -m unittest discover -s tests

# =========================
# Per-seed runs
# =========================
for SEED in "${SEEDS[@]}"; do
  RUN_NAME="seed_${SEED}_shared_fixed_lif_T8"
  RUN_DIR="$EXP/$RUN_NAME"
  LOG_FILE="$EXP/logs/seed_${SEED}.log"

  echo
  echo "========================================"
  echo "Processing backbone seed $SEED"
  echo "Run directory: $RUN_DIR"
  echo "========================================"

  # -------------------------
  # 1. Backbone training
  # -------------------------
  if [[ -f "$RUN_DIR/best_checkpoint.pt" \
     && -f "$RUN_DIR/split_indices.pt" \
     && -f "$RUN_DIR/selection_summary.json" \
     && -f "$RUN_DIR/summary.json" ]]; then
    echo "[1/4] Seed $SEED backbone already completed. Skipping."
  else
    echo "[1/4] Training seed $SEED backbone..."

    python train.py \
      --model fixed_lif \
      --dataset dvs_gesture \
      --epochs "$BACKBONE_EPOCHS" \
      --batch-size "$BACKBONE_BATCH_SIZE" \
      --tmax 8 \
      --device "$DEVICE" \
      --amp \
      --data-dir "$DATA_DIR" \
      --event-frame-mode binary \
      --event-downsample-size 64 \
      --val-ratio 0.2 \
      --split-seed "$SPLIT_SEED" \
      --checkpoint-selection best_val \
      --selection-metric val_acc \
      --seed "$SEED" \
      --prefix-diagnostics \
      --save-prefix-trajectories \
      --num-workers "$NUM_WORKERS" \
      --results-dir "$EXP" \
      --run-name "$RUN_NAME" \
      2>&1 | tee "$LOG_FILE"
  fi

  # -------------------------
  # 2. Trajectory export
  # -------------------------
  TRAJECTORY_DIR="$RUN_DIR/trajectories"

  if [[ -f "$TRAJECTORY_DIR/train_trajectories.pt" \
     && -f "$TRAJECTORY_DIR/val_trajectories.pt" \
     && -f "$TRAJECTORY_DIR/test_trajectories.pt" \
     && -f "$TRAJECTORY_DIR/trajectory_export_summary.json" ]]; then
    echo "[2/4] Seed $SEED trajectories already exported. Skipping."
  else
    echo "[2/4] Exporting seed $SEED trajectories..."

    python export_split_trajectories.py \
      --run-dir "$RUN_DIR" \
      --device "$DEVICE" \
      --batch-size "$BACKBONE_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      2>&1 | tee -a "$LOG_FILE"
  fi

  # -------------------------
  # 3. Stopping predictors
  # -------------------------
  KILL_DIR="$RUN_DIR/logit_kill_test"

  if [[ -f "$KILL_DIR/kill_test_summary.json" \
     && -f "$KILL_DIR/validation_selected_test_results.csv" \
     && -f "$KILL_DIR/tolerance_matched_comparisons.csv" ]]; then
    echo "[3/4] Seed $SEED kill test already completed. Skipping."
  else
    echo "[3/4] Training seed $SEED stopping predictors..."

    python train_stopping_predictors.py \
      --trajectory-dir "$TRAJECTORY_DIR" \
      --output-dir "$KILL_DIR" \
      --predictors \
        recoverability_final \
        final_horizon_gain \
        one_step \
        multi_horizon \
      --feature-modes \
        current_logits \
        logit_history \
      --lambdas 0 0.5 1 1.5 2 3 4 6 8 \
      --hidden-dim 64 \
      --dropout 0.1 \
      --epochs "$PREDICTOR_EPOCHS" \
      --patience "$PREDICTOR_PATIENCE" \
      --batch-size "$PREDICTOR_BATCH_SIZE" \
      --lr 1e-3 \
      --weight-decay 1e-4 \
      --policy-seed 0 \
      --selective-weighting none \
      --device "$DEVICE" \
      2>&1 | tee -a "$LOG_FILE"
  fi

  echo
  echo "Seed $SEED summary:"
  cat "$KILL_DIR/kill_test_summary.json"
  echo
done

# =========================
# 4. Aggregate three seeds
# =========================
AGG_DIR="$EXP/logit_kill_test_aggregate"

echo
echo "========================================"
echo "[4/4] Aggregating seeds 0, 1, and 2"
echo "========================================"

python aggregate_stopping_predictor_results.py \
  --run-dirs \
    "$EXP/seed_0_shared_fixed_lif_T8" \
    "$EXP/seed_1_shared_fixed_lif_T8" \
    "$EXP/seed_2_shared_fixed_lif_T8" \
  --output-dir "$AGG_DIR" \
  2>&1 | tee "$EXP/logs/aggregate.log"

echo
echo "========================================"
echo "Phase A complete"
echo "========================================"
echo
echo "Final aggregate summary:"
cat "$AGG_DIR/aggregate_summary.json"

echo
echo "Primary result files:"
echo "  $AGG_DIR/aggregate_summary.json"
echo "  $AGG_DIR/aggregate_selected_operating_points.csv"
echo "  $AGG_DIR/aggregate_predictor_metrics.csv"
echo "  $AGG_DIR/aggregate_tolerance_comparisons.csv"
echo "  $AGG_DIR/aggregate_accuracy_vs_timestep.png"
