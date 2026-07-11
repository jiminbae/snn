#!/usr/bin/env bash
set -euo pipefail

DATASET=dvs_gesture
EPOCHS=10
BS=32
BASE_TMAX=8
DEVICE=cuda
SEED=0

RESULTS_DIR=results/event_chronoskip_tradeoff_dvs_gesture_e10_bs32

python run_event_chronoskip_tradeoff.py \
  --dataset $DATASET \
  --epochs $EPOCHS \
  --batch-size $BS \
  --base-tmax $BASE_TMAX \
  --device $DEVICE \
  --amp \
  --seed $SEED \
  --event-frame-mode binary \
  --event-downsample-size 64 \
  --lambda-spike 0.05 \
  --eta-time 0.05 \
  --hard-budget-sharpness 5.0 \
  --target-budget-weight 0.05 \
  --target-budget-mode two_sided \
  --min-target-timestep 3 \
  --min-target-weight 0.05 \
  --results-dir $RESULTS_DIR

echo ""
echo "Done."
echo "Comparison CSV:"
echo "$RESULTS_DIR/comparison.csv"
