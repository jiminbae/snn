#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
BS=${BS:-64}
EPOCHS=${EPOCHS:-10}

# Smoke option:
# $PYTHON run_event_chronoskip_tradeoff.py \
#   --dataset dvs_gesture \
#   --epochs 1 \
#   --batch-size 16 \
#   --base-tmax 8 \
#   --device "$DEVICE" \
#   --amp \
#   --event-frame-mode count \
#   --event-downsample-size 64 \
#   --limit-train-batches 1 \
#   --limit-test-batches 1 \
#   --results-dir results/event_smoke_tradeoff_dvs_gesture_count

$PYTHON run_event_chronoskip_tradeoff.py \
  --dataset dvs_gesture \
  --epochs "$EPOCHS" \
  --batch-size "$BS" \
  --base-tmax 8 \
  --device "$DEVICE" \
  --amp \
  --event-frame-mode count \
  --event-downsample-size 64 \
  --lambda-spike 0.05 \
  --eta-time 0.05 \
  --hard-budget-sharpness 5.0 \
  --target-budget-weight 0.05 \
  --target-budget-mode two_sided \
  --min-target-timestep 3 \
  --min-target-weight 0.05 \
  --results-dir results/event_chronoskip_tradeoff_dvs_gesture_count_e10
