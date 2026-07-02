#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}

$PYTHON train.py \
  --model global_chronoskip_s2h \
  --dataset fashionmnist \
  --epochs 1 \
  --batch-size 128 \
  --device "$DEVICE" \
  --amp \
  --hard-prefix-eval \
  --hard-prefix-unscaled \
  --limit-train-batches 2 \
  --limit-test-batches 2

$PYTHON - <<'PY'
import torch
from models import build_model

model = build_model("global_chronoskip_s2h", dataset="nmnist", tmax=8, gate_init=2.5)
x = torch.randn(2, 8, 2, 34, 34)
out_soft = model(x, mode="soft")
out_hard = model(x, mode="hard_prefix", hard_prefix_unscaled=True)
assert out_soft["logits"].shape == (2, 10), out_soft["logits"].shape
assert out_hard["logits"].shape == (2, 10), out_hard["logits"].shape
assert "executed_timestep" in out_hard
print("synthetic temporal forward OK")
PY

$PYTHON - <<'PY'
import torch
from models import build_model

model = build_model("layerwise_chronoskip_s2h", dataset="nmnist", tmax=8, gate_init=2.5)
x = torch.randn(2, 8, 2, 34, 34)
out = model(
    x,
    mode="hard_prefix",
    hard_prefix_unscaled=True,
    dependency_constrained_prefix=True,
)
assert out["logits"].shape == (2, 10), out["logits"].shape
assert "layer_hard_timesteps" in out
print("synthetic layer-wise temporal forward OK")
PY
