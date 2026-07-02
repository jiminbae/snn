# ChronoSkip SNN Prototype

ChronoSkip learns deployable prefix timestep budgets for efficient spiking neural network inference.

Full title: **ChronoSkip: Soft-to-Hard Learning of Layer-wise Timestep Skipping for Efficient Spiking Neural Networks**.

## Motivation

Fixed-timestep SNNs execute all timesteps even when later timesteps may be redundant. ChronoSkip learns monotonic prefix timestep budgets during training and converts soft training gates into hard-prefix inference that actually skips later timesteps.

This repository is focused on timestep skipping and prefix timestep budget learning. It does not include neuron candidate search, NAS, PLIF, or architecture search.

## Models

- `fixed_lif`: fixed-timestep Conv-SNN baseline. Change `--tmax` to run fixed T baselines.
- `soft_gate`: learnable monotonic soft timestep gates while still running all timesteps.
- `global_chronoskip`: one global monotonic gate shared by both spiking layers.
- `global_chronoskip_s2h`: global ChronoSkip with soft-to-hard consistency training.
- `layerwise_chronoskip`: separate monotonic gates for layer 1 and layer 2.
- `layerwise_chronoskip_s2h`: layer-wise ChronoSkip with soft-to-hard consistency training.

## Key Ideas

- Monotonic prefix gates: `cumprod(sigmoid(theta))`, initialized with `--gate-init`.
- Hard-prefix timestep skipping: evaluation can run only active prefix timesteps.
- Layer-wise hard-prefix semantics: if an upstream layer is skipped at a timestep, downstream layers receive zero new input but may continue membrane updates if active. `--dependency-constrained-prefix` enforces downstream activity only when upstream layers are also active.
- Unscaled hard-prefix inference: `--hard-prefix-unscaled` uses binary spikes on active prefix timesteps.
- Soft-to-hard consistency training: S2H models train soft and hard-prefix passes together. The hard-prefix pass uses non-differentiable binary prefix decisions, so hard CE and consistency train weights for robustness; soft gates and time/hard-budget regularizers remain the differentiable path for learning budgets.
- Layer-wise timestep budgets: layer-wise models learn separate budgets for each spiking layer.
- Energy proxy only: reported values are not measured hardware power.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

```bash
python train.py   --model layerwise_chronoskip_s2h   --dataset fashionmnist   --epochs 1   --batch-size 128   --device cuda   --amp   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1   --gate-init 5.0   --limit-train-batches 2   --limit-test-batches 2
```

If CUDA is unavailable, the script falls back to CPU.

## Single Run

```bash
python train.py   --model global_chronoskip_s2h   --dataset fashionmnist   --epochs 20   --batch-size 256   --tmax 8   --gate-init 5.0   --lambda-spike 0.05   --eta-time 0.02   --spike-cost-mode gated   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1   --hard-ce-weight 0.5   --consistency-weight 0.1   --reg-warmup-epochs 5   --device cuda   --amp
```

## Event / Temporal Datasets

Static datasets use tensors shaped `[B, C, H, W]`; the same image is reused across SNN timesteps, preserving the original Fashion-MNIST/CIFAR-10 behavior. Event datasets use binned event frames shaped `[B, T, C, H, W]`; the model consumes `x[:, t]` at timestep `t`.

ChronoSkip hard-prefix inference on event datasets actually skips later event frames, which is different from repeated-timestep evaluation on static images. This enables evaluation on datasets with real temporal structure.

Supported event datasets:

- `nmnist`
- `dvs_gesture`

Event datasets require `tonic`; install it through `requirements.txt`.

```bash
python train.py \
  --model global_chronoskip_s2h \
  --dataset nmnist \
  --epochs 10 \
  --batch-size 256 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode binary \
  --hard-prefix-eval \
  --hard-prefix-unscaled \
  --gate-init 2.5 \
  --eta-time 0.05 \
  --lambda-hard-budget 0.05 \
  --hard-budget-sharpness 5.0 \
  --target-timestep 6 \
  --target-budget-weight 0.05
```

For event datasets, there are two fixed timestep baselines:

1. `fixed_rebin_Tk`: the entire event stream is rebinned into `k` frames. This evaluates coarse temporal rebinning.
2. `fixed_prefix_Tk`: the event stream is first binned into `base_tmax` frames, then only the first `k` frames are executed. This is the fixed-prefix baseline for ChronoSkip hard-prefix inference.

ChronoSkip should primarily be compared against `fixed_prefix_Tk` at matched executed timesteps. `fixed_rebin_Tk` is still useful as an additional baseline, but it answers a different question because it compresses the full event duration into fewer bins.

```bash
python run_event_chronoskip_tradeoff.py \
  --dataset nmnist \
  --epochs 10 \
  --batch-size 256 \
  --base-tmax 8 \
  --device cuda \
  --amp
```

The tradeoff script writes `results/event_chronoskip_tradeoff/comparison.csv`. Do not claim ChronoSkip beats fixed shorter-T baselines unless it beats or matches `fixed_prefix_Tk` at similar executed timestep budgets.

## Threshold-Aware Hard Budget Loss

The original soft time loss penalizes the sum of gate values, but it may not push gates below the hard-prefix threshold. ChronoSkip therefore supports a threshold-aware hard budget proxy:

```text
hard_budget_proxy = sum(sigmoid(k * (gate - threshold)))
```

This differentiable proxy penalizes timesteps whose gates remain above the hard-prefix threshold and helps convert soft gate reduction into actual hard-prefix timestep skipping. For layer-wise models, the proxy is averaged across layer 1 and layer 2 for the loss and target comparison.

Options:

- `--lambda-hard-budget`: weight for the threshold-aware hard budget proxy.
- `--hard-budget-sharpness`: sigmoid sharpness `k`.
- `--target-timestep`: optional target budget; `0.0` disables target pressure.
- `--target-budget-weight`: optional weight for penalizing proxy budget above the target.

Suggested diagnostic command:

```bash
python train.py \
  --model global_chronoskip_s2h \
  --dataset fashionmnist \
  --epochs 5 \
  --batch-size 1024 \
  --device cuda \
  --amp \
  --hard-prefix-eval \
  --hard-prefix-unscaled \
  --gate-init 4.0 \
  --eta-time 0.05 \
  --lambda-hard-budget 0.05 \
  --hard-budget-sharpness 20.0 \
  --target-timestep 6 \
  --target-budget-weight 0.05
```

This is still a proxy and not measured hardware energy. The hard-prefix mask remains non-differentiable; the threshold-aware budget loss is a differentiable surrogate that pressures gates toward crossing the threshold.

## Main Experiment Suite

```bash
python run_chronoskip_experiments.py   --dataset fashionmnist   --epochs 20   --batch-size 2048   --device cuda   --amp   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1   --gate-init 5.0   --lambda-hard-budget 0.05   --target-timestep 6   --target-budget-weight 0.05
```

The suite runs:

- `fixed_lif_T8`
- `fixed_lif_T6`
- `fixed_lif_T4`
- `fixed_lif_T2`
- `soft_gate_T8`
- `global_chronoskip_T8`
- `global_chronoskip_s2h_T8`
- `layerwise_chronoskip_T8`
- `layerwise_chronoskip_s2h_T8`

The comparison CSV is saved to `results/comparison.csv`.

## Hard-Budget Diagnostics

```bash
python run_chronoskip_diagnostics.py   --dataset fashionmnist   --epochs 5   --batch-size 1024   --device cuda   --amp
```

The diagnostic sweep focuses on hard timestep reduction across global and layer-wise S2H models with different `--lambda-hard-budget` values. It saves its comparison table to `results/diagnostics_hard_budget/comparison.csv`.

## Metrics

Each run writes:

```text
results/<run_name>/metrics.csv
results/<run_name>/config.json
results/<run_name>/summary.json
results/<run_name>/plots/
```

Logged metrics include:

- `test_acc`
- `soft_acc`
- `hard_acc`
- `raw_spike_rate`
- `gated_spike_rate`
- `prefix_spike_rate`
- `effective_timestep`
- `hard_effective_timestep`
- `executed_timestep`
- `layer1_effective_timestep`
- `layer2_effective_timestep`
- `layer1_hard_timestep`
- `layer2_hard_timestep`
- `energy_proxy`
- `prefix_energy_proxy`
- `loop_energy_proxy`
- `train_hard_budget_cost`
- `train_target_budget_loss`
- `train_hard_budget_proxy`

Energy proxy definitions:

```text
energy_proxy = gated_spike_rate * effective_timestep
prefix_energy_proxy = prefix_spike_rate * hard_effective_timestep
loop_energy_proxy = prefix_spike_rate * executed_timestep
```

`prefix_spike_rate` is actual executed-prefix spike activity only when hard-prefix evaluation is enabled. In soft-only runs, it is a compatibility field equal to the soft gated spike activity. `prefix_energy_proxy` uses the average hard layer budget, while `loop_energy_proxy` uses the recurrent loop count actually executed. `summary.json` also stores `timestep_gates`, `hard_prefix_masks`, `hard_prefix_steps`, and `hard_budget_proxy` so threshold-crossing behavior can be diagnosed directly. These are proxy metrics for comparing spike activity and active timestep budgets. They are not actual hardware energy measurements.

## Scientific Comparisons

The main comparisons are:

- ChronoSkip versus fixed shorter-T baselines at similar hard timestep budgets.
- Soft-to-hard consistency training versus non-S2H training.
- Global timestep budgets versus layer-wise timestep budgets.
