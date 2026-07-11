# When More Events Hurt: Selective Monotonic Training for Event-Based SNNs

## Current Research Direction

This project studies a counterintuitive behavior in event-based spiking neural networks: processing more event frames can sometimes reduce classification accuracy. The current phase focuses on diagnosing this behavior before introducing a new training objective.

The repository provides prefix-wise evaluation for one shared SNN, independently trained prefix specialists, and the completed ChronoSkip implementation as a baseline and historical experiment. This phase does not implement selective monotonic loss, adaptive early exit, temporal attention, or a new architecture.

## Motivation

A shared model can follow a trajectory such as:

```text
Prefix T1: correct
Prefix T2: correct
Prefix T3: incorrect
Prefix T4: incorrect
```

Additional temporal evidence should not automatically be assumed to improve an SNN. Late events may introduce noise, overwrite useful membrane states, or expose optimization weaknesses. These are hypotheses to investigate, not established causes.

Static inputs have shape `[B,C,H,W]` and are reused at every recurrent timestep. Event inputs have shape `[B,T,C,H,W]`, and frame `x[:,t]` is consumed at timestep `t`.

## Negative Temporal Gain

- **Negative temporal gain**: accuracy decreases from one temporal prefix to the next. This project reports the sum and mean of these decreases.
- **Prediction regression**: a sample is classified correctly at one prefix and incorrectly after receiving more events.
- **First-correct timestep**: the first prefix at which a sample is correct.
- **Stable-correct timestep**: the first prefix after which a sample remains correct at every later prefix.

`negative temporal gain` is a project diagnostic term, not an established standard metric.

## Prefix Diagnostics

Use `--prefix-diagnostics` to evaluate one trained model at every executed prefix:

```bash
python train.py \
  --model fixed_lif \
  --dataset nmnist \
  --epochs 1 \
  --batch-size 16 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode binary \
  --prefix-diagnostics \
  --limit-train-batches 2 \
  --limit-test-batches 2 \
  --results-dir results/prefix_diag_smoke \
  --run-name nmnist_shared_T8_smoke
```

The run saves:

```text
summary.json
prefix_metrics.json
prefix_accuracy_curve.csv
prefix_regression_curve.csv
plots/prefix_accuracy_curve.png
plots/prefix_regression_curve.png
```

Prefix logits are normalized by the number of observed timesteps at each prefix. For static datasets, the same mechanism measures recurrent integration rather than newly arriving temporal evidence.

## Shared Model vs Prefix Specialists

A **shared anytime model** is one model expected to operate at several prefix lengths. A **prefix specialist** is trained using one fixed prefix length. Their matched-budget difference is reported as:

```text
prefix regret at T=k = specialist accuracy at T=k - shared accuracy at T=k
```

`prefix regret` is introduced here as a diagnostic quantity and is not claimed to be an established standard metric.

The single-seed runner trains `shared_fixed_lif_T8` and specialists at T1, T2, T4, T6, and T8:

```bash
python run_prefix_diagnostics.py \
  --dataset nmnist \
  --epochs 10 \
  --batch-size 256 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode binary \
  --seed 0 \
  --results-dir results/prefix_diagnostics_nmnist_seed0
```

The multi-seed runner preserves every seed directory and writes aggregate CSV files:

```bash
python run_prefix_diagnostics_multiseed.py \
  --dataset nmnist \
  --epochs 10 \
  --batch-size 256 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode binary \
  --seeds 0 1 2 \
  --results-dir results/prefix_diagnostics_nmnist_e10
```

## Datasets

Supported datasets are Fashion-MNIST, CIFAR-10, N-MNIST, and DVS Gesture. N-MNIST and DVS Gesture require `tonic` and support binary or count event frames. DVS Gesture defaults to spatial downsampling at 64 when no size is supplied.

N-MNIST is the first sanity check because earlier experiments indicated strong early-prefix performance. DVS Gesture is then evaluated with both binary and count frames to test whether the observed behavior persists across frame representations.

## Running Diagnostic Experiments

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DVS Gesture binary:

```bash
python run_prefix_diagnostics_multiseed.py \
  --dataset dvs_gesture \
  --epochs 30 \
  --batch-size 32 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode binary \
  --event-downsample-size 64 \
  --seeds 0 1 2 \
  --results-dir results/prefix_diagnostics_dvs_binary_e30
```

DVS Gesture count:

```bash
python run_prefix_diagnostics_multiseed.py \
  --dataset dvs_gesture \
  --epochs 30 \
  --batch-size 32 \
  --tmax 8 \
  --device cuda \
  --amp \
  --event-frame-mode count \
  --event-downsample-size 64 \
  --seeds 0 1 2 \
  --results-dir results/prefix_diagnostics_dvs_count_e30
```

## ChronoSkip Baseline

Full title: **ChronoSkip: Soft-to-Hard Learning of Layer-wise Timestep Skipping for Efficient Spiking Neural Networks**.

ChronoSkip demonstrated deployable hard-prefix timestep skipping, but in the current Fashion-MNIST, N-MNIST, and DVS Gesture experiments it did not outperform independently trained fixed-prefix specialists at matched executed timesteps. This is not a claim that ChronoSkip fails universally. It remains in the repository as a completed baseline, ablation, historical experiment, and motivation for studying prefix-wise robustness.

Available models:

- `fixed_lif`: fixed-timestep Conv-SNN baseline.
- `soft_gate`: monotonic soft timestep gates.
- `global_chronoskip` and `global_chronoskip_s2h`: one global gate.
- `layerwise_chronoskip` and `layerwise_chronoskip_s2h`: separate layer gates.

Single ChronoSkip run:

```bash
python train.py \
  --model global_chronoskip_s2h \
  --dataset fashionmnist \
  --epochs 20 \
  --batch-size 256 \
  --tmax 8 \
  --gate-init 5.0 \
  --lambda-spike 0.05 \
  --eta-time 0.02 \
  --hard-prefix-eval \
  --hard-prefix-unscaled \
  --reg-warmup-epochs 5 \
  --device cuda \
  --amp
```

Event tradeoff suite:

```bash
python run_event_chronoskip_tradeoff.py \
  --dataset nmnist \
  --epochs 10 \
  --batch-size 256 \
  --base-tmax 8 \
  --device cuda \
  --amp
```

The event suite retains two fixed baselines. `fixed_rebin_Tk` compresses the full event interval into `k` bins, while `fixed_prefix_Tk` bins to the common base T and executes only the first `k` frames. ChronoSkip should be compared primarily with `fixed_prefix_Tk` at matched executed timesteps.

Existing suites remain available:

```bash
python run_chronoskip_experiments.py --dataset fashionmnist --device cuda --amp
python run_chronoskip_diagnostics.py --dataset fashionmnist --device cuda --amp
./run_dvs_gesture_chronoskip_tradeoff.sh
./run_dvs_gesture_count_tradeoff.sh
```

Threshold-aware controls such as `--lambda-hard-budget`, `--target-timestep`, `--target-budget-mode`, and `--min-target-timestep` remain supported for baseline reproduction.

## Metrics and Scientific Limitations

Prefix diagnostics include the full accuracy curve, consecutive and ever-regressed rates, negative temporal gain, worst-prefix accuracy, discrete mean prefix accuracy (`prefix_accuracy_auc`), and first/stable-correct statistics. `prefix_accuracy_auc` is a convenient name for the discrete mean, not a continuous integral.

Existing ChronoSkip metrics remain available: raw, gated, and prefix spike rates; effective, hard-effective, and executed timesteps; layer budgets; and energy proxies. Energy values are analytical proxies, not measured hardware power.

These experiments diagnose whether accuracy decreases across temporal prefixes and compare a shared model against independently trained specialists. They do not yet show that selective monotonic training solves regression, establish state-of-the-art performance, prove that more events generally hurt SNNs, or identify sensor noise as the cause.
