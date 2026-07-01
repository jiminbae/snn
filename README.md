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

- Monotonic prefix gates: `cumprod(sigmoid(theta))`.
- Hard-prefix timestep skipping: evaluation can run only active prefix timesteps.
- Unscaled hard-prefix inference: `--hard-prefix-unscaled` uses binary spikes on active prefix timesteps.
- Soft-to-hard consistency training: S2H models train soft and hard-prefix passes together.
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
python train.py   --model layerwise_chronoskip_s2h   --dataset fashionmnist   --epochs 1   --batch-size 128   --device cuda   --amp   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1   --limit-train-batches 2   --limit-test-batches 2
```

If CUDA is unavailable, the script falls back to CPU.

## Single Run

```bash
python train.py   --model global_chronoskip_s2h   --dataset fashionmnist   --epochs 20   --batch-size 256   --tmax 8   --lambda-spike 0.05   --eta-time 0.02   --spike-cost-mode gated   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1   --hard-ce-weight 0.5   --consistency-weight 0.1   --reg-warmup-epochs 5   --device cuda   --amp
```

## Main Experiment Suite

```bash
python run_chronoskip_experiments.py   --dataset fashionmnist   --epochs 20   --batch-size 2048   --device cuda   --amp   --hard-prefix-eval   --hard-prefix-unscaled   --min-prefix-steps 1
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
- `layer1_effective_timestep`
- `layer2_effective_timestep`
- `layer1_hard_timestep`
- `layer2_hard_timestep`
- `energy_proxy`
- `prefix_energy_proxy`

Energy proxy definitions:

```text
energy_proxy = gated_spike_rate * effective_timestep
prefix_energy_proxy = prefix_spike_rate * hard_effective_timestep
```

These are proxy metrics for comparing spike activity and active timestep budgets. They are not actual hardware energy measurements.

## Scientific Comparisons

The main comparisons are:

- ChronoSkip versus fixed shorter-T baselines at similar hard timestep budgets.
- Soft-to-hard consistency training versus non-S2H training.
- Global timestep budgets versus layer-wise timestep budgets.
