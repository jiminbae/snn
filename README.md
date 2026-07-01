# SpikeGate SNN Prototype

This is a pure PyTorch + torchvision pilot implementation for testing whether
SpikeGate can reduce spike activity and effective timestep while maintaining
classification accuracy.

## Hypothesis

If SpikeGate is promising, it should maintain accuracy close to the fixed LIF
baseline while reducing spike rate and effective timestep. A strong positive
signal is lower energy proxy with less than 1% absolute accuracy drop.

This pilot evaluates whether SpikeGate reduces spike activity and effective
timestep, which are commonly used proxies for neuromorphic energy efficiency.
Actual hardware power measurement is not included in this prototype.

## Models

- `fixed_lif`: fixed Conv-SNN baseline with one LIF setting.
- `gate_only`: fixed LIF candidate with a learnable timestep gate.
- `neuron_only`: Hard Gumbel-Softmax neuron candidate search without timestep gating.
- `softmax_spikegate`: softmax mixture candidate search plus timestep gating.
- `spikegate`: Hard Gumbel-Softmax candidate search plus timestep gating.

Each spiking layer chooses from:

- `fast_sensitive`: `v_th=0.5`, `tau=2.0`
- `fast_balanced`: `v_th=1.0`, `tau=2.0`
- `memory_balanced`: `v_th=1.0`, `tau=6.0`
- `memory_sparse`: `v_th=1.5`, `tau=6.0`

## Setup

```bash
pip install -r requirements.txt
```

## Single Runs

Fixed LIF baseline:

```bash
python train.py \
  --model fixed_lif \
  --dataset fashionmnist \
  --epochs 20 \
  --batch-size 256 \
  --tmax 8 \
  --device cuda \
  --amp
```

SpikeGate:

```bash
python train.py \
  --model spikegate \
  --dataset fashionmnist \
  --epochs 20 \
  --batch-size 256 \
  --tmax 8 \
  --lambda-spike 0.05 \
  --eta-time 0.02 \
  --gumbel-tau 1.0 \
  --monotonic-gate \
  --hard-prefix-eval \
  --hard-prefix-unscaled \
  --spike-cost-mode mixed \
  --reg-warmup-epochs 5 \
  --device cuda \
  --amp
```

CIFAR-10 is available with `--dataset cifar10`. Regularization is warmed up with `--reg-warmup-epochs` so spike and time penalties do not dominate the earliest epochs. `--hard-prefix-eval` adds an evaluation pass that runs only the first `hard_effective_timestep` steps. Add `--hard-prefix-unscaled` to run those active prefix timesteps with unscaled binary spikes, approximating deployment-style prefix inference.

## Experiment Suite

```bash
python run_experiments.py
```

The suite runs:

- `fixed_lif`
- `gate_only`
- `neuron_only`
- `spikegate`

and writes `results/comparison.csv`.

## Outputs

Each run writes:

```text
results/<run_name>/metrics.csv
results/<run_name>/config.json
results/<run_name>/summary.json
results/<run_name>/plots/
```

Plots include accuracy, raw spike rate, gated spike rate, prefix spike rate,
effective timestep, hard effective timestep, energy proxy, prefix energy proxy,
final timestep gates, and candidate probabilities per layer.

## Notes

Metrics:

- `raw_spike_rate`: average spike activity before timestep gates are applied.
- `gated_spike_rate`: average spike activity after timestep gates are applied. The legacy `spike_rate` key is kept equal to this value.
- `effective_timestep`: sum of soft timestep gates.
- `hard_effective_timestep`: number of active prefix timesteps with gate values above the threshold.
- `prefix_spike_rate`: spike rate from an optional hard-prefix evaluation pass that actually skips timesteps after `T_eff`. With `--hard-prefix-unscaled`, this is the most relevant metric for deployment-style prefix inference.
- `prefix_energy_proxy`: `prefix_spike_rate * hard_effective_timestep`.
- `--spike-cost-mode gated`: use gated spike activity for the spike penalty, matching the original behavior.
- `--spike-cost-mode raw`: use raw spike activity to test whether actual spike generation decreases.
- `--spike-cost-mode mixed`: use an equal raw and gated blend for a balanced pilot setting.

Default spike cost mode is `gated` for backward compatibility. For pilot comparisons, use `gated` to test effective soft-gated activity reduction, `raw` to test actual spike generation reduction, and `mixed` as a balanced setting for follow-up experiments.

The reported `energy proxy` is:

```text
gated_spike_rate * effective_timestep
```

It should not be interpreted as measured hardware energy.
