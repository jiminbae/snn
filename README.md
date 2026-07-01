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
  --device cuda \
  --amp
```

CIFAR-10 is available with `--dataset cifar10`.

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

Plots include accuracy, spike rate, effective timestep, energy proxy, final
timestep gates, and candidate probabilities per layer.

## Notes

The reported `energy proxy` is:

```text
average_spike_rate * effective_timestep
```

It should not be interpreted as measured hardware energy.
