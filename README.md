# SALaD

Code for [**Explicitly Modeling Censoring Produces Superior Survival Predictors**](https://openreview.net/forum?id=0xJt4PqPJv) (ICML 2026).

SALaD (Survival Analysis via Latent Decomposed Representation) is a
survival-layer-agnostic framework that explicitly models event-time and
censoring-time distributions with event-specific, censoring-specific, and shared
representations.

## Installation

Create an environment with Python 3.10 or newer, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Minimal Usage

Train and evaluate SALaD on SUPPORT:

```bash
python run.py --data SUPPORT --model N-MTLR-salad --n-exp 1 --n-epochs 50
```

Run a two-branch version:

```bash
python run.py --data SUPPORT --model N-MTLR-2B --n-exp 1 --n-epochs 50
```

Run the baseline:

```bash
python run.py --data SUPPORT --model N-MTLR --n-exp 1 --n-epochs 50
```

Useful SALaD options include:

```bash
python run.py \
  --data SUPPORT \
  --model AFTNN-Weibull-salad \
  --neurons 64,64 \
  --e-dims 16 \
  --c-dims 16 \
  --ipm mmd-rbf \
  --alpha 10 \
  --beta 0.01
```
