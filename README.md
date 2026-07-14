# hybrid-bioprocess-lab

A personal lab for practising production-grade ML engineering on hybrid bioprocess models.

The domain is a fed-batch mammalian cell culture. The scientific question is:
how do you add a data-driven layer to a mechanistic bioprocess model without
letting it quietly violate mass balances, produce lactate from nowhere, or
learn assay noise as if it were biology?

The engineering question is: how do you wrap that model in the tooling a real
ML team uses — Flyte, MLflow, Optuna, Ray Tune, packaging, CI, and a clean
inference path — while keeping the science legible?

This repo is an answer to both.

---

## What is inside

| Layer | Files | Purpose |
|-------|-------|---------|
| Mechanistic core | `src/hybridbio/mechanistic.py` | Monod-type ODE for Xv, S, L, P, V |
| Data-driven correction | `src/hybridbio/corrections.py`, `torch_correction.py` | Bounded growth-rate multiplier behind a `Protocol` |
| Hybrid composition | `src/hybridbio/hybrid.py` | Wires mechanism + correction together |
| Scientific constraints | `src/hybridbio/constraints.py` | Non-negotiable biology/physics checks |
| Training | `src/hybridbio/training.py`, `rollout.py` | One-step and rollout training |
| Evaluation | `src/hybridbio/evaluation.py` | Metrics + admissibility, co-equal |
| Inference | `src/hybridbio/inference.py` | Load a saved model and predict trajectories |
| Reporting | `src/hybridbio/reporting.py` | Markdown/HTML reports for scientists and CI |
| Registry | `src/hybridbio/registry.py` | MLflow model registry with validation gate |
| Workflows | `workflows/` | Flyte, Optuna, and Ray Tune examples |
| CLI | `src/hybridbio/cli.py` | `hybridbio train | predict | sweep` |

---

## Design decisions

### The learned correction is a bounded multiplier on growth rate

```
mu_eff = mu_mech(S, L) * correction(features)
```

Why this seam:
- Mass balances stay structurally safe.
- The learned object is a curve a scientist can inspect and reject.
- It is narrow enough that swapping sklearn ↔ PyTorch is one new file.

### Admissibility is a gate, not a metric

A model passes only if it is accurate **and** satisfies scientific constraints.
The gate has teeth in four places:
1. `EvaluationReport.passed`
2. `tests/test_scientific_constraints.py` (injects a 3× rogue correction)
3. Flyte `validation_gate` task (fails the DAG)
4. Optuna/Ray Tune pruning (inadmissible trials are rejected, not penalised)

### Smoothness matters inside an ODE

The default correction is a smooth MLP. Tree ensembles are kept as a
documented warning: their discontinuities make adaptive ODE solvers hang.

### Train/serve skew is avoided deliberately

- Feature contract is versioned (`FEATURE_VERSION`, `FEATURE_NAMES`).
- Point features used inside the ODE are tested to match batch features used in
  training.
- Rollout training closes the feedback loop by training on simulated trajectories.

---

## Quick start

```bash
# Clone and enter the repo
cd hybrid-bioprocess-lab

# Create a virtual environment (optional but recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install the base package
pip install -e .

# Install with all extras for development
pip install -e ".[tracking,torch,ray,dev]"
```

### Run tests

```bash
pytest
```

Tests are split by concern:
- `test_mechanistic.py` — ODE, kinetics, mass balances
- `test_scientific_constraints.py` — biology/physics guardrails
- `test_regression.py` — golden values and feature contracts
- `test_torch_correction.py` — PyTorch correction model
- `test_inference.py` — saved model loading and prediction
- `test_registry.py` — MLflow registry integration
- `test_rollout.py` — rollout training
- `test_reporting.py` — report generation

### Train and save a model from the CLI

```bash
hybridbio train --out-dir ./models/run-001 --n-batches 24 --n-test 6 --report report.md
```

Use the PyTorch backend:

```bash
hybridbio train --out-dir ./models/run-torch --backend torch --report report.md
```

### Predict from a saved model

```bash
hybridbio predict --model ./models/run-001 --report prediction-report.md
```

### Run an Optuna sweep

```bash
hybridbio sweep --trials 30 --report sweep-report.md
```

### Run the Flyte workflow locally

```bash
pyflyte run workflows/flyte_training.py train_hybrid_wf --n_batches 24
```

### Run the Ray Tune sweep

```bash
python workflows/ray_tune_sweep.py --trials 30
```

---

## Docker

```bash
docker build -t hybridbio:latest .
docker run --rm hybridbio:latest train --out-dir /tmp/model --n-batches 8 --n-test 2
```

---

## Project structure

```
hybrid-bioprocess-lab/
├── src/hybridbio/          # Package source
├── tests/                  # Pytest suite
├── workflows/              # Flyte, Optuna, Ray Tune
├── notebooks/              # Exploratory notebook
├── docs/                   # Job description, learning log
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## Optional dependency groups

| Group | Install | Includes |
|-------|---------|----------|
| Base | `pip install -e .` | numpy, scipy, scikit-learn, joblib |
| Tracking | `pip install -e ".[tracking]"` | mlflow, optuna |
| Orchestration | `pip install -e ".[orchestration]"` | flytekit |
| Torch | `pip install -e ".[torch]"` | torch |
| Ray | `pip install -e ".[ray]"` | ray[tune] |
| Dev | `pip install -e ".[dev]"` | pytest, pytest-cov, ruff, mypy |

---

## Why this repo exists

See `docs/LEARNING_LOG.md` for the running notes: the mistakes, the fixes, and
the lessons. It is the most honest part of the project.

---

## License

See `LICENSE`.
