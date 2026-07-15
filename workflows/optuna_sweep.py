"""Optuna sweep over the correction model, constrained by biology.

The one idea worth stealing from this file:

    A trial that violates a scientific constraint is PRUNED, not scored.

If you merely penalise violations in the objective, Optuna will happily trade
them off -- it will discover that a small mass-balance violation buys a large
RMSE improvement, and it will take that deal every single time, because that is
precisely what you asked it to do. Optimisers are extremely good at finding the
cheapest lie in your loss function.

So admissibility is not a term in the objective. It is a precondition for
having an objective at all.

`build_objective` itself lives in `hybridbio.optuna_sweep`, and is reused as-is
by both this standalone workflow script and the `hybridbio sweep` CLI command.
Two entry points into the same search space used to mean two copies of this
function drifting slowly apart -- exactly the kind of duplicated modelling
pattern a Process Model Scientist should never have to maintain twice. This
file is now the thin one.

    python workflows/optuna_sweep.py --trials 30
"""

from __future__ import annotations

import argparse

from hybridbio.optuna_sweep import build_objective


def main() -> None:
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    objective, baseline_nrmse = build_objective(args.n_batches, args.n_test, args.seed)

    study = optuna.create_study(direction="minimize", study_name="hybrid-correction")
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "=" * 60)
    print(f"mechanistic baseline nrmse : {baseline_nrmse:.4f}")
    print(f"best hybrid nrmse          : {study.best_value:.4f}")
    improvement = 100.0 * (1.0 - study.best_value / baseline_nrmse)
    print(f"improvement                : {improvement:+.1f}%")
    print(f"best params                : {study.best_params}")
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"pruned (inadmissible)      : {len(pruned)}/{len(study.trials)}")
    print("=" * 60)

    if improvement <= 0:
        raise SystemExit("the hybrid model did not beat the mechanistic baseline")


if __name__ == "__main__":
    main()
