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

    python -m hybridbio sweep --trials 30
"""

from __future__ import annotations

import argparse
from typing import Any

import optuna

from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    TrainingConfig,
    evaluate,
    generate_dataset,
    mlp_estimator,
    train_correction,
    train_test_split_batches,
)


def build_objective(
    n_batches: int,
    n_test: int,
    seed: int,
) -> tuple[Any, float]:
    params = KineticParameters()
    batches = generate_dataset(n_batches=n_batches, seed=seed)
    train_batches, test_batches = train_test_split_batches(batches, n_test=n_test)
    baseline = evaluate(HybridModel.mechanistic_only(params), test_batches)
    baseline_nrmse = baseline.metrics["nrmse_mean"]

    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_int("width", 4, 48, log=True)
        depth = trial.suggest_int("depth", 1, 2)
        alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
        lo = trial.suggest_float("bound_lo", 0.2, 0.7)
        hi = trial.suggest_float("bound_hi", 1.2, 2.5)

        hidden = (width,) if depth == 1 else (width, max(width // 2, 2))
        cfg = TrainingConfig(bounds=(lo, hi))

        correction = train_correction(
            train_batches, params, cfg, estimator=mlp_estimator(hidden=hidden, alpha=alpha)
        )
        model = HybridModel(params=params, feed=FeedProfile(), correction=correction)
        report = evaluate(model, test_batches)

        # The gate. Not a penalty -- a veto.
        if not report.constraints_ok:
            raise optuna.TrialPruned(
                f"scientifically inadmissible: {report.n_violations()} violation(s)"
            )

        trial.set_user_attr("final_titre_rel_err", report.metrics.get("final_titre_rel_err", -1.0))
        trial.set_user_attr("baseline_nrmse", baseline_nrmse)
        return report.metrics["nrmse_mean"]

    return objective, baseline_nrmse


def main() -> None:
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
