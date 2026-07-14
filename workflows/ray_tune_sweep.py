"""Ray Tune sweep over the correction model, with the same scientific gate.

This mirrors optuna_sweep.py but exercises Ray Tune, which the job description
lists explicitly. The contract is identical: a trial that violates a scientific
constraint is recorded for diagnostics and excluded from selection, not scored
with an artificial penalty.

Run:
    python workflows/ray_tune_sweep.py --trials 30
"""

from __future__ import annotations

import argparse
import os
import tempfile

from ray import train as ray_train
from ray import tune

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


def _objective(config: dict) -> None:
    """Single Ray Tune trial."""
    params = KineticParameters()
    cfg = TrainingConfig(bounds=(config["bound_lo"], config["bound_hi"]))
    hidden = (config["width"],) if config["depth"] == 1 else (config["width"], config["width"] // 2)

    correction = train_correction(
        _TRAIN_BATCHES,
        params,
        cfg,
        estimator=mlp_estimator(hidden=hidden, alpha=config["alpha"]),
    )
    model = HybridModel(params=params, feed=FeedProfile(), correction=correction)
    report = evaluate(model, _TEST_BATCHES)

    if not report.constraints_ok:
        # NaN is deliberately not a loss. Invalid trials remain visible in Ray
        # diagnostics but are filtered out before any candidate is selected.
        ray_train.report(
            {
                "nrmse_mean": float("nan"),
                "constraints_ok": False,
                "n_violations": report.n_violations(),
            }
        )
        return

    ray_train.report(
        {
            "nrmse_mean": report.metrics["nrmse_mean"],
            "final_titre_rel_err": report.metrics.get("final_titre_rel_err", -1.0),
            "constraints_ok": True,
            "n_violations": 0,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    global _TRAIN_BATCHES, _TEST_BATCHES  # noqa: PLW0603
    batches = generate_dataset(n_batches=args.n_batches, seed=args.seed)
    _TRAIN_BATCHES, _TEST_BATCHES = train_test_split_batches(batches, n_test=args.n_test)

    baseline = evaluate(HybridModel.mechanistic_only(KineticParameters()), _TEST_BATCHES)
    baseline_nrmse = baseline.metrics["nrmse_mean"]

    search_space = {
        "width": tune.lograndint(4, 48),
        "depth": tune.choice([1, 2]),
        "alpha": tune.loguniform(1e-4, 1.0),
        "bound_lo": tune.uniform(0.2, 0.7),
        "bound_hi": tune.uniform(1.2, 2.5),
    }

    # Use a local temp directory so the sweep does not need cluster storage.
    storage_path = tempfile.mkdtemp(prefix="ray-tune-hybrid-")

    tuner = tune.Tuner(
        _objective,
        param_space=search_space,
        run_config=tune.RunConfig(
            name="hybrid-correction-ray",
            storage_path=storage_path,
        ),
        tune_config=tune.TuneConfig(
            metric="nrmse_mean",
            mode="min",
            num_samples=args.trials,
        ),
    )
    results = tuner.fit()
    admissible = [result for result in results if result.metrics.get("constraints_ok")]
    if not admissible:
        raise SystemExit("no scientifically admissible Ray Tune trial was produced")
    best = min(admissible, key=lambda result: result.metrics["nrmse_mean"])

    print("=" * 60)
    print(f"mechanistic baseline nrmse : {baseline_nrmse:.4f}")
    print(f"best hybrid nrmse          : {best.metrics['nrmse_mean']:.4f}")
    improvement = 100.0 * (1.0 - best.metrics["nrmse_mean"] / baseline_nrmse)
    print(f"improvement                : {improvement:+.1f}%")
    print(f"best params                : {best.config}")
    inadmissible = sum(1 for result in results if not result.metrics.get("constraints_ok", True))
    print(f"inadmissible trials        : {inadmissible}/{args.trials}")
    print(f"Ray storage                : {storage_path}")
    print("=" * 60)

    if improvement <= 0:
        raise SystemExit("the hybrid model did not beat the mechanistic baseline")


if __name__ == "__main__":
    # Limit Ray's resource detection noise on a laptop.
    os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
    main()
