"""Command-line interface for training, predicting, and reporting.

Examples
--------
Train and save a model:
    python -m hybridbio train --out-dir ./models/run-001

Predict from a saved model:
    python -m hybridbio predict --model ./models/run-001 --report report.md

Run an Optuna sweep:
    python -m hybridbio sweep --trials 30 --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audit import audit_correction
from .evaluation import compare
from .inference import HybridPredictor
from .lineage import build_manifest, write_manifest
from .mechanistic import FeedProfile, KineticParameters
from .reporting import write_report
from .training import TrainingConfig, train_and_evaluate


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("train", help="train a hybrid correction model")
    parser.add_argument("--out-dir", required=True, help="directory to save the model")
    parser.add_argument("--n-batches", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    parser.add_argument("--report", default=None, help="markdown report output path")
    parser.set_defaults(func=_train)


def _train(args: argparse.Namespace) -> int:
    from .data import generate_dataset, train_test_split_batches

    params = KineticParameters()
    cfg = TrainingConfig()
    batches = generate_dataset(n_batches=args.n_batches, seed=args.seed)
    train_batches, test_batches = train_test_split_batches(batches, n_test=args.n_test)

    estimator = None
    if args.backend == "torch":
        from .torch_correction import torch_estimator

        estimator = torch_estimator()

    hybrid, test_report, baseline_report = train_and_evaluate(
        train_batches, test_batches, params, cfg, estimator=estimator
    )

    out_dir = Path(args.out_dir)
    hybrid.save(out_dir)
    manifest = build_manifest(
        data_source="synthetic-fed-batch",
        dataset_id=f"seed-{args.seed}",
        train_batches=train_batches,
        test_batches=test_batches,
        params=params,
        candidate_report=test_report,
        baseline_report=baseline_report,
        training_config={
            "seed": args.seed,
            "backend": args.backend,
            "n_batches": args.n_batches,
            "n_test": args.n_test,
            "t_end_h": cfg.t_end_h,
            "dt_h": cfg.dt_h,
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)
    correction_audit = audit_correction(hybrid, train_batches)
    print(f"model saved to {out_dir}")
    print(f"lineage manifest written to {out_dir / 'manifest.json'}")
    print(test_report.render())
    print("\nbaseline vs candidate delta:")
    for k, v in compare(baseline_report, test_report).items():
        print(f"  {k}: {v:+.5f}")

    if args.report:
        write_report(
            path=args.report,
            hybrid=hybrid,
            baseline_report=baseline_report,
            candidate_report=test_report,
            batches=test_batches,
            manifest=manifest,
            audit=correction_audit,
        )
        print(f"report written to {args.report}")
    return 0


def _add_predict_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("predict", help="predict a trajectory from a saved model")
    parser.add_argument("--model", required=True, help="saved model directory")
    parser.add_argument("--y0", default=None, help="JSON list of 5 initial-state values")
    parser.add_argument("--feed-rate", type=float, default=None)
    parser.add_argument("--feed-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--report", default=None, help="markdown report output path")
    parser.set_defaults(func=_predict)


def _predict(args: argparse.Namespace) -> int:
    predictor = HybridPredictor.load(args.model)

    y0 = None
    if args.y0 is not None:
        y0_list = json.loads(args.y0)
        if len(y0_list) != 5:
            raise ValueError("y0 must be a list of 5 values")
        import numpy as np

        y0 = np.array(y0_list, dtype=float)

    feed = None
    if args.feed_rate is not None or args.feed_start is not None:
        feed = FeedProfile(
            rate=args.feed_rate if args.feed_rate is not None else predictor.model.feed.rate,
            start_h=(
                args.feed_start if args.feed_start is not None else predictor.model.feed.start_h
            ),
        )

    prediction = predictor.predict(y0=y0, feed=feed, t_end_h=args.t_end)
    print(f"final titre: {prediction.final_titre:.2f} mg/L")
    print(f"constraints: {prediction.constraint_report.summary()}")
    if prediction.final_titre_rel_err is not None:
        print(f"final titre rel err: {prediction.final_titre_rel_err:.4f}")

    if args.report:
        write_report(
            path=args.report,
            hybrid=predictor.model,
            baseline_report=None,
            candidate_report=None,
            trajectory=prediction,
        )
        print(f"report written to {args.report}")
    return 0


def _add_sweep_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser("sweep", help="run an Optuna hyperparameter sweep")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--n-batches", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", default=None, help="markdown report output path")
    parser.set_defaults(func=_sweep)


def _sweep(args: argparse.Namespace) -> int:
    from .optuna_sweep import build_objective

    objective, baseline_nrmse = build_objective(args.n_batches, args.n_test, args.seed)
    import optuna

    study = optuna.create_study(direction="minimize", study_name="hybrid-correction-cli")
    study.optimize(objective, n_trials=args.trials)

    print("=" * 60)
    print(f"mechanistic baseline nrmse : {baseline_nrmse:.4f}")
    print(f"best hybrid nrmse          : {study.best_value:.4f}")
    improvement = 100.0 * (1.0 - study.best_value / baseline_nrmse)
    print(f"improvement                : {improvement:+.1f}%")
    print(f"best params                : {study.best_params}")
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"pruned (inadmissible)      : {len(pruned)}/{len(study.trials)}")
    print("=" * 60)
    return 0 if improvement > 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hybridbio",
        description="Hybrid bioprocess model training, inference, and reporting",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(subparsers)
    _add_predict_parser(subparsers)
    _add_sweep_parser(subparsers)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
