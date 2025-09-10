"""Flyte workflow: train a hybrid correction model behind a scientific gate.

The shape of this workflow is the argument.

`validation_gate` sits between training and registration, and it can veto. A
model that improves nRMSE but violates a mass balance never reaches the
registry -- it fails the workflow, loudly, with the violated constraint named
in the failure message. The gate is not a report someone might read. It is a
task that can fail the DAG.

Run locally, no cluster needed:
    pyflyte run workflows/flyte_training.py train_hybrid_wf --n_batches 24

Register:
    pyflyte register workflows/
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from flytekit import ImageSpec, Resources, task, workflow
except ImportError:  # pragma: no cover - flyte is an optional extra
    # Local fallback so the module remains importable (and testable) without
    # flytekit installed. The decorators become no-ops; the logic is unchanged.
    def task(_fn=None, **_kwargs):  # type: ignore[no-redef]
        def wrap(fn):
            return fn

        return wrap(_fn) if _fn else wrap

    def workflow(_fn=None, **_kwargs):  # type: ignore[no-redef]
        def wrap(fn):
            return fn

        return wrap(_fn) if _fn else wrap

    Resources = ImageSpec = None  # type: ignore[assignment]


from hybridbio import (
    FeedProfile,
    HybridModel,
    KineticParameters,
    TrainingConfig,
    evaluate,
    generate_dataset,
    track_run,
    train_correction,
    train_test_split_batches,
)


@dataclass
class TrainingResult:
    """Flyte passes dataclasses between tasks; keep them flat and serialisable."""

    nrmse_mean: float
    final_titre_rel_err: float
    baseline_nrmse_mean: float
    n_violations: int
    passed: bool


@task(requests=Resources(cpu="1", mem="2Gi") if Resources else None)
def make_dataset(n_batches: int, seed: int) -> int:
    """In a real deployment this reads batch records from the historian.

    Here it returns the seed, and downstream tasks regenerate deterministically
    -- because shipping numpy arrays between Flyte tasks as literals is a fine
    way to discover the limits of your blob store.
    """
    _ = generate_dataset(n_batches=n_batches, seed=seed)  # validate it builds
    return seed


@task(requests=Resources(cpu="2", mem="4Gi") if Resources else None)
def train_and_score(seed: int, n_batches: int, n_test: int) -> TrainingResult:
    params = KineticParameters()
    cfg = TrainingConfig()

    batches = generate_dataset(n_batches=n_batches, seed=seed)
    train_batches, test_batches = train_test_split_batches(batches, n_test=n_test)

    with track_run("flyte-hybrid-train") as run:
        run.feature_contract()
        run.params(n_batches=n_batches, n_test=n_test, seed=seed, **params.as_dict())

        correction = train_correction(train_batches, params, cfg)
        hybrid = HybridModel(params=params, feed=FeedProfile(), correction=correction)
        baseline = HybridModel.mechanistic_only(params)

        report = evaluate(hybrid, test_batches)
        baseline_report = evaluate(baseline, test_batches)

        run.metrics(report.metrics)
        run.metrics({f"baseline_{k}": v for k, v in baseline_report.metrics.items()})
        run.tag("constraints_ok", str(report.constraints_ok))
        print(report.render())

    return TrainingResult(
        nrmse_mean=report.metrics["nrmse_mean"],
        final_titre_rel_err=report.metrics.get("final_titre_rel_err", float("nan")),
        baseline_nrmse_mean=baseline_report.metrics["nrmse_mean"],
        n_violations=report.n_violations(),
        passed=report.passed,
    )


@task
def validation_gate(result: TrainingResult) -> TrainingResult:
    """The gate. It can, and should, fail the pipeline.

    Two independent conditions, and BOTH must hold:

      1. Zero scientific constraint violations. Non-negotiable. A model that
         breaks a mass balance is not a slightly worse model, it is not a model.
      2. The hybrid must beat the mechanistic baseline it corrects. Otherwise
         the ML layer is pure operational cost with negative scientific value,
         and someone should be told rather than allowed to keep tuning.
    """
    if result.n_violations > 0:
        raise ValueError(
            f"SCIENTIFIC VALIDATION FAILED: {result.n_violations} constraint "
            "violation(s). Refusing to register this model regardless of its metrics."
        )
    if result.nrmse_mean >= result.baseline_nrmse_mean:
        raise ValueError(
            f"REGRESSION: hybrid nrmse {result.nrmse_mean:.4f} did not improve on the "
            f"mechanistic baseline {result.baseline_nrmse_mean:.4f}. The correction "
            "model is not earning its keep."
        )
    return result


@task
def register_model(result: TrainingResult) -> str:
    """Stand-in for pushing to a model registry."""
    return (
        f"registered: nrmse={result.nrmse_mean:.4f} "
        f"(baseline {result.baseline_nrmse_mean:.4f}), "
        f"titre_err={result.final_titre_rel_err:.4f}, violations=0"
    )


@workflow
def train_hybrid_wf(n_batches: int = 24, n_test: int = 6, seed: int = 7) -> str:
    seed_out = make_dataset(n_batches=n_batches, seed=seed)
    result = train_and_score(seed=seed_out, n_batches=n_batches, n_test=n_test)
    gated = validation_gate(result=result)
    return register_model(result=gated)


if __name__ == "__main__":
    print(train_hybrid_wf())
