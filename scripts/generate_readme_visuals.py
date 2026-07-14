"""Generate the tracked evidence graphic shown in the README.

The figure is intentionally generated from the public package rather than
hand-edited. It communicates synthetic-study evidence, not production claims.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybridbio import (
    KineticParameters,
    StudyConfig,
    TrainingConfig,
    generate_dataset,
    run_repeated_study,
    train_and_evaluate,
    train_test_split_batches,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "docs" / "assets" / "synthetic-study-evidence.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    params = KineticParameters()
    training_config = TrainingConfig(seed=7)
    batches = generate_dataset(n_batches=18, seed=7)
    train_batches, test_batches = train_test_split_batches(batches, n_test=4)
    _, hybrid_report, baseline_report = train_and_evaluate(
        train_batches,
        test_batches,
        p=params,
        cfg=training_config,
    )
    study = run_repeated_study(
        StudyConfig(
            seeds=(7, 17, 29),
            n_batches=12,
            n_test=3,
            n_bootstrap=500,
            bootstrap_seed=7,
        ),
        params=params,
        training_config=training_config,
    )

    baseline = baseline_report.metrics["nrmse_mean"]
    hybrid = hybrid_report.metrics["nrmse_mean"]
    ci = study.nrmse_delta
    fig, (axis_metrics, axis_study) = plt.subplots(1, 2, figsize=(11, 4.25))
    fig.patch.set_facecolor("white")

    bars = axis_metrics.bar(
        ["Mechanistic\nbaseline", "Constrained\nhybrid"],
        [baseline, hybrid],
        color=["#496A81", "#008B8B"],
        width=0.6,
    )
    axis_metrics.bar_label(bars, labels=[f"{baseline:.3f}", f"{hybrid:.3f}"], padding=4)
    axis_metrics.set_ylim(0, max(baseline, hybrid) * 1.3)
    axis_metrics.set_ylabel("Held-out mean NRMSE")
    axis_metrics.set_title("One held-out split")
    axis_metrics.spines[["top", "right"]].set_visible(False)

    error_left = ci.estimate - ci.lower
    error_right = ci.upper - ci.estimate
    axis_study.errorbar(
        ci.estimate,
        0,
        xerr=np.array([[error_left], [error_right]]),
        fmt="o",
        color="#008B8B",
        capsize=6,
        markersize=9,
    )
    axis_study.axvline(0, color="#9A3412", linestyle="--", linewidth=1.5, label="No difference")
    axis_study.set_yticks([])
    axis_study.set_xlabel("Hybrid minus baseline NRMSE")
    axis_study.set_title("3 seeds x 3 held-out batches\n95% paired bootstrap CI")
    axis_study.text(
        ci.estimate,
        0.08,
        f"{ci.estimate:+.4f}  [{ci.lower:+.4f}, {ci.upper:+.4f}]",
        ha="center",
        color="#004F4F",
        fontweight="bold",
    )
    axis_study.legend(frameon=False, loc="lower right")
    axis_study.spines[["top", "right", "left"]].set_visible(False)
    axis_study.set_ylim(-0.15, 0.22)

    fig.suptitle(
        "Synthetic Study Evidence: Accuracy Must Clear the Scientific Gate",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Synthetic data only. All repeated-study candidates were constraint-admissible.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Wrote {output.relative_to(project_root)}")


if __name__ == "__main__":
    main()
