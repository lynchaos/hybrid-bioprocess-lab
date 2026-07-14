"""Generate human-readable evaluation reports.

The output is meant for two audiences:
- Process Model Scientists, who want to see trajectories, constraints, and the
  correction curve over time.
- Engineers reviewing CI, who want a PASS/FAIL verdict and the metrics delta.

Reports can be markdown or HTML. No external templating engine is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluation import EvaluationReport
from .hybrid import HybridModel
from .inference import TrajectoryPrediction

Array = NDArray[np.float64]


def write_report(
    path: str | Path,
    hybrid: HybridModel,
    baseline_report: EvaluationReport | None = None,
    candidate_report: EvaluationReport | None = None,
    batches: list[Any] | None = None,
    trajectory: TrajectoryPrediction | None = None,
    format: str = "markdown",
) -> Path:
    """Write a report to disk."""
    path = Path(path)
    if format == "markdown":
        text = render_markdown(
            hybrid=hybrid,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            batches=batches,
            trajectory=trajectory,
        )
    elif format == "html":
        text = render_html(
            hybrid=hybrid,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            batches=batches,
            trajectory=trajectory,
        )
    else:
        raise ValueError(f"unsupported report format: {format}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def render_markdown(
    hybrid: HybridModel,
    baseline_report: EvaluationReport | None = None,
    candidate_report: EvaluationReport | None = None,
    batches: list[Any] | None = None,
    trajectory: TrajectoryPrediction | None = None,
) -> str:
    """Render a markdown report."""
    lines: list[str] = [
        "# Hybrid Bioprocess Model Report",
        "",
        "## Model configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in hybrid.params.as_dict().items():
        lines.append(f"| {k} | {v:.6g} |")
    lines.extend(["", f"- Feature version: `{hybrid.feature_version}`", ""])

    if candidate_report is not None:
        lines.extend(["## Candidate metrics", "", _metrics_table(candidate_report), ""])
    if baseline_report is not None:
        lines.extend(["## Baseline metrics", "", _metrics_table(baseline_report), ""])
    if candidate_report is not None and baseline_report is not None:
        lines.extend(
            [
                "## Delta (candidate - baseline)",
                "",
                _delta_table(candidate_report, baseline_report),
                "",
            ]
        )

    if trajectory is not None:
        lines.extend(
            [
                "## Single trajectory prediction",
                "",
                f"- Final titre: **{trajectory.final_titre:.2f} mg/L**",
                f"- Constraints: {trajectory.constraint_report.summary()}",
                (
                    f"- Final titre rel err: {trajectory.final_titre_rel_err:.4f}"
                    if trajectory.final_titre_rel_err is not None
                    else "- Final titre rel err: N/A"
                ),
                "",
            ]
        )

    if batches is not None and candidate_report is not None:
        lines.extend(["## Per-batch constraint violations", ""])
        for i, report in enumerate(candidate_report.constraints):
            lines.append(f"- Batch {i}: {report.summary()}")
        lines.append("")

    lines.append("## Correction curve sample")
    lines.append("")
    lines.append(_correction_curve_text(hybrid, trajectory))
    lines.append("")

    return "\n".join(lines)


def render_html(
    hybrid: HybridModel,
    baseline_report: EvaluationReport | None = None,
    candidate_report: EvaluationReport | None = None,
    batches: list[Any] | None = None,
    trajectory: TrajectoryPrediction | None = None,
) -> str:
    """Render a simple HTML report."""
    md = render_markdown(
        hybrid=hybrid,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        batches=batches,
        trajectory=trajectory,
    )
    body = md.replace("\n", "\n    ")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hybrid Bioprocess Model Report</title>
<style>
  body {{
    font-family: system-ui, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1rem;
  }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; }}
</style>
</head>
<body>
    {body}
</body>
</html>
"""


def _metrics_table(report: EvaluationReport) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k in sorted(report.metrics):
        lines.append(f"| {k} | {report.metrics[k]:.6f} |")
    lines.append(f"| violations | {report.n_violations()} |")
    lines.append(f"| passed | {report.passed} |")
    return "\n".join(lines)


def _delta_table(candidate: EvaluationReport, baseline: EvaluationReport) -> str:
    keys = sorted(set(candidate.metrics) & set(baseline.metrics))
    lines = ["| Metric | Delta |", "|--------|-------|"]
    for k in keys:
        delta = candidate.metrics[k] - baseline.metrics[k]
        lines.append(f"| {k} | {delta:+.6f} |")
    return "\n".join(lines)


def _correction_curve_text(
    hybrid: HybridModel,
    trajectory: TrajectoryPrediction | None = None,
) -> str:
    """Text summary of the correction the model applies."""
    from .hybrid import growth_multiplier_curve

    if trajectory is not None:
        t, Y = trajectory.t, trajectory.Y
    else:
        t, Y = hybrid.simulate()

    curve = growth_multiplier_curve(hybrid, t, Y)
    return (
        f"Sample correction multiplier over trajectory: "
        f"min={float(np.min(curve)):.4f}, max={float(np.max(curve)):.4f}, "
        f"mean={float(np.mean(curve)):.4f}"
    )
