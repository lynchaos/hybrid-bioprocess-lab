"""Generate human-readable evaluation reports.

The output is meant for two audiences:
- Process Model Scientists, who want to see trajectories, constraints, and the
  correction curve over time.
- Engineers reviewing CI, who want a PASS/FAIL verdict and the metrics delta.

Reports can be markdown or HTML. No external templating engine is required.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audit import CorrectionAudit
from .evaluation import EvaluationReport
from .hybrid import HybridModel
from .inference import TrajectoryPrediction
from .lineage import ExperimentManifest

Array = NDArray[np.float64]


def write_report(
    path: str | Path,
    hybrid: HybridModel,
    baseline_report: EvaluationReport | None = None,
    candidate_report: EvaluationReport | None = None,
    batches: list[Any] | None = None,
    trajectory: TrajectoryPrediction | None = None,
    manifest: ExperimentManifest | None = None,
    audit: CorrectionAudit | None = None,
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
            manifest=manifest,
            audit=audit,
        )
    elif format == "html":
        text = render_html(
            hybrid=hybrid,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            batches=batches,
            trajectory=trajectory,
            manifest=manifest,
            audit=audit,
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
    manifest: ExperimentManifest | None = None,
    audit: CorrectionAudit | None = None,
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

    if manifest is not None:
        lines.extend(
            [
                "## Experiment lineage",
                "",
                f"- Dataset: `{manifest.data_source}` / `{manifest.dataset_id}`",
                f"- Train batches: {', '.join(manifest.train_batch_ids)}",
                f"- Held-out batches: {', '.join(manifest.test_batch_ids)}",
                f"- Git revision: `{manifest.git_sha}`",
                f"- Promotion decision: **{manifest.promotion_decision}** "
                f"({manifest.promotion_reason})",
                "",
            ]
        )

    if audit is not None:
        lines.extend(["## Correction audit", ""])
        lines.append(
            "Multiplier range over observed states: "
            f"{audit.correction_min:.4f} to {audit.correction_max:.4f} "
            f"(mean {audit.correction_mean:.4f})."
        )
        lines.extend(
            [
                "",
                "| Feature | 10th-to-90th percentile effect |",
                "|---------|-------------------------------:|",
            ]
        )
        for name in audit.feature_names:
            lines.append(f"| {name} | {audit.feature_effects[name]:.6f} |")
        lines.extend(
            [
                "",
                "Effects are one-feature local perturbation diagnostics, not causal attributions.",
                "",
            ]
        )

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
    manifest: ExperimentManifest | None = None,
    audit: CorrectionAudit | None = None,
) -> str:
    """Render a simple HTML report."""
    md = render_markdown(
        hybrid=hybrid,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        batches=batches,
        trajectory=trajectory,
        manifest=manifest,
        audit=audit,
    )
    body = _markdown_to_html(md)
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


def _markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    rendered: list[str] = []
    index = 0
    in_list = False

    while index < len(lines):
        line = lines[index]
        if not line:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            index += 1
            continue

        if (
            line.startswith("| ")
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        ):
            if in_list:
                rendered.append("</ul>")
                in_list = False
            headers = _table_cells(line)
            rendered.append("<table>")
            rendered.append(
                "<thead><tr>"
                + "".join(f"<th>{_inline_html(cell)}</th>" for cell in headers)
                + "</tr></thead>"
            )
            rendered.append("<tbody>")
            index += 2
            while index < len(lines) and lines[index].startswith("| "):
                cells = _table_cells(lines[index])
                rendered.append(
                    "<tr>" + "".join(f"<td>{_inline_html(cell)}</td>" for cell in cells) + "</tr>"
                )
                index += 1
            rendered.extend(["</tbody>", "</table>"])
            continue

        heading = re.match(r"^(#{1,6}) (.+)$", line)
        if heading:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            rendered.append(f"<h{level}>{_inline_html(heading.group(2))}</h{level}>")
        elif line.startswith("- "):
            if not in_list:
                rendered.append("<ul>")
                in_list = True
            rendered.append(f"<li>{_inline_html(line[2:])}</li>")
        else:
            if in_list:
                rendered.append("</ul>")
                in_list = False
            rendered.append(f"<p>{_inline_html(line)}</p>")
        index += 1

    if in_list:
        rendered.append("</ul>")
    return "\n".join(rendered)


def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s|:-]+\|", line))


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
