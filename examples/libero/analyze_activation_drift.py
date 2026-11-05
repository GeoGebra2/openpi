"""Visualize layer-wise representation drift from activation-drift sweep summaries.

Scans one or more directories for ``summary.json`` files produced by
``run_activation_drift_sweep.py`` (experiment ``libero_activation_drift_sweep``;
other sweeps are skipped with a warning) and plots how the drift between delayed
and current observations evolves with network depth:

1. ``drift_vs_depth.png`` — drift along the concatenated pipeline axis
   (one named SigLIP camera -> Gemma prefix layers + final -> action tokens averaged
   over denoise steps -> final-step effective 7D flow). These are different spaces;
   cross-module changes are not amplification factors.
2. ``action_expert_heatmap.png`` — action-expert drift per (denoise step, layer),
   one heatmap per delay target x delay level.
3. ``flow_vt_breakdown.png`` — flow vector field drift per denoise step, split by
   action-dimension group (pos / rot / gripper).

Usage:
    # Auto-discover summaries under ../rollout and ./rollout (default):
    python analyze_activation_drift.py

    # Explicit directories or summary.json files, relative L2 instead of
    # cosine distance, show windows interactively:
    python analyze_activation_drift.py ../rollout/20260915_163411 --metric rel_l2 --show

Requires only the standard library, numpy, and matplotlib (imported lazily).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Optional

import numpy as np

EXPERIMENT = "libero_activation_drift_sweep"
BASELINE_TARGET = "__none__"
BASELINE_DELAY_MS = 0

# Region boundaries along the concatenated depth axis, in order.
REGION_NAMES = ("siglip", "gemma prefix", "prefix final", "action expert", "v_t (final)")


def find_summaries(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Resolve files/directories into a sorted list of summary.json paths."""
    found: list[pathlib.Path] = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif (path / "summary.json").is_file():
            found.append(path / "summary.json")
        elif path.is_dir():
            found.extend(sorted(path.glob("*/summary.json")))
        else:
            print(f"warning: {path} does not exist, skipping", file=sys.stderr)
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise SystemExit(f"No summary.json found under: {', '.join(map(str, paths))}")
    return unique


def load_summary(path: pathlib.Path) -> Optional[dict[str, Any]]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"warning: could not parse {path}: {error}", file=sys.stderr)
        return None
    experiment = summary.get("experiment")
    if experiment != EXPERIMENT:
        print(f"warning: {path} has experiment {experiment!r} (expected {EXPERIMENT!r}), skipping", file=sys.stderr)
        return None
    if summary.get("schema_version") != 2:
        print(f"warning: {path} uses legacy unpaired/pooled metrics; run a new schema-v2 sweep", file=sys.stderr)
        return None
    if not summary.get("drift"):
        print(f"warning: {path} contains no drift block, skipping", file=sys.stderr)
        return None
    return summary


def collect_drift(
    summaries: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Average the summary drift blocks: target -> delay_ms -> flat key -> array."""
    grouped: dict[str, dict[int, dict[str, list[np.ndarray]]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for _path, summary in summaries:
        if summary.get("schema_version") != 2:
            raise ValueError("Cannot combine legacy metrics with paired token-level schema v2")
        for target, delays in summary["drift"].items():
            for delay_str, entry in delays.items():
                for key, values in entry.items():
                    grouped[target][int(delay_str)][key].append(np.asarray(values, dtype=np.float64))
    return {
        target: {
            delay: {key: np.mean(samples, axis=0) for key, samples in entry.items()} for delay, entry in delays.items()
        }
        for target, delays in grouped.items()
    }


def _metric_suffix(metric: str) -> str:
    return {"cosine": "cosine", "rel_l2": "rel_l2"}[metric]


def depth_curve(
    entry: dict[str, np.ndarray], metric: str, camera: str = "base_0_rgb"
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Concatenate per-region drift into one depth axis.

    Returns (curve, boundaries) where boundaries lists (start_index, region_name).
    For metric="cosine" the curve is cosine distance (1 - cosine similarity), so
    larger always means more drift.
    """
    suffix = _metric_suffix(metric)
    parts: list[np.ndarray] = []
    boundaries: list[tuple[int, str]] = []
    cursor = 0

    def _append(name: str, values: np.ndarray) -> None:
        nonlocal cursor
        boundaries.append((cursor, name))
        parts.append(np.atleast_1d(values))
        cursor += len(parts[-1])

    key = f"siglip/{camera}/block_tokens_{suffix}"
    if key in entry:
        _append("siglip", entry[key])
    _append("gemma prefix", entry[f"gemma_prefix/all/layer_tokens_{suffix}"])
    _append("prefix final", np.atleast_1d(entry[f"gemma_prefix/all/final_tokens_{suffix}"]))
    # Action expert: average the [num_steps, L] drift over denoise steps.
    _append("action expert", np.asarray(entry[f"action_expert/action/layer_tokens_{suffix}"]).mean(axis=0))
    # Flow vector field at the final denoise step.
    _append("v_t (final)", np.atleast_1d(np.asarray(entry[f"flow/v_t_{suffix}"])[-1]))

    curve = np.concatenate(parts)
    if metric == "cosine":
        curve = 1.0 - curve
    return curve, boundaries


def _metric_label(metric: str) -> str:
    return "cosine distance (1 - cos)" if metric == "cosine" else "relative L2"


def region_means(curve: np.ndarray, boundaries: list[tuple[int, str]]) -> dict[str, float]:
    """Mean drift per depth region (for the compact text table)."""
    means: dict[str, float] = {}
    for index, (start, name) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(curve)
        means[name] = float(np.mean(curve[start:end]))
    return means


def plot_drift_vs_depth(
    drift: dict[str, dict[int, dict[str, np.ndarray]]],
    metric: str,
    output: pathlib.Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # headless servers have no usable interactive backend
    import matplotlib.pyplot as plt

    targets = sorted(target for target in drift if target != BASELINE_TARGET)
    if not targets:
        print("warning: no non-baseline delay targets found, skipping drift_vs_depth", file=sys.stderr)
        return
    baseline_curve = None
    baseline_entry = drift.get(BASELINE_TARGET, {}).get(BASELINE_DELAY_MS)
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5.5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        camera = "left_wrist_0_rgb" if target == "observation/wrist_image" else "base_0_rgb"
        boundaries = None
        for delay in sorted(drift[target]):
            curve, boundaries = depth_curve(drift[target][delay], metric, camera)
            ax.plot(np.arange(len(curve)), curve, marker=".", markersize=3, linewidth=1.2, label=f"{delay} ms")
        if baseline_entry is not None:
            baseline_curve, _ = depth_curve(baseline_entry, metric, camera)
            ax.plot(
                np.arange(len(baseline_curve)),
                baseline_curve,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="same-input/shared-noise baseline",
            )
        if boundaries:
            for start, name in boundaries[1:]:
                ax.axvline(start - 0.5, color="black", alpha=0.25, linewidth=0.8)
            for start, name in boundaries:
                ax.annotate(
                    name,
                    xy=(start, 1.0),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    fontsize=7,
                    va="top",
                    ha="right",
                    alpha=0.7,
                )
        ax.set_title(f"{target}\nSigLIP: {camera}")
        ax.set_xlabel("pipeline depth (siglip -> prefix -> action expert -> v_t)")
        ax.set_ylabel(_metric_label(metric))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Activation drift vs. network depth ({_metric_label(metric)})")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()


def plot_action_expert_heatmap(
    drift: dict[str, dict[int, dict[str, np.ndarray]]],
    metric: str,
    output: pathlib.Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = _metric_suffix(metric)
    targets = sorted(target for target in drift if target != BASELINE_TARGET)
    delays = sorted({delay for target in targets for delay in drift[target]})
    if not targets or not delays:
        print("warning: nothing to plot for action_expert_heatmap", file=sys.stderr)
        return
    fig, axes = plt.subplots(len(targets), len(delays), figsize=(4 * len(delays), 3.2 * len(targets)), squeeze=False)
    all_values = [
        np.asarray(entry[f"action_expert/action/layer_tokens_{suffix}"])
        for target in targets
        for entry in drift[target].values()
    ]
    maximum = max(float(np.max(1.0 - x if metric == "cosine" else x)) for x in all_values)
    for row, target in enumerate(targets):
        for col, delay in enumerate(delays):
            ax = axes[row][col]
            entry = drift[target].get(delay)
            if entry is None:
                ax.axis("off")
                continue
            values = np.asarray(entry[f"action_expert/action/layer_tokens_{suffix}"])  # [num_steps, L]
            if metric == "cosine":
                values = 1.0 - values
            image = ax.imshow(
                values.T, aspect="auto", origin="lower", interpolation="nearest", vmin=0, vmax=max(maximum, 1e-8)
            )
            fig.colorbar(image, ax=ax, fraction=0.046)
            if row == 0:
                ax.set_title(f"{delay} ms")
            if col == 0:
                ax.set_ylabel(f"{target}\nlayer")
            if row == len(targets) - 1:
                ax.set_xlabel("denoise step")
    fig.suptitle(f"Action-token drift per (denoise step, layer), shared color scale - {_metric_label(metric)}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()


def plot_flow_vt_breakdown(
    drift: dict[str, dict[int, dict[str, np.ndarray]]],
    metric: str,
    output: pathlib.Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = _metric_suffix(metric)
    groups = ("pos", "rot", "gripper")
    group_styles = {"pos": "-", "rot": "--", "gripper": ":"}
    targets = sorted(target for target in drift if target != BASELINE_TARGET)
    if not targets:
        print("warning: no non-baseline delay targets found, skipping flow_vt_breakdown", file=sys.stderr)
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        for delay in sorted(drift[target]):
            entry = drift[target][delay]
            for group in groups:
                values = np.asarray(entry[f"flow/v_t_{group}_{suffix}"])  # [num_steps]
                if metric == "cosine":
                    values = 1.0 - values
                ax.plot(
                    np.arange(len(values)),
                    values,
                    group_styles[group],
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    label=f"{delay} ms {group}",
                )
        baseline = drift.get(BASELINE_TARGET, {}).get(BASELINE_DELAY_MS)
        if baseline is not None:
            for group in groups:
                values = np.asarray(baseline[f"flow/v_t_{group}_{suffix}"])
                if metric == "cosine":
                    values = 1.0 - values
                ax.plot(
                    np.arange(len(values)),
                    values,
                    group_styles[group],
                    color="gray",
                    linewidth=1.5,
                    label=f"shared-noise baseline {group}",
                )
        ax.set_title(target)
        ax.set_xlabel("denoise step")
        ax.set_ylabel(_metric_label(metric))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"Flow vector field drift by action-dim group ({_metric_label(metric)})")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()


def print_table(drift: dict[str, dict[int, dict[str, np.ndarray]]], metric: str) -> None:
    """Compact per-region mean drift table, one block per delay target."""
    print(f"\nActivation drift by pipeline region ({_metric_label(metric)}, mean over queries/runs)")
    for target in sorted(drift):
        delays = sorted(drift[target])
        header = ["delay"] + list(REGION_NAMES)
        print(f"\n  {target}")
        print("    " + " | ".join(f"{h:>14}" for h in header))
        for delay in delays:
            camera = "left_wrist_0_rgb" if target == "observation/wrist_image" else "base_0_rgb"
            curve, boundaries = depth_curve(drift[target][delay], metric, camera)
            means = region_means(curve, boundaries)
            cells = [f"{means.get(name, float('nan')):>14.4f}" for name in REGION_NAMES]
            label = "baseline" if target == BASELINE_TARGET else f"{delay} ms"
            print("    " + f"{label:>14}" + " | " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        default=[repo_root.parent / "rollout", repo_root / "rollout"],
        help="Directories to scan for summary.json, or summary.json files "
        "(default: <repo>/../rollout and <repo>/rollout)",
    )
    parser.add_argument(
        "--metric",
        choices=("cosine", "rel_l2"),
        default="cosine",
        help="Drift metric to plot (default: cosine, shown as 1 - cosine similarity)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory for the output PNGs (default: current directory)",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows")
    args = parser.parse_args()

    summary_paths = find_summaries(args.paths)
    summaries = []
    for path in summary_paths:
        summary = load_summary(path)
        if summary is not None:
            summaries.append((path, summary))
            print(f"loaded {summary['experiment']:<30} <- {path}")
    if not summaries:
        raise SystemExit("No usable activation-drift summaries found.")

    drift = collect_drift(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_table(drift, args.metric)
    plot_drift_vs_depth(drift, args.metric, args.output_dir / "drift_vs_depth.png", args.show)
    plot_action_expert_heatmap(drift, args.metric, args.output_dir / "action_expert_heatmap.png", args.show)
    plot_flow_vt_breakdown(drift, args.metric, args.output_dir / "flow_vt_breakdown.png", args.show)


if __name__ == "__main__":
    main()
