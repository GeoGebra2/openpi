"""Parse LIBERO delay-sweep summaries and visualize failure rate vs. delay.

Scans one or more directories for ``summary.json`` files produced by
``run_image_delay_sweep.py`` / ``run_state_delay_sweep.py`` /
``run_state_noise_sweep.py``, detects the experiment type from each summary
(directory names are not trusted), and plots failure rate as a function of
delay for every camera / the robot state (delay and equal-strength noise).

Usage:
    # Auto-discover summaries under ../rollout and ./rollout (default):
    python analyze_delay_results.py

    # Explicit directories or summary.json files:
    python analyze_delay_results.py ../rollout/20260915_163411 path/to/summary.json

    # Success rate instead of failure rate, show the window interactively:
    python analyze_delay_results.py --metric success --show

Requires only the standard library and matplotlib.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Optional

COMPLETED_OUTCOMES = ("success", "failure")
BASELINE_CAMERA = "__none__"

CAMERA_LABELS = {
    "observation/image": "agentview (observation/image)",
    "observation/wrist_image": "wrist (observation/wrist_image)",
    "__all__": "all cameras",
}

EXPERIMENT_TITLES = {
    "libero_image_delay_sweep": "Image delay",
    "libero_state_delay_sweep": "State delay",
    "libero_state_noise_sweep": "State noise",
}

# Display names for the collapsed state series; camera labels stay as-is.
DISPLAY_LABELS = {
    "state": "self state (delay)",
    "state noise": "self state (noise)",
}


def _series_sort_key(label: str) -> tuple[int, str]:
    # Delay before noise before cameras; baseline and cameras keep name order.
    return ({"state": 0, "state noise": 1}.get(label, 2), label)


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
    if experiment not in EXPERIMENT_TITLES:
        print(f"warning: {path} has unknown experiment {experiment!r}, skipping", file=sys.stderr)
        return None
    if not summary.get("records"):
        print(f"warning: {path} contains no records, skipping", file=sys.stderr)
        return None
    return summary


def failure_rates(records: list[dict[str, Any]], key: Optional[str]) -> dict[int, float]:
    """Compute failure rate per delay level from raw records.

    ``key`` is the record field that selects a series (``delayed_camera`` for
    image sweeps, ``None`` for state sweeps). Returns {(series, delay): rate}.
    """
    grouped: dict[tuple[Optional[str], int], list[bool]] = collections.defaultdict(list)
    for record in records:
        if record.get("outcome") not in COMPLETED_OUTCOMES:
            continue
        series = record.get(key) if key else "state"
        grouped[(series, int(record["delay_ms_requested"]))].append(bool(record["success"]))
    return {
        (series, delay): 1.0 - sum(flags) / len(flags)
        for (series, delay), flags in grouped.items()
    }


def collect_curves(
    summaries: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Group failure-rate samples by experiment -> series -> delay.

    Each delay level maps to a list with one sample per summary file, so
    repeated runs of the same setting can be averaged.
    """
    curves: dict[str, dict[str, dict[int, list[float]]]] = {}
    for path, summary in summaries:
        experiment = summary["experiment"]
        key = "delayed_camera" if experiment == "libero_image_delay_sweep" else None
        rates = failure_rates(summary["records"], key)
        series_map = curves.setdefault(experiment, {})
        for (series, delay), rate in rates.items():
            if experiment == "libero_state_noise_sweep":
                label = "state noise"
            else:
                label = CAMERA_LABELS.get(series, series or "state")
            series_map.setdefault(label, {}).setdefault(delay, []).append(rate)
    return curves


def plot_curves(
    curves: dict[str, dict[str, dict[int, list[float]]]],
    metric: str,
    output: pathlib.Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # headless servers have no usable interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Merge all experiments into one axes; image-delay cameras keep their label,
    # the state series are named "self state (delay)" / "self state (noise)".
    merged: dict[str, dict[int, list[float]]] = {}
    for experiment in EXPERIMENT_TITLES:
        for label, delays_map in curves.get(experiment, {}).items():
            merged.setdefault(label, {}).update(
                {delay: list(samples) for delay, samples in delays_map.items()}
            )
    baseline = merged.get(BASELINE_CAMERA)
    if baseline:
        rate = sum(baseline[0]) / len(baseline[0])
        if metric == "success":
            rate = 1.0 - rate
        ax.axhline(rate, color="gray", linestyle="--", alpha=0.7, label=f"baseline (no delay): {rate:.2f}")
    for label in sorted(merged, key=_series_sort_key):
        if label == BASELINE_CAMERA:
            continue
        delays = sorted(merged[label])
        values = [sum(merged[label][d]) / len(merged[label][d]) for d in delays]
        if metric == "success":
            values = [1.0 - v for v in values]
        n_runs = max(len(v) for v in merged[label].values())
        name = DISPLAY_LABELS.get(label, label)
        line_label = name if n_runs == 1 else f"{name} (mean of {n_runs} runs)"
        ax.plot(delays, values, marker="o", label=line_label)
    ax.set_title("PI0 robustness to delayed observations (LIBERO)")
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel(f"{metric} rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()


def print_table(curves: dict[str, dict[str, dict[int, list[float]]]], metric: str) -> None:
    for experiment, series_map in curves.items():
        delays = sorted({d for series in series_map.values() for d in series})
        print(f"\n{EXPERIMENT_TITLES[experiment]} — {metric} rate (mean over runs)")
        header = ["series"] + [f"{d}ms" for d in delays]
        print("  " + " | ".join(f"{h:>42}" if i == 0 else f"{h:>7}" for i, h in enumerate(header)))
        for label in sorted(series_map, key=_series_sort_key):
            cells = []
            for delay in delays:
                samples = series_map[label].get(delay)
                if not samples:
                    cells.append(f"{'-':>7}")
                    continue
                value = sum(samples) / len(samples)
                if metric == "success":
                    value = 1.0 - value
                cells.append(f"{value:>7.2f}")
            print("  " + f"{DISPLAY_LABELS.get(label, label):>42}" + " | " + " | ".join(cells))


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
        choices=("failure", "success"),
        default="failure",
        help="Rate to plot (default: failure)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("delay_analysis.png"),
        help="Output image path (default: delay_analysis.png)",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window")
    args = parser.parse_args()

    summary_paths = find_summaries(args.paths)
    summaries = []
    for path in summary_paths:
        summary = load_summary(path)
        if summary is not None:
            summaries.append((path, summary))
            print(f"loaded {summary['experiment']:<28} <- {path}")
    if not summaries:
        raise SystemExit("No usable summaries found.")

    curves = collect_curves(summaries)
    print_table(curves, args.metric)
    plot_curves(curves, args.metric, args.output, args.show)


if __name__ == "__main__":
    main()
