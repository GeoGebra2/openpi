"""Analyze time-shift-matching sweeps produced by run_shift_match_sweep.py.

Scans one or more directories for ``summary.json`` files with
``experiment == "libero_shift_match_sweep"``, loads the per-run ``*_chunks.npz``
files they reference, and tests the replay hypothesis: if a delayed observation
makes PI0 replay a stale plan, the L2 distance between the delayed chunk at a
replan step t and the shadow (current-observation) chunk at step t-s should be
minimal when the shift s equals the injected delay.

Outputs (written to the output directory):
  - shift_match_spectrum.png    mean chunk L2 vs. shift, per target/delay level
  - shift_match_gap.png         gap (s=0) vs. match (s=true delay) vs. noise floor
  - shift_match_chunk_step.png  per-chunk-step L2 at s=0 and at s=true delay
  - shift_match_action_dims.png executed-portion L2 per action-dim group

A table with gap / match / floor / match-to-gap ratio is printed per target.

Usage:
    # Auto-discover summaries under ../rollout and ./rollout (default):
    python analyze_shift_match.py

    # Explicit directories or summary.json files:
    python analyze_shift_match.py ../rollout/20260915_163411 path/to/summary.json -o out/

Requires only the standard library, numpy, and matplotlib (matplotlib is
imported lazily, so the metric functions are importable without it).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Optional

import numpy as np

EXPERIMENT_NAME = "libero_shift_match_sweep"
BASELINE_TARGET = "__none__"

ACTION_DIM_GROUPS = {
    "position": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(6, 7),
}

TARGET_LABELS = {
    "observation/image": "agentview (observation/image)",
    "observation/wrist_image": "wrist (observation/wrist_image)",
    "state": "self state",
    "__all__": "all inputs (__all__)",
}


# ---------------------------------------------------------------------------
# Summary discovery / loading
# ---------------------------------------------------------------------------


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
    if experiment != EXPERIMENT_NAME:
        print(f"warning: {path} has experiment {experiment!r} (not {EXPERIMENT_NAME!r}), skipping", file=sys.stderr)
        return None
    if not summary.get("records"):
        print(f"warning: {path} contains no records, skipping", file=sys.stderr)
        return None
    return summary


def _resolve_chunks_path(summary_dir: pathlib.Path, chunks_value: str) -> Optional[pathlib.Path]:
    path = pathlib.Path(chunks_value)
    candidates = [path] if path.is_absolute() else [path, summary_dir / path, summary_dir / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_runs(summary_path: pathlib.Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Load chunk arrays for every record that references an existing npz."""
    runs = []
    for record in summary["records"]:
        chunks_value = record.get("chunks_path")
        if not chunks_value:
            print(
                f"warning: run {record.get('run_index')} in {summary_path} has no chunks_path, skipping",
                file=sys.stderr,
            )
            continue
        chunks_path = _resolve_chunks_path(summary_path.parent, chunks_value)
        if chunks_path is None:
            print(f"warning: chunks file {chunks_value} not found, skipping", file=sys.stderr)
            continue
        data = np.load(chunks_path)
        runs.append(
            {
                "delay_target": record["delay_target"],
                "delay_ms": int(record["delay_ms_requested"]),
                "delay_steps": int(record["delay_steps"]),
                "steps": np.asarray(data["steps"]),
                "shadow_chunks": np.asarray(data["shadow_chunks"]),
                "replan_step_indices": np.asarray(data["replan_step_indices"]),
                "delayed_chunks": np.asarray(data["delayed_chunks"]),
            }
        )
    return runs


# ---------------------------------------------------------------------------
# Pure metric functions (importable without matplotlib)
# ---------------------------------------------------------------------------


def chunk_l2(a: np.ndarray, b: np.ndarray, dims: Any = slice(None)) -> np.ndarray:
    """Per-chunk-step L2 distance between two [H, A] chunks, over ``dims`` -> [H]."""
    return np.linalg.norm(np.asarray(a)[:, dims] - np.asarray(b)[:, dims], axis=-1)


def shift_distances(
    delayed_chunk: np.ndarray,
    shadow_chunks: np.ndarray,
    replan_step: int,
    max_shift: int,
) -> dict[int, np.ndarray]:
    """Per-chunk-step L2 between a delayed chunk and shadow chunks at t-s.

    Returns {shift s: [H] distances} for every s in 0..max_shift with
    ``replan_step - s >= 0``.
    """
    result = {}
    for shift in range(0, min(max_shift, replan_step) + 1):
        result[shift] = chunk_l2(delayed_chunk, shadow_chunks[replan_step - shift])
    return result


def accumulate_spectrum(runs: list[dict[str, Any]], max_shift: int) -> dict[int, list[float]]:
    """Collect full-chunk mean L2 samples per shift over all queries and runs."""
    samples: dict[int, list[float]] = collections.defaultdict(list)
    for run in runs:
        for query_index, replan_step in enumerate(run["replan_step_indices"]):
            distances = shift_distances(
                run["delayed_chunks"][query_index], run["shadow_chunks"], int(replan_step), max_shift
            )
            for shift, l2 in distances.items():
                samples[shift].append(float(np.mean(l2)))
    return samples


def mean_spectrum(samples: dict[int, list[float]]) -> dict[int, float]:
    """Average per-shift L2 samples collected by accumulate_spectrum."""
    return {shift: float(np.mean(values)) for shift, values in samples.items() if values}


def chunk_step_curve(runs: list[dict[str, Any]], shift: int) -> Optional[np.ndarray]:
    """Mean per-chunk-step L2 [H] between delayed chunks and shadow at t-shift."""
    curves = []
    for run in runs:
        for query_index, replan_step in enumerate(run["replan_step_indices"]):
            if int(replan_step) - shift < 0:
                continue
            curves.append(
                chunk_l2(run["delayed_chunks"][query_index], run["shadow_chunks"][int(replan_step) - shift])
            )
    if not curves:
        return None
    return np.mean(np.stack(curves), axis=0)


def action_dim_distances(runs: list[dict[str, Any]], executed_steps: int) -> dict[str, Optional[float]]:
    """Executed-portion (chunk steps 0..executed_steps-1) L2 per action-dim group at s=0."""
    result: dict[str, Optional[float]] = {}
    for name, dims in ACTION_DIM_GROUPS.items():
        values = []
        for run in runs:
            for query_index, replan_step in enumerate(run["replan_step_indices"]):
                l2 = chunk_l2(
                    run["delayed_chunks"][query_index], run["shadow_chunks"][int(replan_step)], dims
                )
                values.append(float(np.mean(l2[:executed_steps])))
        result[name] = float(np.mean(values)) if values else None
    return result


def noise_floor(baseline_runs: list[dict[str, Any]]) -> Optional[float]:
    """Delayed-vs-shadow L2 on baseline runs (identical inputs): policy sampling noise."""
    if not baseline_runs:
        return None
    spectrum = mean_spectrum(accumulate_spectrum(baseline_runs, max_shift=0))
    return spectrum.get(0)


def failure_rates(summaries: list[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, dict[int, float]]:
    """Task failure rate per (delay_target, delay_ms) aggregated from record outcomes.

    Only completed runs (success/failure) count; works on summaries alone, no
    chunk files needed. The "__none__" baseline appears under delay 0.
    """
    counts: dict[tuple[str, int], list[int]] = collections.defaultdict(lambda: [0, 0])
    for _path, summary in summaries:
        for record in summary["records"]:
            if record.get("outcome") not in ("success", "failure"):
                continue
            key = (record["delay_target"], int(record["delay_ms_requested"]))
            counts[key][0] += int(bool(record["success"]))
            counts[key][1] += 1
    rates: dict[str, dict[int, float]] = {}
    for (target, delay_ms), (successes, completed) in counts.items():
        rates.setdefault(target, {})[delay_ms] = 1.0 - successes / completed
    return rates


# ---------------------------------------------------------------------------
# Aggregation per (delay_target, delay_ms)
# ---------------------------------------------------------------------------


def analyze_summaries(summaries: list[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, Any]:
    """Group runs by target/delay and compute spectra, gap/match/floor, and decompositions."""
    runs: list[dict[str, Any]] = []
    control_hz = 20.0
    replan_steps = 1
    for path, summary in summaries:
        runs.extend(load_runs(path, summary))
        config = summary.get("config", {})
        control_hz = float(config.get("control_hz", control_hz))
        replan_steps = int(config.get("replan_steps", replan_steps))
    if not runs:
        print(
            "warning: no chunk files found; only the failure-rate view (from summary records) will be produced",
            file=sys.stderr,
        )
        return {
            "targets": {},
            "floor": None,
            "control_hz": control_hz,
            "replan_steps": replan_steps,
            "failure_rates": failure_rates(summaries),
        }

    baseline_runs = [run for run in runs if run["delay_target"] == BASELINE_TARGET]
    floor = noise_floor(baseline_runs)
    delayed_runs = [run for run in runs if run["delay_target"] != BASELINE_TARGET]
    max_shift = max((run["delay_steps"] for run in delayed_runs), default=0)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for run in delayed_runs:
        grouped[(run["delay_target"], run["delay_ms"])].append(run)

    targets: dict[str, Any] = {}
    for (target, delay_ms), group in sorted(grouped.items()):
        delay_steps = group[0]["delay_steps"]
        spectrum = mean_spectrum(accumulate_spectrum(group, max_shift))
        entry = targets.setdefault(
            target,
            {"floor": floor, "control_hz": control_hz, "replan_steps": replan_steps, "max_shift": max_shift, "levels": {}},
        )
        entry["levels"][delay_ms] = {
            "delay_steps": delay_steps,
            "spectrum": spectrum,
            "gap": spectrum.get(0),
            "match": spectrum.get(delay_steps),
            "chunk_step_gap": chunk_step_curve(group, shift=0),
            "chunk_step_match": chunk_step_curve(group, shift=delay_steps),
            "action_dims": action_dim_distances(group, executed_steps=replan_steps),
        }
    return {
        "targets": targets,
        "floor": floor,
        "control_hz": control_hz,
        "replan_steps": replan_steps,
        "failure_rates": failure_rates(summaries),
    }


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------


def print_tables(results: dict[str, Any]) -> None:
    floor = results["floor"]
    for target, entry in results["targets"].items():
        label = TARGET_LABELS.get(target, target)
        print(f"\n{label} — gap (s=0) vs. match (s=true delay) vs. noise floor")
        print(f"  {'delay_ms':>8} | {'gap':>8} | {'match':>8} | {'floor':>8} | {'match/gap':>9}")
        for delay_ms in sorted(entry["levels"]):
            level = entry["levels"][delay_ms]
            gap = level["gap"]
            match = level["match"]
            ratio = match / gap if gap else None
            print(
                f"  {delay_ms:>8} | {_fmt(gap)} | {_fmt(match)} | {_fmt(floor)} | "
                + (f"{ratio:>9.3f}" if ratio is not None else f"{'-':>9}")
            )

    rates = results.get("failure_rates") or {}
    if rates:
        print("\nTask failure rate by delay target")
        delays = sorted({d for target_rates in rates.values() for d in target_rates})
        header = ["target"] + [f"{d}ms" for d in delays]
        print("  " + " | ".join(f"{h:>30}" if i == 0 else f"{h:>7}" for i, h in enumerate(header)))
        for target in _ordered_targets(rates):
            label = "baseline (no delay)" if target == BASELINE_TARGET else TARGET_LABELS.get(target, target)
            cells = []
            for delay_ms in delays:
                rate = rates[target].get(delay_ms)
                cells.append(f"{rate:>7.2f}" if rate is not None else f"{'-':>7}")
            print("  " + f"{label:>30}" + " | " + " | ".join(cells))


def _ordered_targets(targets: Any) -> list[str]:
    """Targets in canonical order (camera keys, state, __all__, baseline last)."""
    order = [*TARGET_LABELS, BASELINE_TARGET]
    return sorted(targets, key=lambda t: (order.index(t) if t in order else len(order), t))


def _fmt(value: Optional[float]) -> str:
    return f"{value:>8.4f}" if value is not None else f"{'-':>8}"


# ---------------------------------------------------------------------------
# Plotting (matplotlib imported lazily)
# ---------------------------------------------------------------------------


def plot_results(results: dict[str, Any], output_dir: pathlib.Path, show: bool) -> list[pathlib.Path]:
    import matplotlib  # noqa: PLC0415

    if not show:
        matplotlib.use("Agg")  # headless servers have no usable interactive backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if results["targets"]:
        written.append(_plot_spectrum(plt, results, output_dir / "shift_match_spectrum.png", show))
        written.append(_plot_gap(plt, results, output_dir / "shift_match_gap.png", show))
        written.append(_plot_chunk_step(plt, results, output_dir / "shift_match_chunk_step.png", show))
        written.append(_plot_action_dims(plt, results, output_dir / "shift_match_action_dims.png", show))
    written.append(_plot_failure_rate(plt, results, output_dir / "shift_match_failure_rate.png", show))
    return [path for path in written if path is not None]


def _plot_failure_rate(
    plt, results: dict[str, Any], output: pathlib.Path, show: bool
) -> Optional[pathlib.Path]:
    rates = results.get("failure_rates") or {}
    if not rates:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = rates.get(BASELINE_TARGET, {})
    if baseline:
        rate = sum(baseline.values()) / len(baseline)
        ax.axhline(rate, color="gray", linestyle="--", alpha=0.7, label=f"baseline (no delay): {rate:.2f}")
    for target in _ordered_targets(rates):
        if target == BASELINE_TARGET:
            continue
        delays = sorted(rates[target])
        ax.plot(
            delays,
            [rates[target][d] for d in delays],
            marker="o",
            label=TARGET_LABELS.get(target, target),
        )
    ax.set_title("Task failure rate vs. delay, by delay target")
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel("failure rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _target_subplots(plt, results: dict[str, Any], suptitle: str):
    targets = results["targets"]
    fig, axes = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 4.5), squeeze=False)
    fig.suptitle(suptitle)
    return fig, dict(zip(targets, axes[0]))


def _plot_spectrum(plt, results: dict[str, Any], output: pathlib.Path, show: bool) -> pathlib.Path:
    fig, axes = _target_subplots(plt, results, "Shift spectrum: delayed chunk vs. shadow chunk at t-shift")
    for target, ax in axes.items():
        entry = results["targets"][target]
        control_hz = entry["control_hz"]
        for delay_ms in sorted(entry["levels"]):
            level = entry["levels"][delay_ms]
            spectrum = level["spectrum"]
            shifts = sorted(spectrum)
            x_ms = [shift * 1000.0 / control_hz for shift in shifts]
            ax.plot(x_ms, [spectrum[shift] for shift in shifts], marker="o", label=f"{delay_ms} ms")
            ax.axvline(delay_ms, color=ax.lines[-1].get_color(), linestyle="--", alpha=0.5)
        if entry["floor"] is not None:
            ax.axhline(entry["floor"], color="gray", linestyle="--", alpha=0.7, label="noise floor")
        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_xlabel("shift (ms)")
        ax.set_ylabel("mean chunk L2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _plot_gap(plt, results: dict[str, Any], output: pathlib.Path, show: bool) -> pathlib.Path:
    fig, axes = _target_subplots(plt, results, "Gap (s=0) vs. match (s=true delay) vs. noise floor")
    width = 0.35
    for target, ax in axes.items():
        entry = results["targets"][target]
        delays = sorted(entry["levels"])
        gaps = [entry["levels"][d]["gap"] or 0.0 for d in delays]
        matches = [entry["levels"][d]["match"] or 0.0 for d in delays]
        x = np.arange(len(delays))
        ax.bar(x - width / 2, gaps, width, label="gap (s=0)")
        ax.bar(x + width / 2, matches, width, label="match (s=delay)")
        if entry["floor"] is not None:
            ax.axhline(entry["floor"], color="gray", linestyle="--", alpha=0.7, label="noise floor")
        ax.set_xticks(x, [f"{d}" for d in delays])
        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_xlabel("delay (ms)")
        ax.set_ylabel("mean chunk L2")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _plot_chunk_step(plt, results: dict[str, Any], output: pathlib.Path, show: bool) -> pathlib.Path:
    fig, axes = _target_subplots(plt, results, "Per-chunk-step L2 (solid: s=0 gap, dashed: s=true delay match)")
    for target, ax in axes.items():
        entry = results["targets"][target]
        replan_steps = entry["replan_steps"]
        for delay_ms in sorted(entry["levels"]):
            level = entry["levels"][delay_ms]
            gap_curve = level["chunk_step_gap"]
            match_curve = level["chunk_step_match"]
            if gap_curve is not None:
                (line,) = ax.plot(np.arange(len(gap_curve)), gap_curve, label=f"{delay_ms} ms gap")
                if match_curve is not None:
                    ax.plot(
                        np.arange(len(match_curve)),
                        match_curve,
                        linestyle="--",
                        color=line.get_color(),
                        label=f"{delay_ms} ms match",
                    )
        ax.axvspan(-0.5, replan_steps - 0.5, color="orange", alpha=0.15, label="executed open-loop")
        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_xlabel("chunk step")
        ax.set_ylabel("mean L2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _plot_action_dims(plt, results: dict[str, Any], output: pathlib.Path, show: bool) -> pathlib.Path:
    fig, axes = _target_subplots(plt, results, "Executed-portion L2 per action-dim group (s=0)")
    group_names = list(ACTION_DIM_GROUPS)
    for target, ax in axes.items():
        entry = results["targets"][target]
        delays = sorted(entry["levels"])
        width = 0.8 / max(len(group_names), 1)
        x = np.arange(len(delays))
        for index, name in enumerate(group_names):
            values = [entry["levels"][d]["action_dims"][name] or 0.0 for d in delays]
            ax.bar(x + (index - (len(group_names) - 1) / 2) * width, values, width, label=name)
        ax.set_xticks(x, [f"{d}" for d in delays])
        ax.set_title(TARGET_LABELS.get(target, target))
        ax.set_xlabel("delay (ms)")
        ax.set_ylabel("mean L2 (executed chunk steps)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        "-o",
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Directory for the output PNGs (default: current working directory)",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows")
    args = parser.parse_args()

    summary_paths = find_summaries(args.paths)
    summaries = []
    for path in summary_paths:
        summary = load_summary(path)
        if summary is not None:
            summaries.append((path, summary))
            print(f"loaded {summary['experiment']:<28} <- {path}")
    if not summaries:
        raise SystemExit(f"No {EXPERIMENT_NAME!r} summaries found.")

    results = analyze_summaries(summaries)
    print_tables(results)
    plot_results(results, args.output_dir, args.show)


if __name__ == "__main__":
    main()
