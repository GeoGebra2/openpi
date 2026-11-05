"""Analyze gripper-triggered camera delay, success rates and trajectories."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

try:
    from examples.libero import analyze_trajectory_delay as _base
    from examples.libero import run_gripper_delay as _run
except ImportError:
    import analyze_trajectory_delay as _base
    import run_gripper_delay as _run


LABELS = {"clean": "Always current", "full_delay": "Always delayed", "closed_gripper_delay": "Delay while closed"}


def wilson(successes, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count ** 2)) / denominator
    return max(0., centre - radius), min(1., centre + radius)


def validate_trace(trace, record, summary):
    """Replay the gate from raw joints and check inputs and executed plans."""
    config = summary["config"]
    gate = _run.GripperGate(config["close_width_mm"], config["open_width_mm"])
    computed = [gate.update(qpos) for qpos in trace["gripper_qpos"]]
    widths = np.asarray([row[0] for row in computed])
    closed = np.asarray([row[1] for row in computed], dtype=bool)
    changed = np.asarray([row[2] for row in computed], dtype=bool)
    if trace["gripper_width_mm"].shape != widths.shape or not np.allclose(trace["gripper_width_mm"], widths, rtol=0, atol=1e-8):
        raise ValueError("Recorded gripper widths disagree with raw joint positions")
    if not np.array_equal(trace["gate_closed"], closed) or not np.array_equal(trace["gate_changed"], changed):
        raise ValueError("Recorded gate disagrees with measured hysteresis")
    n, horizon = record["steps"], config["replan_steps"]
    expected, last = [], -horizon
    for step in range(n):
        if step - last >= horizon or changed[step]:
            expected.append(step)
            last = step
    queries = trace["query_steps"]
    if not np.array_equal(queries, expected):
        raise ValueError("Query schedule did not replan at a measured gripper transition")
    ages = np.asarray([_run.effective_delay(record["condition"], state, summary["delay_steps"])
                       for state in closed[:n]], dtype=np.int32)
    if not np.array_equal(trace["query_delay_steps"], ages[queries]):
        raise ValueError("Query delay does not match measured gripper state")
    if not np.array_equal(trace["action_delay_steps"], ages):
        raise ValueError("An old action chunk crossed a gripper-triggered delay switch")
    indices = np.searchsorted(queries, np.arange(n), side="right") - 1
    if not np.array_equal(trace["action_query_steps"], queries[indices]):
        raise ValueError("Action query provenance is invalid")
    sources = np.repeat(queries[:, None], 3, axis=1)
    sources[:, _run._shift.CAMERA_KEYS.index(config["delay_target"])] -= ages[queries]
    if not np.array_equal(trace["input_source_steps"], sources):
        raise ValueError("Camera or self-state source timestamps are invalid")
    if int(np.count_nonzero(ages)) != record["n_delayed_actions"]:
        raise ValueError("Delayed exposure count disagrees with trace")
    if int(np.count_nonzero(closed[:n])) != record["n_closed_actions"]:
        raise ValueError("Closed-gripper exposure count disagrees with trace")


def _success_plot(rows, directory, title):
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for index, row in enumerate(rows):
        p = row["success_rate"]
        if p is None:
            axis.text(index, 5, "No completed\nepisodes", ha="center")
            continue
        axis.bar(index, p * 100, color=plt.get_cmap("tab10")(index))
        low, high = row["wilson_low"] * 100, row["wilson_high"] * 100
        axis.vlines(index, low, high, color="black")
        axis.hlines([low, high], index - .05, index + .05, color="black")
        axis.text(index, high + 3, f"{row['successes']}/{row['completed']} ({p:.0%})", ha="center")
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([LABELS[row["condition"]] for row in rows])
    axis.set(ylim=(0, 115), ylabel="Task success rate (%)", title=title + "\n95% Wilson intervals")
    axis.grid(axis="y", alpha=.2)
    fig.text(.5, .02, "Errors/interrupted excluded. See CSV for runs that never reached the closed threshold.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(directory / "success_rate.png", dpi=160)
    plt.close(fig)


def _trigger_plot(record, trace, config, directory):
    from matplotlib import pyplot as plt
    t = trace["time_s"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, trace["gripper_width_mm"], label="Measured joint-derived opening")
    axes[0].axhline(config["close_width_mm"], color="red", linestyle="--", label="Close threshold")
    axes[0].axhline(config["open_width_mm"], color="green", linestyle="--", label="Reopen threshold")
    top = max(float(trace["gripper_width_mm"].max()), config["open_width_mm"]) * 1.1 + 1
    # Each measured state applies to [t[i], t[i+1]); fill_between(where=...)
    # would drop the final closed interval just before reopening.
    active = trace["gate_closed"][:record["steps"]].astype(np.int8)
    edges = np.diff(np.r_[0, active, 0])
    for index, (start, end) in enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
        axes[0].axvspan(t[start], t[end], color="orange", alpha=.15,
                        label="Closed gate" if index == 0 else None)
    axes[0].set(ylabel="Opening proxy (mm)", ylim=(0, top))
    if record["steps"]:
        delays = trace["action_delay_steps"] / record["control_hz"] * 1000
        axes[1].step(t, np.r_[delays, delays[-1]], where="post", label="Delay used by executed actions")
        queries = trace["query_steps"]
        axes[1].scatter(queries / record["control_hz"], trace["query_delay_steps"] / record["control_hz"] * 1000,
                        s=12, color="black", label="Policy query")
    for event_step in np.flatnonzero(trace["gate_changed"][:record["steps"]]):
        for axis in axes:
            axis.axvline(t[event_step], color="gray", linestyle=":", alpha=.6)
    axes[1].set(xlabel="Simulator time after warmup (s)", ylabel="Camera delay (ms)",
                ylim=(-30, config["delay_ms"] * 1.2 + 1))
    for axis in axes:
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=.2)
    fig.suptitle(f"{record['condition']} | repeat {record['repeat_index']} | {record['outcome']}")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(directory / f"gripper_trigger_{record['condition']}_{record['repeat_index']:03d}.png", dpi=160)
    plt.close(fig)


def analyze(path, *, episode_plot_limit=12):
    if episode_plot_limit < 0:
        raise ValueError("episode_plot_limit must be nonnegative")
    path = pathlib.Path(path)
    summary_path = path / "summary.json" if path.is_dir() else path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != _run.EXPERIMENT or summary.get("schema_version") != _run.SCHEMA_VERSION:
        raise ValueError("Requires a gripper-delay summary.json")
    args = _run.Args(**summary["config"])
    delay_steps, _ = _run.validate_args(args)
    if summary["delay_steps"] != delay_steps or [row["name"] for row in summary["conditions"]] != list(_run.CONDITIONS):
        raise ValueError("Summary conditions/delay disagree with config")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    directory = summary_path.parent / "gripper_delay_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    totals = _run.aggregate(summary["records"], args.repeats_per_condition)
    for row in totals:
        row["wilson_low"], row["wilson_high"] = wilson(row["successes"], row["completed"])
    colours = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(_run.CONDITIONS)}
    entries, lookup, excluded, state_rows = [], {}, [], []
    plotted = collections.Counter()
    for record in summary["records"]:
        if record["outcome"] not in ("success", "failure"):
            excluded.append({"condition": record["condition"], "repeat_index": record["repeat_index"],
                             "outcome": record["outcome"], "error": record.get("error")})
            continue
        trace = _base.load_trace(summary_path.parent, record)
        validate_trace(trace, record, summary)
        entries.append((record, trace))
        lookup[(record["condition"], record["repeat_index"])] = (record, trace)
        query_set = set(trace["query_steps"].tolist())
        for step in range(record["steps"] + 1):
            acted = step < record["steps"]
            state_rows.append({
                "condition": record["condition"], "repeat_index": record["repeat_index"],
                "step": step, "time_s": float(trace["time_s"][step]),
                "finger_left_qpos": float(trace["gripper_qpos"][step, 0]),
                "finger_right_qpos": float(trace["gripper_qpos"][step, 1]),
                "width_mm": float(trace["gripper_width_mm"][step]),
                "gate_closed": bool(trace["gate_closed"][step]), "gate_changed": bool(trace["gate_changed"][step]),
                "query": step in query_set, "action_executed": acted,
                "action_delay_ms": int(trace["action_delay_steps"][step]) / record["control_hz"] * 1000 if acted else None,
                "action_query_step": int(trace["action_query_steps"][step]) if acted else None,
            })
        if plotted[record["condition"]] < episode_plot_limit:
            _trigger_plot(record, trace, summary["config"], directory)
            plotted[record["condition"]] += 1
    paired, missing_pairs = [], []
    for record, trace in entries:
        if record["condition"] == "clean":
            continue
        reference = lookup.get(("clean", record["repeat_index"]))
        if reference is None:
            missing_pairs.append({"condition": record["condition"], "repeat_index": record["repeat_index"]})
            continue
        fields, _ = _base.paired_metrics(reference[0], record, reference[1], trace)
        paired.append(fields)
    title = (f"{args.task_suite_name} task {args.task_id} | {args.delay_target} | {args.delay_ms}ms\n"
             f"close <= {args.close_width_mm:g}mm, reopen >= {args.open_width_mm:g}mm")
    _success_plot(totals, directory, title)
    if entries:
        _base.overview_plots(entries, colours, directory, title)
    _run.write_csv(directory / "success_rates.csv", totals)
    _run.write_csv(directory / "gripper_state_by_step.csv", state_rows)
    _run.write_csv(directory / "paired_trajectory_metrics.csv", paired)
    report = {
        "experiment": _run.EXPERIMENT, "source_summary": str(summary_path), "protocol": summary["protocol"],
        "conditions": totals, "excluded_runs": excluded, "missing_clean_pairs": missing_pairs,
        "interpretation": [
            "Closure is threshold-defined from current measured finger positions, not proof of successful grasp.",
            "Each rollout uses its own gripper state; trajectories and trigger times can diverge after intervention.",
            "At replan_steps=1, query times match. Larger chunks permit different event-triggered query times.",
            "Reopening restores current camera input; it does not undo earlier action consequences.",
            "Runs with no closed-gate actions remain in overall success rates and are counted separately.",
            "Rates describe the configured task and initial state over paired policy noise seeds.",
        ],
    }
    result = directory / "gripper_delay_analysis.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--episode-plot-limit", type=int, default=12)
    print(analyze(**vars(parser.parse_args())))
