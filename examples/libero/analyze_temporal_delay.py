"""Replot temporal-delay success rates, schedules and measured trajectories."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

try:
    from examples.libero import analyze_trajectory_delay as _base
    from examples.libero import run_temporal_delay as _run
except ImportError:
    import analyze_trajectory_delay as _base
    import run_temporal_delay as _run


LABELS = {"clean": "Always current", "full_delay": "Always delayed",
          "early_delay": "Delay before switch", "late_delay": "Delay after switch"}


def wilson(successes, completed):
    if not completed:
        return None, None
    z = 1.959963984540054
    p = successes / completed
    divisor = 1 + z * z / completed
    centre = (p + z * z / (2 * completed)) / divisor
    radius = z * np.sqrt(p * (1 - p) / completed + z * z / (4 * completed ** 2)) / divisor
    return max(0., centre - radius), min(1., centre + radius)


def validate_schedule(trace, record, summary):
    """Check actual query ages, action provenance and boundary replanning."""
    queries = trace["query_steps"]
    boundary = summary["switch_step"]
    horizon = summary["config"]["replan_steps"]
    n = record["steps"]
    expected_queries = list(range(0, min(n, boundary), horizon)) + list(range(boundary, n, horizon))
    if not np.array_equal(queries, expected_queries):
        raise ValueError(f"Query schedule does not replan at boundary: {record['trace_path']}")
    ages = np.asarray([_run.delay_at_step(int(step), condition=record["condition"],
                                        switch_step=boundary, delay_steps=summary["delay_steps"])
                       for step in queries], dtype=np.int32)
    if not np.array_equal(trace["query_delay_steps"], ages):
        raise ValueError("Saved query delays do not match configured time windows")
    sources = np.repeat(queries[:, None], 3, axis=1)
    column = _run._shift.CAMERA_KEYS.index(summary["config"]["delay_target"])
    sources[:, column] -= ages
    if not np.array_equal(trace["input_source_steps"], sources):
        raise ValueError("Camera/state source steps do not match configured delay")
    indices = np.searchsorted(queries, np.arange(n), side="right") - 1
    if not np.array_equal(trace["action_query_steps"], queries[indices]):
        raise ValueError("Actions from an earlier plan were executed across the switch")
    if not np.array_equal(trace["action_delay_steps"], ages[indices]):
        raise ValueError("Action observation ages do not match query provenance")


def _success_plot(rows, directory, title):
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, row in enumerate(rows):
        rate = row["success_rate"]
        if rate is None:
            axis.text(index, 0.05, "No completed\nepisodes", ha="center", fontsize=10)
            continue
        lo, hi = row["wilson_low"], row["wilson_high"]
        axis.bar(index, rate * 100, color=plt.get_cmap("tab10")(index),
                 yerr=np.array([[max(0, rate - lo)], [max(0, hi - rate)]]) * 100, capsize=4)
        axis.text(index, min(108, hi * 100 + 3), f"{row['successes']}/{row['completed']} ({rate:.0%})",
                  ha="center", fontsize=10)
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([LABELS[row["condition"]] for row in rows], fontsize=9)
    axis.set(ylim=(0, 116), ylabel="Task success rate (%)", title=title + "\n95% Wilson intervals")
    axis.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.02, "Denominator: success + failure. Errors/interrupted/missing listed in success_rates.csv.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(directory / "success_rate.png", dpi=160)
    plt.close(fig)


def _schedule_plot(summary, directory):
    from matplotlib import pyplot as plt
    hz = summary["config"]["control_hz"]
    switch = summary["switch_step"] / hz
    end = summary["max_steps"] / hz
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for axis, name in zip(axes, _run.CONDITIONS):
        ages = [_run.delay_at_step(step, condition=name, switch_step=summary["switch_step"],
                                  delay_steps=summary["delay_steps"]) / hz * 1000
                for step in (0, summary["switch_step"], summary["max_steps"])]
        axis.step([0, switch, end], ages, where="post", linewidth=2)
        axis.axvline(switch, color="black", linestyle="--", linewidth=1)
        axis.set(ylabel="Delay (ms)", ylim=(-30, max(ages + [summary["delay_ms_effective"]]) * 1.2 + 1))
        axis.set_title(LABELS[name], fontsize=10, loc="left")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Simulator time after warmup (s)")
    axes[0].text(switch, axes[0].get_ylim()[1] * .75, f"  switch={switch:g}s")
    fig.suptitle("Planned camera-age schedules (episodes stop on success)")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(directory / "delay_schedule.png", dpi=160)
    plt.close(fig)


def analyze(path):
    path = pathlib.Path(path)
    summary_path = path / "summary.json" if path.is_dir() else path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != _run.EXPERIMENT or summary.get("schema_version") != _run.SCHEMA_VERSION:
        raise ValueError("Requires a temporal-delay summary.json")
    if [item["name"] for item in summary["conditions"]] != list(_run.CONDITIONS):
        raise ValueError("Unexpected conditions")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = summary_path.parent / "temporal_delay_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    config, records = summary["config"], summary["records"]
    totals = _run.aggregate(records, config["repeats_per_condition"])
    for row in totals:
        row["wilson_low"], row["wilson_high"] = wilson(row["successes"], row["completed"])
    colours = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(_run.CONDITIONS)}
    entries, lookup, excluded = [], {}, []
    for record in records:
        if record["outcome"] not in ("success", "failure"):
            excluded.append({"condition": record["condition"], "repeat_index": record["repeat_index"],
                             "outcome": record["outcome"], "error": record.get("error")})
            continue
        trace = _base.load_trace(summary_path.parent, record)
        validate_schedule(trace, record, summary)
        entries.append((record, trace))
        lookup[(record["condition"], record["repeat_index"])] = (record, trace)
    paired, missing_pairs = [], []
    for record, trace in entries:
        if record["condition"] == "clean":
            continue
        reference = lookup.get(("clean", record["repeat_index"]))
        if reference is None:
            missing_pairs.append({"condition": record["condition"], "repeat_index": record["repeat_index"]})
            continue
        clean_record, clean_trace = reference
        fields, _ = _base.paired_metrics(clean_record, record, clean_trace, trace)
        paired.append(fields)
    # Paired rates count only repeats completed in both groups.
    comparisons = []
    for name in _run.CONDITIONS[1:]:
        pairs = [row for row in paired if row["condition"] == name]
        wins = sum(row["clean_outcome"] == "failure" and row["delayed_outcome"] == "success" for row in pairs)
        losses = sum(row["clean_outcome"] == "success" and row["delayed_outcome"] == "failure" for row in pairs)
        comparisons.append({"condition": name, "completed_pairs": len(pairs),
                            "clean_success_condition_failure": losses,
                            "clean_failure_condition_success": wins,
                            "paired_success_rate_difference": (wins - losses) / len(pairs) if pairs else None})
    title = (f"{config['task_suite_name']} task {config['task_id']} | {config['delay_target']} | "
             f"delay={config['delay_ms']}ms | switch={summary['switch_time_s_effective']:g}s")
    _success_plot(totals, directory, title)
    _schedule_plot(summary, directory)
    if entries:
        _base.overview_plots(entries, colours, directory, title)
    _run.write_csv(directory / "success_rates.csv", totals)
    _run.write_csv(directory / "paired_success_vs_clean.csv", comparisons)
    _run.write_csv(directory / "paired_trajectory_metrics.csv", paired)
    report = {
        "experiment": _run.EXPERIMENT, "source_summary": str(summary_path),
        "protocol": summary["protocol"], "conditions": totals, "paired_success": comparisons,
        "excluded_runs": excluded, "missing_clean_pairs": missing_pairs,
        "interpretation": [
            "early_delay tests the effect of initial delayed vision with subsequent current vision; the environment is not reset at the switch.",
            "late_delay tests delayed vision after an initially current-vision interval.",
            "A fixed time boundary is not a labelled approach/alignment/grasp stage.",
            "Early and late windows can have different durations and numbers of delayed queries; inspect per-episode exposure counts.",
            "Success before the switch remains in overall success rate; successes_before_switch and post-switch exposure counts identify these episodes.",
            "Rates describe the configured task and fixed initial state over paired noise seeds, not the entire benchmark.",
        ],
    }
    result = directory / "temporal_delay_analysis.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    print(analyze(parser.parse_args().path))
