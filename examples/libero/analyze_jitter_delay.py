"""Validate and plot random-increment camera ages and LIBERO outcomes."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

try:
    from examples.libero import analyze_trajectory_delay as _base
    from examples.libero import run_jitter_delay as _run
except ImportError:
    import analyze_trajectory_delay as _base
    import run_jitter_delay as _run


LABELS = {
    "clean": "Current camera",
    "fixed_delay": "Fixed delay",
    "increment_delay": "Random-increment delay",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def wilson(successes, total):
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def write_schedule_plot(args, directory, rows_by_condition, *, filename="delay_schedule.png", title_suffix=""):
    """Plot requested and actual ages on the actual query index, not video time."""
    plt = _pyplot()
    names = list(rows_by_condition)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.6 * len(names)), sharex=True, squeeze=False)
    for axis, name in zip(axes[:, 0], names):
        rows = rows_by_condition[name]
        query = np.asarray([int(row["query_index"]) for row in rows]) + 1
        uncapped = np.asarray([float(row["uncapped_ms"]) for row in rows])
        requested = np.asarray([float(row["requested_ms"]) for row in rows])
        effective = np.asarray([float(row["effective_ms"]) for row in rows])
        axis.plot(query, uncapped, linestyle="--", color="0.45", linewidth=1.3, label="Before upper cap")
        axis.plot(query, requested, color="#d28b26", linewidth=1, alpha=0.85, label="Requested after cap")
        axis.step(query, effective, where="post", color="#267e99", linewidth=1.7, label="Actual historical frame age")
        for row in rows:
            if int(row["query_index"]) > 0 and int(row["query_in_cycle"]) == 0 and name == "increment_delay":
                axis.axvline(int(row["query_index"]) + 1, color="0.6", linestyle=":", alpha=0.5)
        axis.set(title=LABELS[name], ylabel="Camera age (ms)",
                 ylim=(-15, max(args.delay_upper_ms, float(uncapped.max())) + 45))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("Policy camera-read count (1-based; dotted lines mark resets)")
    fig.suptitle(
        f"{args.delay_target} | {args.delay_lower_ms:g}-{args.delay_upper_ms:g} ms | "
        f"increment Uniform({args.increment_min_ms:g}, {args.increment_max_ms:g}) ms per query"
        + title_suffix
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


def load_delay_rows(directory, record):
    if not record.get("delay_schedule_path"):
        return []
    with (directory / record["delay_schedule_path"]).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_schedule(args, record, trace, rows):
    """Check actual queries, CSV draws, frame selection and action provenance."""
    queries = trace["query_steps"]
    n = record["steps"]
    expected = np.arange(0, n, args.replan_steps)
    if not np.array_equal(queries, expected) or len(rows) != len(queries):
        raise ValueError(f"Missing or miscounted policy queries: {record['trace_path']}")
    ages = []
    for query_index, (row, step) in enumerate(zip(rows, queries)):
        sample = _run.sample_delay(args, record["condition"], record["repeat_index"], query_index)
        if row["condition"] != record["condition"] or int(row["repeat_index"]) != record["repeat_index"]:
            raise ValueError("Delay CSV identifies a different condition/repeat")
        if int(row["step"]) != step or not np.isclose(float(row["time_s"]), step / args.control_hz):
            raise ValueError("Delay CSV query step/time mismatch")
        for key in ("query_index", "cycle_index", "query_in_cycle", "delay_steps"):
            if int(row[key]) != getattr(sample, key):
                raise ValueError(f"Delay CSV {key} does not match reproducible schedule")
        for key in ("increment_ms", "uncapped_ms", "requested_ms", "effective_ms"):
            if not np.isclose(float(row[key]), getattr(sample, key), rtol=0, atol=1e-7):
                raise ValueError(f"Delay CSV {key} does not match reproducible schedule")
        if row["reached_upper"].lower() != str(sample.reached_upper).lower():
            raise ValueError("Delay CSV reached_upper flag is inconsistent")
        ages.append(sample.delay_steps)
    ages = np.asarray(ages, dtype=np.int32)
    if not np.array_equal(trace["query_delay_steps"], ages):
        raise ValueError("Actual inference camera ages differ from the increment schedule")
    expected_sources = np.repeat(queries[:, None], 3, axis=1)
    column = _run._shift.CAMERA_KEYS.index(args.delay_target)
    expected_sources[:, column] -= ages
    if not np.array_equal(trace["input_source_steps"], expected_sources):
        raise ValueError("The selected camera/state source timestamps are incorrect")
    csv_sources = np.asarray([
        [int(row[key]) for key in ("agentview_source_step", "wrist_source_step", "state_source_step")]
        for row in rows
    ], dtype=np.int32).reshape(-1, 3)
    if not np.array_equal(csv_sources, expected_sources):
        raise ValueError("Delay CSV source timestamps differ from actual trace")
    if n:
        query_indices = np.searchsorted(queries, np.arange(n), side="right") - 1
        if not np.array_equal(trace["action_query_steps"], queries[query_indices]):
            raise ValueError("Executed actions do not come from the saved query plan")
        if not np.array_equal(trace["action_delay_steps"], ages[query_indices]):
            raise ValueError("Action provenance refers to different image ages")


def _success_plot(totals, directory, title):
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(11, 6))
    for index, row in enumerate(totals):
        rate = row["success_rate"]
        if rate is None:
            axis.text(index, 20, "No completed\nruns", ha="center")
            continue
        low, high = wilson(row["successes"], row["completed"])
        axis.bar(index, rate * 100, color=plt.get_cmap("tab10")(index),
                 yerr=np.asarray([[max(0, rate - low)], [max(0, high - rate)]]) * 100, capsize=5)
        axis.text(index, high * 100 + 2, f"{row['successes']}/{row['completed']} ({rate:.1%})", ha="center")
    axis.set(ylim=(0, 115), ylabel="Task success rate (%)", title=title + "\n95% Wilson intervals")
    axis.set_xticks(range(len(totals)))
    axis.set_xticklabels([LABELS[row["condition"]] for row in totals])
    axis.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.02, "Denominator = success + failure; errors and interruptions are counted separately.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(directory / "success_rate.png", dpi=160)
    plt.close(fig)


def _delay_distribution(entries, rows_by_key, directory):
    plt = _pyplot()
    grouped = collections.defaultdict(list)
    for record, _ in entries:
        grouped[(record["condition"], record["outcome"])].extend(
            float(row["effective_ms"]) for row in rows_by_key[(record["condition"], record["repeat_index"])]
        )
    fig, axis = plt.subplots(figsize=(11, 5))
    for (name, outcome), ages in grouped.items():
        values, counts = np.unique(ages, return_counts=True)
        axis.plot(values, counts / counts.sum(), marker="o", linestyle="-" if outcome == "success" else "--",
                  label=f"{LABELS[name]} / {outcome} ({len(ages)} queries)")
    axis.set(xlabel="Actual camera age (ms)", ylabel="Fraction of policy queries",
             title="Realized query-age exposure; successful episodes can end earlier")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "delay_distribution.png", dpi=160)
    plt.close(fig)


def analyze(path, *, threshold_cm=2.0, consecutive=3, pair_limit=30, interactive=False):
    if not np.isfinite(threshold_cm) or threshold_cm <= 0 or consecutive < 1 or pair_limit < 0:
        raise ValueError("Invalid trajectory-analysis thresholds")
    plt = _pyplot()
    path = pathlib.Path(path)
    summary_path = path / "summary.json" if path.is_dir() else path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != _run.EXPERIMENT or summary.get("schema_version") != _run.SCHEMA_VERSION:
        raise ValueError("Requires a run_jitter_delay summary.json")
    # Older schema-v2 runs included plotting options in the runner config.
    config = dict(summary["config"])
    for key in ("skip_analysis", "interactive", "threshold_cm", "consecutive", "pair_limit"):
        config.pop(key, None)
    args = _run.Args(**config)
    _run.validate_args(args)
    names = _run.conditions_for(args)
    if [condition["name"] for condition in summary["conditions"]] != list(names):
        raise ValueError("Saved conditions do not match the configuration")
    if args.preview_only:
        raise ValueError("This is a schedule-only preview, not collected task results")
    directory = summary_path.parent / "increment_delay_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    records = summary["records"]
    totals = _run.aggregate(args, records)
    for row in totals:
        row["wilson_low"], row["wilson_high"] = wilson(row["successes"], row["completed"])
    entries, lookup, excluded, rows_by_key = [], {}, [], {}
    for record in records:
        key = (record["condition"], record["repeat_index"])
        if record["outcome"] not in ("success", "failure"):
            excluded.append({"condition": key[0], "repeat_index": key[1], "outcome": record["outcome"],
                             "error": record.get("error")})
            continue
        trace = _base.load_trace(summary_path.parent, record)
        rows = load_delay_rows(summary_path.parent, record)
        validate_schedule(args, record, trace, rows)
        entries.append((record, trace))
        lookup[key] = (record, trace)
        rows_by_key[key] = rows
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(names)}
    title = (f"{args.task_suite_name} task {args.task_id} | {args.delay_target}\n"
             f"bounds={args.delay_lower_ms:g}-{args.delay_upper_ms:g} ms | "
             f"increment Uniform({args.increment_min_ms:g}, {args.increment_max_ms:g}) ms/query")
    _success_plot(totals, directory, title)
    if entries:
        _base.overview_plots(entries, colors, directory, title)
        _delay_distribution(entries, rows_by_key, directory)
        if interactive:
            _base.interactive_plot(entries, colors, directory)
    by_repeat = collections.defaultdict(dict)
    for key, rows in rows_by_key.items():
        by_repeat[key[1]][key[0]] = rows
    for repeat, rows_by_condition in sorted(by_repeat.items()):
        # The schedule is a primary diagnostic even when pair_limit=0 disables
        # individual trajectory-comparison panels.
        if repeat >= max(1, pair_limit):
            continue
        ordered = {name: rows_by_condition[name] for name in names if name in rows_by_condition}
        write_schedule_plot(args, directory, ordered, filename=f"delay_schedule_repeat_{repeat:03d}.png",
                            title_suffix=f"\nActual query ages | repeat {repeat}; lines stop when the episode ends")
    paired, pair_fields, missing_pairs, plotted = [], [], [], collections.Counter()
    for record, trace in entries:
        if record["condition"] == "clean":
            continue
        reference = lookup.get(("clean", record["repeat_index"]))
        if reference is None:
            missing_pairs.append({"condition": record["condition"], "repeat_index": record["repeat_index"]})
            continue
        clean_record, clean_trace = reference
        fields, series = _base.paired_metrics(clean_record, record, clean_trace, trace, threshold_cm / 100, consecutive)
        paired.append((fields, series))
        pair_fields.append(fields)
        if plotted[record["condition"]] < pair_limit:
            _base.pair_plot(clean_record, record, clean_trace, trace, fields, series,
                            colors, directory, threshold_cm / 100)
            plotted[record["condition"]] += 1
    if paired:
        _base.separation_plot(paired, colors, directory)
    comparisons = []
    for condition in names:
        if condition == "clean":
            continue
        matches = [(lookup[("clean", repeat)][0], lookup[(condition, repeat)][0])
                   for repeat in range(args.repeats_per_condition)
                   if ("clean", repeat) in lookup and (condition, repeat) in lookup]
        losses = sum(a["success"] and not b["success"] for a, b in matches)
        wins = sum(not a["success"] and b["success"] for a, b in matches)
        comparisons.append({
            "reference": "clean", "condition": condition, "completed_pairs": len(matches),
            "reference_success_condition_failure": losses,
            "reference_failure_condition_success": wins,
            "paired_success_rate_difference": (wins - losses) / len(matches) if matches else None,
        })
    _run.write_csv(directory / "success_rates.csv", totals)
    _run.write_csv(directory / "paired_success.csv", comparisons)
    _run.write_csv(directory / "paired_trajectory_metrics.csv", pair_fields)
    all_rows = [row for rows in rows_by_key.values() for row in rows]
    _run.write_csv(directory / "delay_queries.csv", all_rows)
    report = {
        "experiment": _run.EXPERIMENT, "source_summary": str(summary_path),
        "conditions": totals, "paired_success": comparisons,
        "excluded_runs": excluded, "missing_clean_pairs": missing_pairs,
        "validated_completed_traces": len(entries),
        "interpretation": [
            "Delay advances per policy query, independent of video frames, simulation speed and inference wall time.",
            "Requested ages are clipped and rounded upward to available historical frames. Inspect the effective-age curves.",
            f"Each cycle starts at the lower bound; subsequent queries add Uniform({args.increment_min_ms:g}, {args.increment_max_ms:g}) ms until a query uses the capped upper age, then the next query resets.",
            "Clipping at the upper bound and frame quantization change the realized ages; inspect the per-query CSV and curves.",
            "Success/failure terminate independent closed-loop rollouts; exposure distributions are affected by early termination.",
            "Selected frame timestamps may repeat or move backward as random ages increase. This models per-query historical sampling, not a network packet queue.",
            "The measurements apply to this task and initial state over repeat seeds, not the entire benchmark.",
        ],
    }
    target = directory / "increment_delay_analysis.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, help="Run directory or summary.json")
    parser.add_argument("--threshold-cm", type=float, default=2.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--pair-limit", type=int, default=30)
    parser.add_argument("--interactive", action="store_true")
    print(analyze(**vars(parser.parse_args())))
