"""Compare clean images with a random-increment image-age ramp in LIBERO.

Only one selected camera is delayed. Its requested age starts at the lower
bound, increases by a random 150-210 ms on each subsequent policy query until
it reaches the upper bound, and resets on the following query. Simulator and
video frames do not advance the ramp. The simulator is never paused or slept.
The runner prints success rates and plots camera image ages per policy
observation. Trajectory plots are not generated; video is opt-in.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import json
import logging
import math
import pathlib
from typing import Optional

import numpy as np

try:
    from examples.libero import run_trajectory_delay as _trajectory
except ImportError:
    import run_trajectory_delay as _trajectory

_shift = _trajectory._shift
EXPERIMENT = "libero_increment_delay"
SCHEMA_VERSION = 2
CONDITIONS = ("clean", "fixed_delay", "increment_delay")


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None
    delay_target: str = "observation/wrist_image"
    delay_lower_ms: float = 200.0
    delay_upper_ms: float = 450.0
    increment_min_ms: float = 150.0
    increment_max_ms: float = 210.0
    increment_seed: Optional[int] = None
    fixed_delay_ms: Optional[float] = None
    conditions: str = "clean,increment_delay"
    repeats_per_condition: int = 30
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 1
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = False
    save_delay_plots: bool = True
    video_fps: int = 20
    object_names: str = ""
    output_dir: str = "rollout/libero_increment_delay"
    preview_only: bool = False
    preview_queries: int = 120


@dataclasses.dataclass(frozen=True)
class DelaySample:
    query_index: int
    cycle_index: int
    query_in_cycle: int
    increment_ms: float
    uncapped_ms: float
    requested_ms: float
    delay_steps: int
    effective_ms: float
    reached_upper: bool


def _split_csv(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("conditions must be a non-empty comma-separated list without duplicates")
    unknown = set(names) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions {sorted(unknown)}; choose from {CONDITIONS}")
    if "clean" not in names:
        raise ValueError("Include clean as a paired baseline")
    return names


def conditions_for(args: Args) -> tuple[str, ...]:
    return _split_csv(args.conditions)


def _frame_period_ms(control_hz: float) -> float:
    return 1000.0 / control_hz


def delay_to_steps(delay_ms: float, control_hz: float) -> int:
    """Round age upward so the selected historical frame is never fresher."""
    if not np.isfinite(delay_ms) or delay_ms < 0 or not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("delay_ms must be finite and nonnegative; control_hz must be positive")
    steps = delay_ms * control_hz / 1000.0
    return int(math.ceil(steps - 1e-10))


def validate_args(args: Args) -> tuple[int, int]:
    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unsupported task suite {args.task_suite_name!r}")
    if args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError(f"delay_target must select one camera from {_shift.CAMERA_KEYS}")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if args.increment_seed is not None and (type(args.increment_seed) is not int or args.increment_seed < 0):
        raise ValueError("increment_seed must be a nonnegative integer")
    if type(args.repeats_per_condition) is not int or args.repeats_per_condition < 1:
        raise ValueError("repeats_per_condition must be a positive integer")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.video_fps) < 1 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size, video rate or warmup")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not np.isfinite(args.delay_lower_ms) or not np.isfinite(args.delay_upper_ms):
        raise ValueError("Delay bounds must be finite")
    if args.delay_lower_ms < 0 or args.delay_upper_ms <= args.delay_lower_ms:
        raise ValueError("Require 0 <= delay_lower_ms < delay_upper_ms")
    if not np.isfinite(args.increment_min_ms) or not np.isfinite(args.increment_max_ms):
        raise ValueError("Increment bounds must be finite")
    if args.increment_min_ms <= 0 or args.increment_max_ms < args.increment_min_ms:
        raise ValueError("Require 0 < increment_min_ms <= increment_max_ms")
    if type(args.preview_queries) is not int or args.preview_queries < 1:
        raise ValueError("preview_queries must be a positive integer")
    upper_steps = delay_to_steps(args.delay_upper_ms, args.control_hz)
    effective_upper = upper_steps / args.control_hz * 1000.0
    if not np.isclose(effective_upper, args.delay_upper_ms, rtol=0, atol=1e-8):
        raise ValueError(
            f"delay_upper_ms={args.delay_upper_ms:g} must align with the "
            f"{_frame_period_ms(args.control_hz):g} ms frame grid; otherwise the maximum age exceeds the cap"
        )
    if args.fixed_delay_ms is not None and (
        not np.isfinite(args.fixed_delay_ms) or not 0 <= args.fixed_delay_ms <= args.delay_upper_ms
    ):
        raise ValueError("fixed_delay_ms must be finite and between zero and delay_upper_ms")
    if "fixed_delay" in conditions_for(args) and args.fixed_delay_ms is None:
        raise ValueError("Set --fixed-delay-ms when conditions includes fixed_delay")
    max_steps = args.max_steps if args.max_steps is not None else _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
    return upper_steps, max_steps


def sample_delay(args: Args, condition: str, repeat_index: int, query_index: int) -> DelaySample:
    """Reconstruct the cumulative random-increment age at an actual query."""
    if query_index < 0 or repeat_index < 0:
        raise ValueError("repeat_index and query_index must be nonnegative")
    conditions_for(args)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    if condition == "clean":
        cycle_index, query_in_cycle = 0, query_index
        increment_ms = uncapped_ms = requested_ms = 0.0
        reached_upper = False
    elif condition == "fixed_delay":
        if args.fixed_delay_ms is None:
            raise ValueError("fixed_delay requires fixed_delay_ms")
        cycle_index, query_in_cycle = 0, query_index
        increment_ms = 0.0
        uncapped_ms = requested_ms = args.fixed_delay_ms
        reached_upper = False
    else:
        # Reconstruct from the same dedicated RNG stream on every lookup. This
        # makes video/terminal lookups and out-of-order tests side-effect free.
        seed = args.seed if args.increment_seed is None else args.increment_seed
        rng = np.random.default_rng(np.random.SeedSequence([seed, repeat_index, 40697]))
        cycle_index = query_in_cycle = 0
        increment_ms = 0.0
        uncapped_ms = requested_ms = args.delay_lower_ms
        reached_upper = False
        for _ in range(query_index):
            if reached_upper:
                cycle_index += 1
                query_in_cycle = 0
                increment_ms = 0.0
                uncapped_ms = requested_ms = args.delay_lower_ms
                reached_upper = False
            else:
                increment_ms = float(rng.uniform(args.increment_min_ms, args.increment_max_ms))
                uncapped_ms = requested_ms + increment_ms
                requested_ms = min(args.delay_upper_ms, uncapped_ms)
                query_in_cycle += 1
                reached_upper = requested_ms >= args.delay_upper_ms
    steps = delay_to_steps(requested_ms, args.control_hz)
    upper_steps, _ = validate_args(args)
    if steps > upper_steps:
        raise ValueError("Quantized image age exceeds delay_upper_ms")
    return DelaySample(
        query_index=query_index,
        cycle_index=cycle_index,
        query_in_cycle=query_in_cycle,
        increment_ms=float(increment_ms),
        uncapped_ms=float(uncapped_ms),
        requested_ms=float(requested_ms),
        delay_steps=steps,
        effective_ms=steps / args.control_hz * 1000.0,
        reached_upper=bool(reached_upper),
    )


def delay_schedule(args: Args, condition: str, repeat_index: int):
    """Make a delay lookup keyed by replan query, safe to call in any order."""
    upper_steps, _ = validate_args(args)

    def schedule(step: int) -> int:
        if type(step) is not int or step < 0:
            raise ValueError("environment step must be a nonnegative integer")
        query_index = step // args.replan_steps
        sample = sample_delay(args, condition, repeat_index, query_index)
        if sample.delay_steps > upper_steps:
            raise ValueError("scheduled camera age exceeds allocated history")
        return sample.delay_steps

    return schedule


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def aggregate(args: Args, records: list[dict]) -> list[dict]:
    grouped = collections.defaultdict(list)
    seen = set()
    conditions = conditions_for(args)
    for record in records:
        key = (record["condition"], record["repeat_index"])
        if key in seen or key[0] not in conditions or not 0 <= key[1] < args.repeats_per_condition:
            raise ValueError(f"Invalid or duplicate condition/repeat {key}")
        if record["outcome"] not in ("success", "failure", "error", "interrupted"):
            raise ValueError(f"Unknown outcome {record['outcome']!r}")
        seen.add(key)
        grouped[key[0]].append(record)
    totals = []
    for condition in conditions:
        runs = grouped[condition]
        counts = collections.Counter(row["outcome"] for row in runs)
        completed = counts["success"] + counts["failure"]
        totals.append({
            "condition": condition,
            "runs": len(runs),
            "completed": completed,
            "missing_runs": args.repeats_per_condition - len(runs),
            "successes": counts["success"],
            "failures": counts["failure"],
            "errors": counts["error"],
            "interrupted": counts["interrupted"],
            "success_rate": counts["success"] / completed if completed else None,
        })
    return totals


def _sample_rows(args: Args, condition: str, repeat_index: int, query_count: int) -> list[dict]:
    rows = []
    for query_index in range(query_count):
        sample = sample_delay(args, condition, repeat_index, query_index)
        rows.append({"condition": condition, "repeat_index": repeat_index, **dataclasses.asdict(sample)})
    return rows


def save_query_schedule(directory: pathlib.Path, args: Args, record: dict) -> None:
    """Save cumulative increments and realized ages for actual policy queries."""
    with np.load(directory / record["trace_path"], allow_pickle=False) as archive:
        queries = archive["query_steps"].copy()
        ages = archive["query_delay_steps"].copy()
        sources = archive["input_source_steps"].copy()
    expected_queries = np.arange(0, record["steps"], args.replan_steps)
    if record["outcome"] in ("success", "failure") and not np.array_equal(queries, expected_queries):
        raise ValueError("Saved policy query schedule differs from query-count ramp schedule")
    condition, repeat_index = record["condition"], record["repeat_index"]
    samples = [sample_delay(args, condition, repeat_index, index) for index in range(len(queries))]
    if not np.array_equal(ages, [sample.delay_steps for sample in samples]):
        raise ValueError("Saved camera ages differ from generated query-count delay samples")
    rows = []
    for step, source, sample in zip(queries, sources, samples):
        rows.append({
            "condition": condition,
            "repeat_index": repeat_index,
            "step": int(step),
            "time_s": float(step / args.control_hz),
            **dataclasses.asdict(sample),
            "agentview_source_step": int(source[0]),
            "wrist_source_step": int(source[1]),
            "state_source_step": int(source[2]),
        })
    path = directory / f"{condition}_repeat_{repeat_index:03d}_delay.csv"
    write_csv(path, rows)
    record.update({
        "delay_schedule_path": path.name if rows else None,
        "query_index_start": 0 if samples else None,
        "query_index_end": len(samples) - 1 if samples else None,
        "requested_delay_ms_mean": float(np.mean([s.requested_ms for s in samples])) if samples else None,
        "effective_delay_ms_mean": float(np.mean([s.effective_ms for s in samples])) if samples else None,
        "effective_delay_ms_std": float(np.std([s.effective_ms for s in samples])) if samples else None,
        "effective_delay_ms_min": float(min(s.effective_ms for s in samples)) if samples else None,
        "effective_delay_ms_max": float(max(s.effective_ms for s in samples)) if samples else None,
        "increment_draw_mean_ms": (
            float(np.mean([s.increment_ms for s in samples if s.increment_ms > 0]))
            if any(s.increment_ms > 0 for s in samples) else None
        ),
        "n_upper_queries": int(sum(s.reached_upper for s in samples)),
        "n_resets": int(sum(s.query_in_cycle == 0 and s.query_index > 0 for s in samples))
        if condition == "increment_delay" else 0,
        "complete_ramp_cycles": int(sum(s.reached_upper for s in samples))
        if condition == "increment_delay" else 0,
    })


def save_summary(directory: pathlib.Path, args: Args, metadata: dict, records: list[dict]) -> None:
    upper_steps, max_steps = validate_args(args)
    conditions = conditions_for(args)
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "config": dataclasses.asdict(args),
        "server_metadata": metadata,
        "conditions": [{"name": name} for name in conditions],
        "delay_upper_steps": upper_steps,
        "delay_upper_ms_effective": upper_steps / args.control_hz * 1000,
        "max_steps": max_steps,
        "aggregate": aggregate(args, records),
        "records": records,
        "protocol": {
            "delay_target": args.delay_target,
            "unselected_camera_and_state": "always current",
            "increment": "first query of each cycle uses delay_lower_ms; each subsequent policy query adds an independent Uniform(increment_min_ms,increment_max_ms) draw to the prior requested age",
            "rng": "dedicated SeedSequence([increment_seed if set else seed, repeat_index, 40697]); separate from action-noise RNG",
            "cap_and_reset": "a query reaching or exceeding delay_upper_ms uses the capped upper age; the following query resets to the lower age with no increment draw",
            "frame_selection": "age is rounded upward to a historical control frame, never newer than the requested age; delay_upper_ms must align to the frame grid",
            "query_index": "environment step // replan_steps; held constant while a predicted action chunk executes; video and terminal frames do not advance the schedule",
            "simulator": "no sleeps; the old camera image comes from the control-step history",
            "pairing": "same task, initial state, repeat seed, warmup, replan steps and explicit policy noise at matched query steps",
            "outcomes": "success rate denominator = success + failure; errors and interruptions are counted separately",
        },
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    write_csv(directory / "summary.csv", records)
    write_csv(directory / "success_rates.csv", report["aggregate"])


def observation_delay_series(args: Args, record: dict, rows: list[dict]) -> dict:
    """Measure camera ages from recorded query/source steps, not the planned ramp."""
    if not rows or len(rows) != record["n_queries"]:
        raise ValueError("Missing or miscounted observation delay rows")
    for index, row in enumerate(rows):
        if (row["condition"] != record["condition"]
                or int(row["repeat_index"]) != record["repeat_index"]
                or int(row["query_index"]) != index):
            raise ValueError("Observation delay rows identify a different query/condition/repeat")
    steps = np.asarray([int(row["step"]) for row in rows])
    sources = np.asarray([
        [int(row[key]) for key in ("agentview_source_step", "wrist_source_step", "state_source_step")]
        for row in rows
    ])
    if not np.array_equal(steps, np.arange(len(rows)) * args.replan_steps):
        raise ValueError("Observation steps do not match actual policy queries")
    ages = (steps[:, None] - sources) * (1000.0 / args.control_hz)
    effective = np.asarray([float(row["effective_ms"]) for row in rows])
    selected = _shift.CAMERA_KEYS.index(args.delay_target)
    expected = np.zeros_like(ages)
    expected[:, selected] = effective
    if (not np.all(np.isfinite(effective)) or np.any(effective < 0)
            or not np.allclose(ages, expected, rtol=0, atol=1e-7)):
        raise ValueError("Recorded source steps disagree with camera ages or current unselected inputs")
    return {
        "observation_index": np.arange(1, len(rows) + 1),
        "agentview_ms": ages[:, 0],
        "wrist_ms": ages[:, 1],
        "max_ms": np.max(ages[:, :2], axis=1),
        "gap_ms": np.abs(ages[:, 0] - ages[:, 1]),
    }


def write_observation_delay_plots(directory: pathlib.Path, args: Args, records: list[dict]) -> list[str]:
    """Write one camera-age/gap figure per episode using its saved delay CSV."""
    entries = []
    for record in records:
        if not record["n_queries"]:
            continue
        path = record.get("delay_schedule_path")
        if not path:
            raise ValueError("An episode with policy queries is missing its delay CSV")
        with (directory / path).open(newline="", encoding="utf-8") as stream:
            series = observation_delay_series(args, record, list(csv.DictReader(stream)))
        entries.append((record, series))
    if not entries:
        return []

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    target = directory / "observation_delays"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for record, series in entries:
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
        )
        try:
            x = series["observation_index"]
            top.plot(x, series["max_ms"], color="0.3", linestyle="--", linewidth=2.4,
                     alpha=0.65, label="Max over cameras", zorder=1)
            top.plot(x, series["wrist_ms"], color="#267eaf", marker=".", markersize=4,
                     linewidth=1, label="Wrist", zorder=2)
            top.plot(x, series["agentview_ms"], color="#ec8b24", marker=".", markersize=4,
                     linewidth=1, label="Agentview", zorder=2)
            for value, label in ((args.delay_lower_ms, "lower"), (args.delay_upper_ms, "upper")):
                top.axhline(value, color="0.55", linestyle=":", linewidth=1)
                top.text(0.995, value, f" {label}: {value:g} ms", transform=top.get_yaxis_transform(),
                         ha="right", va="bottom", fontsize=8, color="0.35")
            median, p95 = np.percentile(series["max_ms"], [50, 95])
            top.text(0.99, 0.97,
                     f"Max-camera age: median {median:.1f} ms | P95 {p95:.1f} ms | "
                     f"max {series['max_ms'].max():.1f} ms",
                     transform=top.transAxes, ha="right", va="top", fontsize=9)
            top.set_ylabel("Camera image age at observation (ms)")
            top.legend(loc="upper left", ncol=3, fontsize=9)
            bottom.plot(x, series["gap_ms"], color="#e54b4b", marker=".", markersize=4,
                        linewidth=1, label="|Wrist age - Agentview age|")
            bottom.fill_between(x, 0, series["gap_ms"], color="#e54b4b", alpha=0.08)
            bottom.set(xlabel="Policy observation index (1-based)", ylabel="Camera age gap (ms)")
            bottom.legend(loc="upper left", fontsize=9)
            limit = max(args.delay_upper_ms, float(series["max_ms"].max()), 1)
            for axis in (top, bottom):
                axis.grid(alpha=0.2)
                axis.set_ylim(-0.06 * limit, 1.16 * limit)
            bottom.set_xlim(0.5, max(1.5, len(x) + 0.5))
            bottom.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
            fig.suptitle(
                f"{args.task_suite_name} task {args.task_id} | {record['condition']} | "
                f"repeat {record['repeat_index']} | {record['outcome']}\n"
                f"Selected camera: {args.delay_target} | {len(x)} policy observations", fontsize=12,
            )
            fig.text(0.5, 0.015,
                     "Age = (observation step - image source step) / control rate; "
                     "simulation history age, excluding inference/network wall time.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            path = target / f"{record['condition']}_repeat_{record['repeat_index']:03d}.png"
            fig.savefig(path, dpi=160)
            paths.append(path.relative_to(directory).as_posix())
        finally:
            plt.close(fig)
    logging.info("Wrote %d observation-delay plots to %s", len(paths), target)
    return paths


def _preview(args: Args) -> pathlib.Path:
    validate_args(args)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = pathlib.Path(args.output_dir) / f"preview_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    requested = args.preview_queries
    rows = [row for name in conditions_for(args) for row in _sample_rows(args, name, 0, requested)]
    write_csv(directory / "delay_schedule.csv", rows)
    save_summary(directory, args, {}, [])
    logging.info("Schedule preview written to %s", directory)
    return directory / "summary.json"


def _run_rollouts(args: Args) -> pathlib.Path:
    upper_steps, max_steps = validate_args(args)
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError("task_id out of range")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError("initial_state_id out of range")
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        metadata = client.get_server_metadata()
        shape = _trajectory.noise_shape_from_metadata(metadata)
        if not callable(getattr(client, "infer_with_noise", None)) or args.replan_steps > shape[0]:
            raise ValueError("Updated infer_with_noise client required; replan_steps must not exceed action horizon")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = pathlib.Path(args.output_dir) / stamp
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        warmup = max(args.num_steps_wait, upper_steps)
        save_summary(directory, args, metadata, records)
        logging.info(
            "Running %s on %s: delay %.0f-%.0f ms, "
            "increment Uniform(%.0f, %.0f) ms per query; %d repeats",
            ",".join(conditions_for(args)), args.delay_target,
            args.delay_lower_ms, args.delay_upper_ms,
            args.increment_min_ms, args.increment_max_ms, args.repeats_per_condition,
        )
        for repeat_index in range(args.repeats_per_condition):
            order = np.random.default_rng(np.random.SeedSequence([args.seed, repeat_index, 987])).permutation(
                len(conditions_for(args))
            )
            for condition_index in order:
                condition = conditions_for(args)[condition_index]

                def frame_label(step, query, delay):
                    query_index = step // args.replan_steps
                    sample = sample_delay(args, condition, repeat_index, query_index)
                    return (
                        f"{condition} | query={query_index} cycle={sample.cycle_index} "
                        f"position={sample.query_in_cycle + 1}\n"
                        f"increment={sample.increment_ms:.1f}ms "
                        f"requested={sample.requested_ms:.1f}ms effective={delay / args.control_hz * 1000:.1f}ms "
                        f"{'QUERY' if query else 'NO QUERY'}"
                    )

                setting = _trajectory.Setting(
                    condition,
                    _shift.BASELINE_TARGET if condition == "clean" else args.delay_target,
                    0 if condition == "clean" else round(sample_delay(args, condition, repeat_index, 0).effective_ms),
                    0 if condition == "clean" else sample_delay(args, condition, repeat_index, 0).delay_steps,
                )
                schedule = delay_schedule(args, condition, repeat_index)
                record = _trajectory.run_episode(
                    args=args,
                    client=client,
                    task=task,
                    initial_state=initial_states[args.initial_state_id],
                    prompt=prompt,
                    setting=setting,
                    repeat_index=repeat_index,
                    noise_shape=shape,
                    warmup_steps=warmup,
                    history_steps=upper_steps,
                    max_steps=max_steps,
                    output_dir=directory,
                    delay_schedule=schedule,
                    frame_label=frame_label,
                )
                save_query_schedule(directory, args, record)
                record.update({
                    "delay_target": args.delay_target,
                    "delay_ms_requested": (
                        0.0 if condition == "clean" else args.fixed_delay_ms if condition == "fixed_delay" else None
                    ),
                    "delay_steps": (
                        setting.delay_steps if condition in ("clean", "fixed_delay") else None
                    ),
                    "effective_delay_ms": (
                        setting.delay_steps / args.control_hz * 1000 if condition in ("clean", "fixed_delay") else None
                    ),
                    "delay_lower_ms": args.delay_lower_ms,
                    "delay_upper_ms": args.delay_upper_ms,
                    "increment_min_ms": args.increment_min_ms if condition == "increment_delay" else 0.0,
                    "increment_max_ms": args.increment_max_ms if condition == "increment_delay" else 0.0,
                })
                records.append(record)
                save_summary(directory, args, metadata, records)
                logging.info(
                    "[%d/%d] %s repeat=%d outcome=%s steps=%d queries=%d mean_age=%.1fms",
                    len(records), len(conditions_for(args)) * args.repeats_per_condition,
                    condition, repeat_index, record["outcome"], record["steps"], record["n_queries"],
                    record["effective_delay_ms_mean"] or 0.0,
                )
                if record["outcome"] == "error":
                    raise RuntimeError(f"Stopped after error; partial data saved in {directory}: {record['error']}")
                if record["outcome"] == "interrupted":
                    return directory / "summary.json"
        return directory / "summary.json"
    finally:
        client._ws.close()


def run(args: Args) -> pathlib.Path:
    validate_args(args)
    if args.preview_only:
        return _preview(args)
    summary_path = _run_rollouts(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = ["Task success rates (successes / completed episodes):"]
    for row in aggregate(args, summary["records"]):
        rate = row["success_rate"]
        value = f"{rate:.2%}" if rate is not None else "N/A"
        lines.append(
            f"  {row['condition']}: {row['successes']}/{row['completed']} = {value}"
            f" | errors={row['errors']} interrupted={row['interrupted']} missing={row['missing_runs']}"
        )
    report = "\n".join(lines)
    (summary_path.parent / "success_rates.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    if args.save_delay_plots:
        write_observation_delay_plots(summary_path.parent, args, summary["records"])
    return summary_path


def parse_args(argv=None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        default = getattr(defaults, field.name)
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if field.name == "repeats_per_condition":
            flags.extend(("--repeats-per-setting", "--repeats_per_setting"))
        if field.name == "save_video":
            parser.add_argument(*flags, dest=field.name, action="store_true", default=False)
            parser.add_argument("--no-video", "--no_video", dest=field.name, action="store_false")
        elif field.name == "save_delay_plots":
            parser.add_argument(*flags, dest=field.name, action="store_true", default=True)
            parser.add_argument("--no-delay-plots", "--no_delay_plots", dest=field.name, action="store_false")
        elif isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=default)
        else:
            if field.name in ("max_steps", "increment_seed"):
                kind = int
            elif field.name == "fixed_delay_ms":
                kind = float
            else:
                kind = str if default is None else type(default)
            parser.add_argument(*flags, dest=field.name, type=kind, default=default)
    return Args(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run(parse_args()))
