"""Compare clean, persistent, early-only and late-only camera delay.

Time zero is the first policy-controlled action after common warmup. At
--switch-time-s, all four groups discard their queued actions and replan.
Only the selected camera changes age; no object poses or pixels are edited.
Success-rate and trajectory figures are generated automatically after collection.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import functools
import json
import logging
import pathlib
from typing import Optional

import numpy as np

try:
    from examples.libero import run_trajectory_delay as _trajectory
except ImportError:
    import run_trajectory_delay as _trajectory

_shift = _trajectory._shift
EXPERIMENT = "libero_temporal_delay"
SCHEMA_VERSION = 1
CONDITIONS = ("clean", "full_delay", "early_delay", "late_delay")


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None
    delay_target: str = "observation/wrist_image"
    delay_ms: int = 600
    switch_time_s: float = 2.0
    repeats_per_condition: int = 30
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    object_names: str = ""
    output_dir: str = "rollout/libero_temporal_delay"
    skip_analysis: bool = False


def _exact_steps(seconds, hz, label):
    steps = seconds * hz
    if not np.isfinite(steps) or steps <= 0 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError(f"{label} must be positive and an integer multiple of 1/control_hz ({1 / hz:g} s)")
    return int(round(steps))


def validate_args(args):
    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unsupported task suite {args.task_suite_name!r}")
    if args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError(f"delay_target must be one camera from {_shift.CAMERA_KEYS}")
    if args.seed < 0 or args.repeats_per_condition < 1:
        raise ValueError("seed must be nonnegative and repeats_per_condition positive")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.video_fps) < 1 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size, video rate or warmup")
    delay_steps = _exact_steps(args.delay_ms / 1000, args.control_hz, "delay_ms")
    switch_step = _exact_steps(args.switch_time_s, args.control_hz, "switch_time_s")
    max_steps = args.max_steps if args.max_steps is not None else _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
    if max_steps <= switch_step:
        raise ValueError("max_steps must be greater than the switch step so both time windows can be tested")
    return delay_steps, switch_step, max_steps


def delay_at_step(step, *, condition, switch_step, delay_steps):
    if step < 0 or condition not in CONDITIONS:
        raise ValueError(f"Invalid step/condition: {step}, {condition}")
    active = (condition == "full_delay"
              or condition == "early_delay" and step < switch_step
              or condition == "late_delay" and step >= switch_step)
    return delay_steps if active else 0


def aggregate(records, repeats):
    grouped = collections.defaultdict(list)
    seen = set()
    for record in records:
        key = (record["condition"], record["repeat_index"])
        if key in seen or key[0] not in CONDITIONS or not 0 <= key[1] < repeats:
            raise ValueError(f"Duplicate or invalid condition/repeat {key}")
        if record["outcome"] not in ("success", "failure", "error", "interrupted"):
            raise ValueError(f"Unknown outcome {record['outcome']!r}")
        seen.add(key)
        grouped[key[0]].append(record)
    result = []
    for name in CONDITIONS:
        group = grouped[name]
        counts = collections.Counter(record["outcome"] for record in group)
        completed = counts["success"] + counts["failure"]
        result.append({
            "condition": name, "runs": len(group), "expected_runs": repeats,
            "missing_runs": repeats - len(group), "completed": completed,
            "successes": counts["success"], "failures": counts["failure"],
            "errors": counts["error"], "interrupted": counts["interrupted"],
            "success_rate": counts["success"] / completed if completed else None,
            "successes_before_switch": sum(record.get("success_before_switch", False) for record in group),
            "completed_with_post_switch_actions": sum(
                record.get("post_switch_executed", False) and record["outcome"] in ("success", "failure")
                for record in group
            ),
        })
    return result


def write_csv(path, rows):
    if rows:
        with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_summary(directory, args, metadata, records):
    delay, boundary, max_steps = validate_args(args)
    totals = aggregate(records, args.repeats_per_condition)
    report = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION,
        "config": dataclasses.asdict(args), "server_metadata": metadata,
        "switch_step": boundary, "switch_time_s_effective": boundary / args.control_hz,
        "delay_steps": delay, "delay_ms_effective": delay / args.control_hz * 1000,
        "max_steps": max_steps,
        "conditions": [{"name": name} for name in CONDITIONS],
        "protocol": {
            "time_origin": "first policy-controlled action after common warmup; simulator time, not inference wall time",
            "early_delay": "delayed camera if step < switch_step, otherwise current camera",
            "late_delay": "current camera if step < switch_step, otherwise delayed camera",
            "switch": "discard queued actions and replan at switch_step in ALL groups; no environment reset",
            "history": "record every control step, including warmup and intervals with delay disabled",
            "delay_semantics": "at an active query t, selected camera uses frame t-delay_steps, even across switch",
            "other_inputs": "unselected camera and self state are always current; no pose edits or image compositing",
            "pairing": "same task, initial state, seed, warmup, query schedule and explicit noise at matched steps",
            "noise": "PCG64 SeedSequence([seed, repeat_index, env_step])",
            "success": "LIBERO success predicate; denominator = success + failure; errors/interrupted separate",
            "early_termination": "success before boundary remains in overall rate and is flagged; no forced continuation",
            "input_source_columns": [*_shift.CAMERA_KEYS, _shift.STATE_TARGET],
            "video": "top=current cameras; bottom=candidate policy inputs; only QUERY frames are sent to server",
            "scope": "fixed-time interventions in independent closed-loop rollouts; not manually labelled task stages",
        },
        "aggregate": totals, "records": records,
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    write_csv(directory / "summary.csv", records)
    write_csv(directory / "success_rates.csv", totals)


def run_episode(*, args, condition, **kwargs):
    delay_steps, switch_step, _ = validate_args(args)
    schedule = functools.partial(delay_at_step, condition=condition,
                                 switch_step=switch_step, delay_steps=delay_steps)

    def frame_label(step, query, delay):
        return (f"{condition} | t={step / args.control_hz:.2f}s | switch={args.switch_time_s:g}s\n"
                f"{args.delay_target} | delay={delay / args.control_hz * 1000:g}ms | "
                f"{'QUERY' if query else 'NO QUERY'}")

    setting = _trajectory.Setting(condition, args.delay_target, 0 if condition == "clean" else args.delay_ms,
                                  0 if condition == "clean" else delay_steps)
    record = _trajectory.run_episode(
        args=args, setting=setting, delay_schedule=schedule,
        forced_replan_steps=(switch_step,), frame_label=frame_label, **kwargs,
    )
    record.update({
        "switch_step": switch_step, "switch_time_s": switch_step / args.control_hz,
        "replan_steps": args.replan_steps,
        "post_switch_executed": record["steps"] > switch_step,
        "success_before_switch": record["outcome"] == "success" and record["steps"] <= switch_step,
    })
    return record


def _run_rollouts(args):
    delay, boundary, max_steps = validate_args(args)
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
        warmup = max(args.num_steps_wait, delay)
        save_summary(directory, args, metadata, records)
        logging.info("Temporal delay: %s, %d ms; switch at %.3f s (step %d after warmup)",
                     args.delay_target, args.delay_ms, boundary / args.control_hz, boundary)
        logging.info("Four conditions x %d repeats; all groups replan at switch; outputs: %s",
                     args.repeats_per_condition, directory)
        for repeat in range(args.repeats_per_condition):
            order = np.random.default_rng(np.random.SeedSequence([args.seed, repeat, 987])).permutation(len(CONDITIONS))
            for index in order:
                record = run_episode(
                    args=args, condition=CONDITIONS[index], client=client, task=task,
                    initial_state=initial_states[args.initial_state_id], prompt=prompt,
                    repeat_index=repeat, noise_shape=shape, warmup_steps=warmup,
                    history_steps=delay, max_steps=max_steps, output_dir=directory,
                )
                records.append(record)
                save_summary(directory, args, metadata, records)
                logging.info("[%d/%d] %s repeat=%d outcome=%s steps=%d delayed_queries=%d",
                             len(records), len(CONDITIONS) * args.repeats_per_condition,
                             record["condition"], repeat, record["outcome"], record["steps"],
                             record["n_delayed_queries"])
                if record["outcome"] == "error":
                    raise RuntimeError(f"Stopped after error; partial data in {directory}: {record['error']}")
                if record["outcome"] == "interrupted":
                    return directory / "summary.json"
        return directory / "summary.json"
    finally:
        client._ws.close()


def run(args):
    validate_args(args)
    if args.skip_analysis:
        return _run_rollouts(args)
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Automatic analysis requires matplotlib; use --skip-analysis for collection only") from error
    try:
        from examples.libero import analyze_temporal_delay as analysis
    except ImportError:
        import analyze_temporal_delay as analysis
    summary = _run_rollouts(args)
    try:
        result = analysis.analyze(summary)
    except Exception as error:
        raise RuntimeError(
            f"Raw results are saved at {summary}; analysis failed. "
            "Rerun analyze_temporal_delay.py on this directory after fixing the analysis error."
        ) from error
    logging.info("Success rates and figures saved to %s", result.parent)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        default = getattr(defaults, field.name)
        if field.name == "save_video":
            parser.add_argument("--no-video", "--no_video", dest=field.name, action="store_false", default=True)
            continue
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=default)
        else:
            kind = int if field.name == "max_steps" else str if default is None else type(default)
            parser.add_argument(*flags, dest=field.name, type=kind, default=default)
    return Args(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run(parse_args()))
