"""Delay one camera while the measured gripper is closed; restore on opening.

Independent of run_temporal_delay.py. Three closed-loop conditions share task,
initial state and explicit noise: clean, full_delay, and closed_gripper_delay.
Default replan_steps=1 matches query times across groups. With larger chunks,
each group's own measured gripper transitions force an immediate replan.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import hashlib
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
EXPERIMENT = "libero_gripper_delay"
SCHEMA_VERSION = 1
CONDITIONS = ("clean", "full_delay", "closed_gripper_delay")


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
    close_width_mm: float = 20.0
    open_width_mm: float = 40.0
    repeats_per_condition: int = 30
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 1
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    object_names: str = ""
    output_dir: str = "rollout/libero_gripper_delay"
    skip_analysis: bool = False
    episode_plot_limit: int = 12


def validate_args(args):
    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError("Unsupported task suite")
    if args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError(f"Choose one camera from {_shift.CAMERA_KEYS}")
    if args.seed < 0 or args.repeats_per_condition < 1:
        raise ValueError("seed must be nonnegative; repeats_per_condition must be positive")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.video_fps) < 1 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size, video rate or warmup")
    if args.episode_plot_limit < 0:
        raise ValueError("episode_plot_limit must be nonnegative")
    GripperGate(args.close_width_mm, args.open_width_mm)
    exact = args.delay_ms * args.control_hz / 1000
    if not np.isfinite(exact) or exact <= 0 or not np.isclose(exact, round(exact), rtol=0, atol=1e-8):
        raise ValueError("delay_ms must be positive and map to an integer control-step count")
    max_steps = args.max_steps if args.max_steps is not None else _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    return int(round(exact)), max_steps


def gripper_width_mm(qpos):
    """Opening proxy for two prismatic fingers: |q_left| + |q_right|.

    Inputs are measured joint displacements in metres from the closed zero.
    Absolute values support both same-sign and opposite-sign finger axes.
    This is not a contact, grasp-success or action-command detector.
    """
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected two finite measured finger joint positions")
    return float(np.abs(values).sum() * 1000)


class GripperGate:
    def __init__(self, close_width_mm=20.0, open_width_mm=40.0):
        if (not np.isfinite(close_width_mm) or not np.isfinite(open_width_mm)
                or not 0 <= close_width_mm < open_width_mm):
            raise ValueError("Require finite 0 <= close_width_mm < open_width_mm")
        self.close_width_mm = close_width_mm
        self.open_width_mm = open_width_mm
        self.closed = False

    def update(self, qpos):
        width = gripper_width_mm(qpos)
        previous = self.closed
        if self.closed:
            if width >= self.open_width_mm:
                self.closed = False
        elif width <= self.close_width_mm:
            self.closed = True
        return width, self.closed, self.closed != previous


def effective_delay(condition, closed, delay_steps):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    return delay_steps if condition == "full_delay" or condition == "closed_gripper_delay" and closed else 0


def write_csv(path, rows):
    if rows:
        with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def aggregate(records, repeats):
    seen, grouped = set(), collections.defaultdict(list)
    for record in records:
        key = (record["condition"], record["repeat_index"])
        if key in seen or key[0] not in CONDITIONS or not 0 <= key[1] < repeats:
            raise ValueError(f"Invalid or duplicate condition/repeat: {key}")
        if record["outcome"] not in ("success", "failure", "error", "interrupted"):
            raise ValueError("Unknown outcome")
        seen.add(key)
        grouped[key[0]].append(record)
    rows = []
    for condition in CONDITIONS:
        group = grouped[condition]
        counts = collections.Counter(record["outcome"] for record in group)
        completed = [record for record in group if record["outcome"] in ("success", "failure")]
        rows.append({
            "condition": condition, "runs": len(group), "completed": len(completed),
            "missing_runs": repeats - len(group), "successes": counts["success"], "failures": counts["failure"],
            "errors": counts["error"], "interrupted": counts["interrupted"],
            "success_rate": counts["success"] / len(completed) if completed else None,
            "completed_without_closed_actions": sum(record["n_closed_actions"] == 0 for record in completed),
            "completed_with_delayed_actions": sum(record["n_delayed_actions"] > 0 for record in completed),
        })
    return rows


def run_episode(*, args, client, task, initial_state, prompt, condition, repeat_index,
                noise_shape, output_dir):
    delay_steps, max_steps = validate_args(args)
    effective_delay(condition, False, delay_steps)
    warmup = max(args.num_steps_wait, delay_steps)
    env = None
    images = {source: collections.deque(maxlen=delay_steps + 1) for source in _shift.CAMERA_SOURCES}
    states, plan = collections.deque(maxlen=delay_steps + 1), collections.deque()
    samples, frames, actions, chunks, noises, queries, source_steps, policy_states = [], [], [], [], [], [], [], []
    widths, closed_samples, changed_samples, query_delays, action_queries, action_delays, events = [], [], [], [], [], [], []
    gate = GripperGate(args.close_width_mm, args.open_width_mm)
    outcome, reason, error_text = "failure", "max_steps", None
    hz, objects, missing_objects, start_sim = None, [], [], np.empty(0)

    def record_observation(obs):
        sample = _trajectory.state_sample(env, obs, objects)
        width, closed, changed = gate.update(sample["gripper_qpos"])
        samples.append(sample)
        widths.append(width)
        closed_samples.append(closed)
        changed_samples.append(changed)
        return width, closed, changed

    def policy_observation(delay):
        return _shift._build_delayed_observation(image_history=images, state_history=states,
                                                 delay_target=args.delay_target, delay_steps=delay, prompt=prompt)

    def save_frame(step, policy_obs, query, delay):
        if not args.save_video:
            return
        from PIL import Image, ImageDraw
        frame = _shift._make_video_frame(
            current_images={key: images[source][-1] for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)},
            policy_images={key: policy_obs[key] for key in _shift.CAMERA_KEYS},
            current_state=states[-1], policy_state=policy_obs["observation/state"],
            delay_target=args.delay_target, effective_delay_ms=delay / hz * 1000,
            step=step, policy_query=query,
        )
        # Pad to a codec-friendly size while keeping both images untouched.
        w, h = ((frame.shape[1] + 15) // 16) * 16, ((frame.shape[0] + 48 + 15) // 16) * 16
        canvas = Image.new("RGB", (w, h), "black")
        canvas.paste(Image.fromarray(frame), (0, 48))
        text = (f"{condition} | t={step / hz:.2f}s | {'QUERY' if query else 'NO QUERY'}\n"
                f"width={widths[-1]:.1f}mm | gate={'CLOSED' if gate.closed else 'OPEN'} | delay={delay / hz * 1000:g}ms")
        ImageDraw.Draw(canvas).text((6, 4), text, fill="white")
        frames.append(np.asarray(canvas))

    try:
        env = _shift._create_env(task, args.seed + repeat_index)
        hz = _trajectory.simulator_hz(env)
        if not np.isclose(hz, args.control_hz, rtol=0, atol=1e-8):
            raise ValueError(f"control_hz={args.control_hz} differs from simulator control_freq={hz}")
        env.reset()
        obs = env.set_init_state(np.array(initial_state, copy=True))
        _shift._append_snapshot(images, states, obs, args.resize_size)
        for _ in range(warmup):
            obs, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _shift._append_snapshot(images, states, obs, args.resize_size)
            if done or env.check_success():
                raise RuntimeError("Task ended during warmup")
        objects, missing_objects = _trajectory.object_references(env, args.object_names)
        start_sim = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        record_observation(obs)
        for step in range(max_steps):
            changed = changed_samples[-1]
            if changed:
                # Apply the same event-replan rule to each control's own
                # gripper too. With chunks > 1 event times may differ.
                plan.clear()
                events.append({"step": step, "time_s": step / hz, "width_mm": widths[-1],
                               "closed": gate.closed, "initial": step == 0})
                logging.info("%s repeat=%d step=%d width=%.2f mm gate=%s", condition, repeat_index,
                             step, widths[-1], "closed" if gate.closed else "open")
            delay = effective_delay(condition, gate.closed, delay_steps)
            policy_obs = policy_observation(delay)
            query = not plan
            save_frame(step, policy_obs, query, delay)
            if query:
                noise = _trajectory.paired_noise(args.seed, repeat_index, step, noise_shape)
                chunk = np.asarray(client.infer_with_noise(policy_obs, noise)["actions"], dtype=np.float32)
                if chunk.shape != (noise_shape[0], 7) or not np.all(np.isfinite(chunk)):
                    raise ValueError(f"Expected finite actions {(noise_shape[0], 7)}, got {chunk.shape}")
                queries.append(step)
                query_delays.append(delay)
                source_steps.append([step - delay if target == args.delay_target else step
                                     for target in (*_shift.CAMERA_KEYS, _shift.STATE_TARGET)])
                noises.append(noise)
                chunks.append(chunk.copy())
                policy_states.append(np.array(policy_obs["observation/state"], copy=True))
                plan.extend(chunk[:args.replan_steps])
            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            actions.append(action.copy())
            action_queries.append(queries[-1])
            action_delays.append(query_delays[-1])
            _shift._append_snapshot(images, states, obs, args.resize_size)
            record_observation(obs)
            if env.check_success():
                outcome, reason = "success", "task_success"
                break
            if done:
                reason = "environment_terminated_without_success"
                break
        terminal_delay = effective_delay(condition, gate.closed, delay_steps)
        save_frame(len(actions), policy_observation(terminal_delay), False, terminal_delay)
    except KeyboardInterrupt:
        outcome, reason, error_text = "interrupted", "interrupted", "KeyboardInterrupt"
    except Exception as error:
        outcome, reason, error_text = "error", "runtime_error", repr(error)
        logging.exception("Gripper-delay run failed: %s repeat=%d", condition, repeat_index)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("Environment cleanup failed")

    slug = f"{condition}_repeat_{repeat_index:03d}"
    trace_path = output_dir / f"{slug}_trajectory.npz"
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]} if samples else {}
    np.savez_compressed(
        trace_path, **arrays, steps=np.arange(len(samples), dtype=np.int32),
        time_s=np.arange(len(samples)) / args.control_hz,
        gripper_width_mm=np.asarray(widths, dtype=np.float64),
        gate_closed=np.asarray(closed_samples, dtype=bool), gate_changed=np.asarray(changed_samples, dtype=bool),
        executed_actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
        query_steps=np.asarray(queries, dtype=np.int32), query_delay_steps=np.asarray(query_delays, dtype=np.int32),
        action_query_steps=np.asarray(action_queries, dtype=np.int32), action_delay_steps=np.asarray(action_delays, dtype=np.int32),
        input_source_steps=np.asarray(source_steps, dtype=np.int32).reshape(-1, 3),
        action_chunks=np.asarray(chunks, dtype=np.float32).reshape(-1, noise_shape[0], 7),
        noise=np.asarray(noises, dtype=np.float32).reshape(-1, *noise_shape),
        policy_states=np.asarray(policy_states, dtype=np.float32).reshape(-1, 8),
        object_names=np.asarray([name for name, _ in objects], dtype=str), start_sim_state=start_sim,
    )
    write_csv(output_dir / f"{slug}_events.csv", events)
    video_path = output_dir / f"{slug}_{outcome}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps) if frames else (False, None)
    n_closed = int(np.count_nonzero(closed_samples[:len(actions)]))
    if condition == "closed_gripper_delay" and outcome in ("success", "failure") and n_closed == 0:
        logging.warning("No closed-gripper actions in repeat %d; inspect gripper widths/thresholds", repeat_index)
    return {
        "condition": condition, "repeat_index": repeat_index, "seed": args.seed + repeat_index,
        "task_suite_name": args.task_suite_name, "task_id": args.task_id, "initial_state_id": args.initial_state_id,
        "initial_state_sha256": hashlib.sha256(np.asarray(initial_state, dtype=np.float64).tobytes()).hexdigest(),
        "prompt": prompt, "delay_target": args.delay_target, "delay_ms_requested": args.delay_ms,
        "delay_steps": delay_steps, "control_hz": hz, "warmup_steps": warmup, "replan_steps": args.replan_steps,
        "close_width_mm": args.close_width_mm, "open_width_mm": args.open_width_mm,
        "outcome": outcome, "termination_reason": reason, "success": outcome == "success",
        "steps": len(actions), "n_samples": len(samples), "n_queries": len(queries),
        "n_closed_actions": n_closed, "n_delayed_queries": int(np.count_nonzero(query_delays)),
        "n_delayed_actions": int(np.count_nonzero(action_delays)),
        "n_close_events": sum(event["closed"] and not event["initial"] for event in events),
        "n_open_events": sum(not event["closed"] and not event["initial"] for event in events),
        "initially_closed": bool(closed_samples[0]) if closed_samples else None,
        "trace_path": trace_path.name, "events_path": f"{slug}_events.csv" if events else None,
        "video_path": video_path.name if video_ok else None, "video_fps": args.video_fps,
        "video_frames": len(frames), "video_error": video_error,
        "object_names": [name for name, _ in objects], "unavailable_objects": missing_objects, "error": error_text,
    }


def save_summary(directory, args, metadata, records):
    delay, max_steps = validate_args(args)
    totals = aggregate(records, args.repeats_per_condition)
    report = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "config": dataclasses.asdict(args),
        "server_metadata": metadata, "conditions": [{"name": name} for name in CONDITIONS],
        "delay_steps": delay, "delay_ms_effective": delay / args.control_hz * 1000, "max_steps": max_steps,
        "protocol": {
            "trigger": "current measured robot0_gripper_qpos, never delayed state or predicted gripper command",
            "width": "1000 * sum(abs(qpos)); two prismatic finger displacements in metres from closed zero",
            "gate": "initially open; width <= close_width_mm sets closed; width >= open_width_mm reopens; hold between",
            "initial_state": "if already below close threshold after warmup, delay starts at policy step zero",
            "replanning": "check at every control step; each group's own gate transitions clear its action queue",
            "noise": "PCG64 SeedSequence([seed, repeat_index, env_step]); shared at matched environment steps",
            "pairing": "same task, initial state, seed and warmup; replan_steps=1 matches query times across groups",
            "chunk_caveat": "replan_steps>1 may yield different query times across groups as measured event times diverge",
            "camera": "only selected camera uses the full historical frame at t-delay; other camera/state remain current",
            "history": "record every control step including warmup, also while gate is open",
            "time_origin": "first policy action after warmup; switching takes effect on first measured threshold crossing",
            "success": "LIBERO task predicate; success/(success+failure); errors and interruptions separate",
            "terminal": "last gate sample is recorded but no further inference occurs after success/termination",
            "video": "top=current views, bottom=candidate inputs; only QUERY frames are sent to server",
            "interpretation": "threshold-defined closure does not establish grasp success; no environment reset on reopening",
        },
        "aggregate": totals, "records": records,
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    write_csv(directory / "summary.csv", records)
    write_csv(directory / "success_rates.csv", totals)


def _run_rollouts(args):
    validate_args(args)
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
        save_summary(directory, args, metadata, records)
        logging.info("Gripper delay: close <= %.1f mm, reopen >= %.1f mm; %s delay %d ms; output %s",
                     args.close_width_mm, args.open_width_mm, args.delay_target, args.delay_ms, directory)
        if args.replan_steps > 1:
            logging.warning("Event-triggered replans with chunks > 1 may have different query times across conditions")
        for repeat in range(args.repeats_per_condition):
            order = np.random.default_rng(np.random.SeedSequence([args.seed, repeat, 987])).permutation(len(CONDITIONS))
            for index in order:
                record = run_episode(args=args, client=client, task=task, prompt=prompt,
                                     initial_state=initial_states[args.initial_state_id], condition=CONDITIONS[index],
                                     repeat_index=repeat, noise_shape=shape, output_dir=directory)
                records.append(record)
                save_summary(directory, args, metadata, records)
                logging.info("[%d/%d] %s repeat=%d outcome=%s steps=%d delayed_actions=%d",
                             len(records), len(CONDITIONS) * args.repeats_per_condition, record["condition"],
                             repeat, record["outcome"], record["steps"], record["n_delayed_actions"])
                if record["outcome"] == "error":
                    raise RuntimeError(f"Stopped after error; raw results in {directory}: {record['error']}")
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
        from examples.libero import analyze_gripper_delay as analysis
    except ImportError:
        import analyze_gripper_delay as analysis
    summary = _run_rollouts(args)
    try:
        result = analysis.analyze(summary, episode_plot_limit=args.episode_plot_limit)
    except Exception as error:
        raise RuntimeError(f"Raw results saved at {summary}; rerun analyze_gripper_delay.py after fixing analysis") from error
    logging.info("Gripper-delay analysis saved to %s", result.parent)
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
