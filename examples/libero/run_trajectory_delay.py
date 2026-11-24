"""Paired closed-loop trajectories: clean versus persistent observation delay.

Only the ordinary action endpoint is used. Simulator observations (not policy
inputs or action commands) supply the measured world-frame trajectory.
Analysis figures are generated automatically after the rollout connection closes.
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
    from examples.libero import rollout_video as _video
    from examples.libero import run_shift_match_sweep as _shift
except ImportError:
    import rollout_video as _video
    import run_shift_match_sweep as _shift


EXPERIMENT = "libero_trajectory_delay"
SCHEMA_VERSION = 1


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None
    delay_targets: str = "observation/wrist_image"
    delay_levels_ms: str = "600"
    repeats_per_setting: int = 30
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    object_names: str = ""  # Empty: available task objects of interest, including fixtures.
    output_dir: str = "rollout/libero_trajectory_delay"
    skip_analysis: bool = False
    interactive: bool = False
    threshold_cm: float = 2.0
    consecutive: int = 3
    pair_limit: int = 30


@dataclasses.dataclass(frozen=True)
class Setting:
    name: str
    target: str
    delay_ms: int
    delay_steps: int


def make_settings(args):
    if args.seed < 0 or args.repeats_per_setting <= 0:
        raise ValueError("seed must be nonnegative and repeats_per_setting positive")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.video_fps) <= 0 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size, video rate or warmup")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not np.isfinite(args.threshold_cm) or args.threshold_cm <= 0 or args.consecutive < 1 or args.pair_limit < 0:
        raise ValueError("Invalid analysis threshold, consecutive count, or pair limit")
    targets = _shift._split_csv(args.delay_targets)
    if any(target not in _shift.VALID_DELAY_TARGETS for target in targets):
        raise ValueError(f"Supported delay targets: {_shift.VALID_DELAY_TARGETS}")
    settings = [Setting("clean", _shift.BASELINE_TARGET, 0, 0)]
    for target in targets:
        for delay in _shift._parse_delay_levels(args.delay_levels_ms):
            exact_steps = delay * args.control_hz / 1000.0
            steps = round(exact_steps)
            if steps < 1 or not np.isclose(steps, exact_steps, rtol=0, atol=1e-8):
                raise ValueError("Each delay must be positive and map to an integer control-step count")
            settings.append(Setting(f"{_shift._slug(target)}_{delay}ms", target, delay, steps))
    return settings


def paired_noise(seed, repeat_index, step, shape):
    generator = np.random.default_rng(np.random.SeedSequence([seed, repeat_index, step]))
    return generator.standard_normal(shape).astype(np.float32)


def noise_shape_from_metadata(metadata):
    for name, version in (("wrist_kv_patch", 1), ("activation_drift", 2), ("attention", 1)):
        capability = metadata.get(name, {})
        if capability.get("schema_version") == version:
            shape = capability.get("noise_shape", [])
            if len(shape) == 2 and all(type(x) is int and x > 0 for x in shape) and shape[1] >= 7:
                return tuple(shape)
    raise RuntimeError("Server lacks noise-shape metadata; use the updated Pi0 server from the patch experiment")


def simulator_hz(env):
    for candidate in (env, getattr(env, "env", None)):
        value = getattr(candidate, "control_freq", None)
        if value is not None:
            return float(value)
    raise ValueError("Cannot verify simulator control_freq")


def object_references(env, requested):
    inner = getattr(env, "env", env)
    body_ids = getattr(inner, "obj_body_id", {})
    names = _shift._split_csv(requested) if requested.strip() else list(getattr(env, "obj_of_interest", []))
    missing = [name for name in names if name not in body_ids]
    if missing and requested.strip():
        raise ValueError(f"Unknown object bodies {missing}; available: {sorted(body_ids)}")
    if missing:
        logging.warning("Task objects without root-body positions: %s", missing)
    return [(name, int(body_ids[name])) for name in names if name in body_ids], missing


def state_sample(env, obs, objects):
    sample = {
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
        "eef_quat_xyzw": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy(),
        "joint_pos": np.asarray(obs.get("robot0_joint_pos", []), dtype=np.float64).copy(),
        "object_pos": np.asarray([env.sim.data.body_xpos[index] for _, index in objects],
                                 dtype=np.float64).reshape(-1, 3).copy(),
    }
    if sample["eef_pos"].shape != (3,) or sample["eef_quat_xyzw"].shape != (4,):
        raise ValueError("Expected EEF XYZ and xyzw quaternion")
    if sample["gripper_qpos"].shape != (2,) or sample["joint_pos"].ndim != 1:
        raise ValueError("Expected LIBERO Panda two-finger gripper and vector joint positions")
    if any(not np.all(np.isfinite(value)) for value in sample.values()):
        raise ValueError("Non-finite simulator state")
    if not np.isclose(np.linalg.norm(sample["eef_quat_xyzw"]), 1.0, atol=1e-3):
        raise ValueError("Invalid EEF quaternion")
    return sample


def run_episode(*, args, client, task, initial_state, prompt, setting, repeat_index,
                noise_shape, warmup_steps, history_steps, max_steps, output_dir,
                delay_schedule=None, forced_replan_steps=(), frame_label=None,
                capture_state_provenance=False, record_comparison_video=False):
    """Run a rollout, optionally changing observation age at selected steps.

    ``delay_schedule(step)`` returns an integer observation age. Boundary
    replans discard queued actions; callers must apply them to control groups
    too, so all conditions share query times and explicit noise.
    """
    env = None
    samples, frames, actions, chunks, noises, queries, source_steps, policy_states = [], [], [], [], [], [], [], []
    query_delays, action_queries, action_delays = [], [], []
    forced_replan_steps = frozenset(forced_replan_steps)
    history = {name: collections.deque(maxlen=history_steps + 1) for name in _shift.CAMERA_SOURCES}
    states = collections.deque(maxlen=history_steps + 1)
    quaternions = collections.deque(maxlen=history_steps + 1)
    query_current_states, query_source_states, query_current_quats, query_source_quats = [], [], [], []
    plan = collections.deque()
    outcome, reason, error_text = "failure", "max_steps", None
    actual_hz, objects, missing_objects, start_sim = None, [], [], np.empty(0)
    slug = f"{setting.name}_repeat_{repeat_index:03d}"
    video = (_video.ComparisonVideoRecorder(output_dir, slug, args.video_fps, args.control_hz, setting.target)
             if args.save_video and record_comparison_video else None)

    def delay_at(step):
        value = setting.delay_steps if delay_schedule is None else delay_schedule(step)
        if not isinstance(value, (int, np.integer)) or not 0 <= value <= history_steps:
            raise ValueError(f"Invalid scheduled delay {value!r} at step {step}")
        return int(value)

    def save_frame(step, policy_obs, query, delay, terminal=None):
        if video is not None:
            video.capture(current_images={key: history[source][-1] for key, source in
                                          zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)},
                          policy_obs=policy_obs, step=step, query=query, delay=delay, terminal=terminal)
            return
        if args.save_video:
            frame = _shift._make_video_frame(
                current_images={key: history[source][-1] for key, source in
                                zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)},
                policy_images={key: policy_obs[key] for key in _shift.CAMERA_KEYS},
                current_state=states[-1], policy_state=policy_obs["observation/state"],
                delay_target=setting.target, effective_delay_ms=delay / actual_hz * 1000,
                step=step, policy_query=query,
            )
            if frame_label is not None:
                from PIL import Image, ImageDraw
                # Labels also mark non-query frames, whose candidate input is
                # shown for inspection but is not sent to the policy server.
                labelled = Image.new("RGB", (frame.shape[1], frame.shape[0] + 48), "black")
                labelled.paste(Image.fromarray(np.asarray(frame, dtype=np.uint8)), (0, 48))
                ImageDraw.Draw(labelled).text((6, 4), frame_label(step, query, delay), fill="white")
                frame = np.asarray(labelled)
            frames.append(frame)

    def policy_observation(delay):
        return _shift._build_delayed_observation(
            image_history=history, state_history=states, delay_target=setting.target,
            delay_steps=delay, prompt=prompt,
        )

    try:
        env = _shift._create_env(task, args.seed + repeat_index)
        actual_hz = simulator_hz(env)
        if not np.isclose(actual_hz, args.control_hz, rtol=0, atol=1e-8):
            raise ValueError(f"control_hz={args.control_hz} differs from simulator control_freq={actual_hz}")
        env.reset()
        obs = env.set_init_state(np.array(initial_state, copy=True))
        _shift._append_snapshot(history, states, obs, args.resize_size)
        if capture_state_provenance:
            quaternions.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy())
        for _ in range(warmup_steps):
            obs, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _shift._append_snapshot(history, states, obs, args.resize_size)
            if capture_state_provenance:
                quaternions.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy())
            if done or env.check_success():
                raise RuntimeError("Task ended during warmup; no valid trajectory comparison")
        objects, missing_objects = object_references(env, args.object_names)
        start_sim = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        samples.append(state_sample(env, obs, objects))
        for step in range(max_steps):
            delay = delay_at(step)
            if step in forced_replan_steps:
                plan.clear()
            policy_obs = policy_observation(delay)
            query = not plan
            save_frame(step, policy_obs, query, delay)
            if query:
                if capture_state_provenance:
                    if setting.target not in (*_shift.CAMERA_KEYS, _shift.BASELINE_TARGET):
                        raise ValueError("State provenance capture requires current state and at most one delayed camera")
                    source_delay = delay if setting.target in _shift.CAMERA_KEYS else 0
                    query_current_states.append(states[-1].copy())
                    query_source_states.append(states[-1 - source_delay].copy())
                    query_current_quats.append(quaternions[-1].copy())
                    query_source_quats.append(quaternions[-1 - source_delay].copy())
                noise = paired_noise(args.seed, repeat_index, step, noise_shape)
                response = client.infer_with_noise(policy_obs, noise)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                if chunk.shape != (noise_shape[0], 7) or not np.all(np.isfinite(chunk)):
                    raise ValueError(f"Expected finite actions {(noise_shape[0], 7)}, got {chunk.shape}")
                queries.append(step)
                query_delays.append(delay)
                # Sources are indexed relative to the post-warmup trajectory start.
                source_steps.append([
                    step - delay if setting.target in (target, _shift.ALL_TARGET) else step
                    for target in (*_shift.CAMERA_KEYS, _shift.STATE_TARGET)
                ])
                chunks.append(chunk.copy())
                noises.append(noise)
                policy_states.append(np.array(policy_obs["observation/state"], copy=True))
                plan.extend(chunk[:args.replan_steps])
            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            actions.append(action.copy())
            action_queries.append(queries[-1])
            action_delays.append(query_delays[-1])
            samples.append(state_sample(env, obs, objects))
            _shift._append_snapshot(history, states, obs, args.resize_size)
            if capture_state_provenance:
                quaternions.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy())
            if env.check_success():
                outcome, reason = "success", "task_success"
                break
            if done:
                reason = "environment_terminated_without_success"
                break
        final_delay = delay_at(len(actions))
        save_frame(len(actions), policy_observation(final_delay), False, final_delay, terminal=outcome)
    except KeyboardInterrupt:
        outcome, reason, error_text = "interrupted", "interrupted", "KeyboardInterrupt"
    except Exception as error:
        outcome, reason, error_text = "error", "runtime_error", repr(error)
        logging.exception("Trajectory run failed: %s repeat=%d", setting.name, repeat_index)
    finally:
        if video is not None:
            if outcome in ("error", "interrupted") and samples:
                save_frame(len(actions), {}, False, 0, terminal=outcome)
            video_record = video.finish(outcome)
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("Environment cleanup failed")

    trace_path = output_dir / f"{slug}_trajectory.npz"
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]} if samples else {}
    if capture_state_provenance:
        # Source states are real measurements from this rollout, including
        # negative source steps in warmup; never borrowed from paired clean.
        arrays.update(
            query_current_states=np.asarray(query_current_states, dtype=np.float32).reshape(-1, 8),
            query_image_source_states=np.asarray(query_source_states, dtype=np.float32).reshape(-1, 8),
            query_current_quats=np.asarray(query_current_quats, dtype=np.float64).reshape(-1, 4),
            query_image_source_quats=np.asarray(query_source_quats, dtype=np.float64).reshape(-1, 4),
        )
    np.savez_compressed(
        trace_path, **arrays,
        steps=np.arange(len(samples), dtype=np.int32),
        time_s=np.arange(len(samples), dtype=np.float64) / args.control_hz,
        executed_actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
        query_steps=np.asarray(queries, dtype=np.int32),
        query_delay_steps=np.asarray(query_delays, dtype=np.int32),
        action_query_steps=np.asarray(action_queries, dtype=np.int32),
        action_delay_steps=np.asarray(action_delays, dtype=np.int32),
        forced_replan_steps=np.asarray(sorted(forced_replan_steps), dtype=np.int32),
        input_source_steps=np.asarray(source_steps, dtype=np.int32).reshape(-1, 3),
        action_chunks=np.asarray(chunks, dtype=np.float32).reshape(-1, noise_shape[0], 7),
        noise=np.asarray(noises, dtype=np.float32).reshape(-1, *noise_shape),
        policy_states=np.asarray(policy_states, dtype=np.float32).reshape(-1, 8),
        object_names=np.asarray([name for name, _ in objects], dtype=str), start_sim_state=start_sim,
    )
    if video is None:
        video_path = output_dir / f"{slug}_{outcome}.mp4"
        video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps) if frames else (False, None)
        video_record = {"video_path": video_path.name if video_ok else None,
                        "video_fps": args.video_fps, "video_frames": len(frames), "video_error": video_error}
    return {
        "condition": setting.name, "repeat_index": repeat_index, "seed": args.seed + repeat_index,
        "task_suite_name": args.task_suite_name, "task_id": args.task_id, "initial_state_id": args.initial_state_id,
        "initial_state_sha256": hashlib.sha256(np.asarray(initial_state, dtype=np.float64).tobytes()).hexdigest(),
        "prompt": prompt, "delay_target": setting.target, "delay_ms_requested": setting.delay_ms,
        "delay_steps": setting.delay_steps, "effective_delay_ms": setting.delay_steps / args.control_hz * 1000,
        "control_hz": actual_hz, "warmup_steps": warmup_steps, "outcome": outcome, "termination_reason": reason,
        "success": outcome == "success", "steps": len(actions), "n_samples": len(samples), "n_queries": len(queries),
        "n_delayed_queries": int(np.count_nonzero(query_delays)),
        "n_delayed_actions": int(np.count_nonzero(action_delays)),
        "trace_path": trace_path.name, **video_record,
        "object_names": [name for name, _ in objects], "unavailable_objects": missing_objects, "error": error_text,
    }


def save_summary(directory, args, metadata, settings, records):
    report = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "config": dataclasses.asdict(args),
        "server_metadata": metadata, "conditions": [dataclasses.asdict(setting) for setting in settings],
        "protocol": {
            "position": "measured robot0_eef_pos, world frame, metres",
            "quaternion": "robot0_eef_quat, xyzw", "objects": "world-frame root-body centres, metres",
            "sampling": "all control steps including initial and terminal observation; samples=actions+1",
            "time_origin": "after common warmup; source steps may be negative during initial history",
            "pairing": "same task, initial state, repeat seed and warmup; independent closed-loop rollouts",
            "noise": "PCG64 SeedSequence([seed, repeat_index, env_step]); same noise at matched steps",
            "endpoint": "infer_action via infer_with_noise; no activation diagnostics or consistency checks",
            "input_source_columns": [*_shift.CAMERA_KEYS, _shift.STATE_TARGET],
        },
        "records": records,
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    if records:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _run_rollouts(args):
    settings = make_settings(args)
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError("Unsupported task suite")
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError("task_id out of range")
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(init_states):
        raise ValueError("initial_state_id out of range")
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        metadata = client.get_server_metadata()
        shape = noise_shape_from_metadata(metadata)
        if not callable(getattr(client, "infer_with_noise", None)) or args.replan_steps > shape[0]:
            raise ValueError("Updated infer_with_noise client required; replan_steps must not exceed action horizon")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = pathlib.Path(args.output_dir) / stamp
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        max_delay = max(setting.delay_steps for setting in settings)
        warmup = max(args.num_steps_wait, max_delay)
        save_summary(directory, args, metadata, settings, records)
        logging.info("Saving paired trajectories to %s; %d conditions x %d repeats", directory,
                     len(settings), args.repeats_per_setting)
        for repeat in range(args.repeats_per_setting):
            order = np.random.default_rng(np.random.SeedSequence([args.seed, repeat, 987])).permutation(len(settings))
            for index in order:
                record = run_episode(
                    args=args, client=client, task=task, initial_state=init_states[args.initial_state_id],
                    prompt=prompt, setting=settings[index], repeat_index=repeat, noise_shape=shape,
                    warmup_steps=warmup, history_steps=max_delay,
                    max_steps=args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name], output_dir=directory,
                )
                records.append(record)
                save_summary(directory, args, metadata, settings, records)
                logging.info("[%d/%d] %s repeat=%d outcome=%s steps=%d", len(records),
                             len(settings) * args.repeats_per_setting, record["condition"], repeat,
                             record["outcome"], record["steps"])
                if record["outcome"] == "error":
                    raise RuntimeError(f"Stopped after error; partial results in {directory}: {record['error']}")
                if record["outcome"] == "interrupted":
                    return directory / "summary.json"
        return directory / "summary.json"
    finally:
        client._ws.close()


def run(args):
    """One entry point for collection followed by analysis of saved results."""
    make_settings(args)
    if args.skip_analysis:
        return _run_rollouts(args)
    try:
        from examples.libero import analyze_trajectory_delay as analysis
    except ImportError:
        import analyze_trajectory_delay as analysis
    # Fail before a long experiment if a requested plotting dependency is missing.
    try:
        __import__("matplotlib")
        if args.interactive:
            __import__("plotly.graph_objects")
    except ImportError as error:
        raise RuntimeError(
            "Automatic analysis requires matplotlib; --interactive also requires plotly. "
            "Install the missing package, or use --skip-analysis to collect data only."
        ) from error
    summary_path = _run_rollouts(args)
    logging.info("Rollouts saved to %s; generating trajectory figures...", summary_path)
    try:
        analysis_path = analysis.analyze(
            summary_path, threshold_cm=args.threshold_cm, consecutive=args.consecutive,
            pair_limit=args.pair_limit, interactive=args.interactive,
        )
    except Exception as error:
        raise RuntimeError(
            f"Rollout data are saved at {summary_path}, but automatic analysis failed. "
            "After fixing the analysis error, rerun analyze_trajectory_delay.py on that directory; "
            "the rollouts do not need to be repeated."
        ) from error
    logging.info("Analysis complete. Images and statistics: %s", analysis_path.parent)
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        if field.name == "save_video":
            parser.add_argument("--no-video", "--no_video", dest="save_video", action="store_false")
            continue
        default = getattr(defaults, field.name)
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=default)
            continue
        kind = int if field.name == "max_steps" else str if default is None else type(default)
        parser.add_argument(*flags, dest=field.name, type=kind, default=default)
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run(parse_args()))
