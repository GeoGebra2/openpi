"""Test the time-shift-matching hypothesis for PI0 under delayed LIBERO observations.

PI0 sees only single-frame input, so a delayed observation is indistinguishable
from "the world is actually at the state it was in d ms ago". If the policy
simply replays a stale plan, the action chunk produced under delay d at time t
should equal the chunk produced at t-d with the then-current observation.

Each rollout therefore performs TWO inferences per policy-relevant moment:

1. Delayed query (every ``replan_steps`` env steps, when the action plan is
   empty): the observation is built with the delayed input (one stale camera,
   a stale state, or — with the ``__all__`` target — all cameras and the state
   delayed together, the strict control for the pure-replay hypothesis); the
   first ``replan_steps`` actions of the returned chunk drive the robot,
   exactly like the failure-rate delay sweeps.
2. Shadow query (EVERY env step): the observation is built with fully current
   images and current state; the returned chunk is recorded but never executed.
   Recording shadow chunks at every step lets the analysis compare the delayed
   chunk at step t against the shadow chunk at step t-k for arbitrary shift k.

Every run writes a ``*_chunks.npz`` with the per-step shadow chunks and the
per-replan delayed chunks, a policy-input video, and the sweep writes JSON/CSV
summaries after every run. Analyze the results with ``analyze_shift_match.py``.

WARNING: shadow queries run every env step, so an episode costs roughly
``max_steps / replan_steps + max_steps`` inferences — an order of magnitude
slower than the failure-rate sweeps. Use small ``repeats_per_setting`` and,
while iterating, a small ``--max-steps``.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import datetime
import json
import logging
import math
import pathlib
import re
import time
from typing import Any, Optional

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

CAMERA_KEYS = ("observation/image", "observation/wrist_image")
CAMERA_SOURCES = ("agentview_image", "robot0_eye_in_hand_image")
STATE_TARGET = "state"
ALL_TARGET = "__all__"
BASELINE_TARGET = "__none__"
VALID_DELAY_TARGETS = (*CAMERA_KEYS, STATE_TARGET, ALL_TARGET)


@dataclasses.dataclass
class Args:
    # Policy server.
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO task selection.
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None

    # Shift-match experiment. Values are quantized to simulator control steps.
    delay_targets: str = "observation/image,observation/wrist_image,state"
    delay_levels_ms: str = "100,200,400,800"
    control_hz: float = 20.0
    repeats_per_setting: int = 3
    include_baseline: bool = True

    # Rollout and output.
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7
    video_fps: int = 10
    output_dir: str = "rollout/libero_spatial_task_000_shift_match"


@dataclasses.dataclass(frozen=True)
class DelaySetting:
    delay_target: str
    delay_ms: int


def run_shift_match_sweep(args: Args) -> pathlib.Path:
    """Run all requested shift-match settings and return the summary JSON path."""
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    _validate_args(args)
    delay_targets = _split_csv(args.delay_targets)
    delay_levels = _parse_delay_levels(args.delay_levels_ms)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite {args.task_suite_name!r}; available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[args.task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}], got {args.task_id}")
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(
            f"initial_state_id must be in [0, {len(initial_states) - 1}], got {args.initial_state_id}"
        )

    task = task_suite.get_task(args.task_id)
    task_description = str(task.language)
    policy_prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else task_description
    max_steps = args.max_steps or MAX_STEPS_BY_SUITE[args.task_suite_name]
    settings = _make_settings(args, delay_targets, delay_levels)
    delay_steps_max = _delay_to_steps(max(delay_levels), args.control_hz)
    warmup_steps = max(args.num_steps_wait, delay_steps_max)
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(settings) * args.repeats_per_setting
    logging.info("Running %d shift-match rollouts", total_runs)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
    logging.info("Delay targets: %s", ", ".join(delay_targets))
    logging.info("Policy prompt: %s", policy_prompt)
    logging.warning(
        "Shadow queries run every step: each episode costs roughly %d inferences",
        max_steps // args.replan_steps + max_steps,
    )

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    records: list[dict[str, Any]] = []
    run_index = 0
    try:
        for setting in settings:
            for repeat_index in range(args.repeats_per_setting):
                run_index += 1
                record = _run_episode(
                    args=args,
                    client=client,
                    task=task,
                    initial_state=initial_states[args.initial_state_id],
                    task_description=task_description,
                    policy_prompt=policy_prompt,
                    setting=setting,
                    repeat_index=repeat_index,
                    run_index=run_index,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    delay_steps_max=delay_steps_max,
                    output_dir=output_dir,
                )
                records.append(record)
                _write_summary(output_dir, args, records)
                logging.info(
                    "[%d/%d] target=%s delay=%d ms outcome=%s steps=%s",
                    run_index,
                    total_runs,
                    setting.delay_target,
                    setting.delay_ms,
                    record["outcome"],
                    record["steps"],
                )
                if record["outcome"] == "interrupted":
                    logging.warning("Sweep interrupted after %d completed runs", len(records))
                    return output_dir / "summary.json"
    except KeyboardInterrupt:
        logging.warning("Sweep interrupted after %d completed runs", len(records))
        _write_summary(output_dir, args, records)

    summary_path = output_dir / "summary.json"
    logging.info("Saved experiment summary: %s", summary_path)
    return summary_path


def _run_episode(
    *,
    args: Args,
    client: Any,
    task,
    initial_state,
    task_description: str,
    policy_prompt: str,
    setting: DelaySetting,
    repeat_index: int,
    run_index: int,
    max_steps: int,
    warmup_steps: int,
    delay_steps_max: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    delay_steps = _delay_to_steps(setting.delay_ms, args.control_hz)
    effective_delay_ms = round(delay_steps * 1000.0 / args.control_hz, 3)
    env = _create_env(task, args.seed)
    frames: list[np.ndarray] = []
    action_plan = collections.deque()
    image_history = {source: collections.deque(maxlen=delay_steps_max + 1) for source in CAMERA_SOURCES}
    state_history = collections.deque(maxlen=delay_steps_max + 1)
    outcome = "failure"
    error_text = None
    steps_completed = 0
    delayed_rtt_ms: list[float] = []
    shadow_rtt_ms: list[float] = []
    step_indices: list[int] = []
    replan_step_indices: list[int] = []
    shadow_chunks: list[np.ndarray] = []
    delayed_chunks: list[np.ndarray] = []
    chunk_shape: Optional[tuple[int, ...]] = None

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        _append_snapshot(image_history, state_history, obs, args.resize_size)
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            _append_snapshot(image_history, state_history, obs, args.resize_size)

        for step in range(max_steps):
            current_images = {key: image_history[source][-1] for key, source in zip(CAMERA_KEYS, CAMERA_SOURCES)}
            current_state = np.array(state_history[-1], copy=True)
            policy_observation = _build_delayed_observation(
                image_history=image_history,
                state_history=state_history,
                delay_target=setting.delay_target,
                delay_steps=delay_steps,
                prompt=policy_prompt,
            )
            will_query_policy = not action_plan
            frames.append(
                _make_video_frame(
                    current_images=current_images,
                    policy_images={key: policy_observation[key] for key in CAMERA_KEYS},
                    current_state=current_state,
                    policy_state=policy_observation["observation/state"],
                    delay_target=setting.delay_target,
                    effective_delay_ms=effective_delay_ms,
                    step=step,
                    policy_query=will_query_policy,
                )
            )

            # Shadow query: fully current observation, never executed.
            shadow_observation = {
                **current_images,
                "observation/state": current_state,
                "prompt": policy_prompt,
            }
            infer_started = time.monotonic()
            shadow_response = client.infer(shadow_observation)
            shadow_rtt_ms.append((time.monotonic() - infer_started) * 1000.0)
            shadow_chunk = np.asarray(shadow_response["actions"], dtype=np.float32)
            if chunk_shape is None:
                chunk_shape = shadow_chunk.shape
            elif shadow_chunk.shape != chunk_shape:
                raise ValueError(
                    f"Policy returned chunk with shape {shadow_chunk.shape}, expected {chunk_shape}"
                )
            step_indices.append(step)
            shadow_chunks.append(shadow_chunk)

            if will_query_policy:
                infer_started = time.monotonic()
                response = client.infer(policy_observation)
                delayed_rtt_ms.append((time.monotonic() - infer_started) * 1000.0)
                action_chunk = np.asarray(response["actions"], dtype=np.float32)
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
                if action_chunk.shape != chunk_shape:
                    raise ValueError(
                        f"Policy returned chunk with shape {action_chunk.shape}, expected {chunk_shape}"
                    )
                replan_step_indices.append(step)
                delayed_chunks.append(action_chunk)
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            _append_snapshot(image_history, state_history, obs, args.resize_size)
            steps_completed = step + 1
            if done:
                outcome = "success"
                break
    except KeyboardInterrupt:
        outcome = "interrupted"
        error_text = "KeyboardInterrupt"
    except Exception as error:  # Keep the sweep running after one bad setting.
        outcome = "error"
        error_text = repr(error)
        logging.exception(
            "Shift-match run failed: target=%s delay=%d ms", setting.delay_target, setting.delay_ms
        )
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    run_slug = (
        f"run_{run_index:03d}_{_slug(setting.delay_target)}_{setting.delay_ms:04d}ms_"
        f"repeat_{repeat_index:02d}"
    )
    chunks_path: Optional[pathlib.Path] = None
    if shadow_chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = output_dir / f"{run_slug}_chunks.npz"
        np.savez_compressed(
            chunks_path,
            steps=np.asarray(step_indices, dtype=np.int32),
            shadow_chunks=np.stack(shadow_chunks).astype(np.float32, copy=False),
            replan_step_indices=np.asarray(replan_step_indices, dtype=np.int32),
            delayed_chunks=(
                np.stack(delayed_chunks).astype(np.float32, copy=False)
                if delayed_chunks
                else np.zeros((0, *chunk_shape), dtype=np.float32)
            ),
        )

    video_path = output_dir / f"{run_slug}_{outcome}.mp4"
    video_written, video_error = _save_video(video_path, frames, args.video_fps)

    return {
        "run_index": run_index,
        "repeat_index": repeat_index,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "initial_state_id": args.initial_state_id,
        "task_description": task_description,
        "policy_prompt": policy_prompt,
        "delay_target": setting.delay_target,
        "delay_ms_requested": setting.delay_ms,
        "delay_steps": delay_steps,
        "delay_ms_effective": effective_delay_ms,
        "control_hz": args.control_hz,
        "replan_steps": args.replan_steps,
        "outcome": outcome,
        "success": outcome == "success",
        "steps": steps_completed,
        "n_replan_queries": len(delayed_chunks),
        "n_shadow_queries": len(shadow_chunks),
        "delayed_rtt_ms_mean": _stat(delayed_rtt_ms, "mean"),
        "delayed_rtt_ms_max": _stat(delayed_rtt_ms, "max"),
        "shadow_rtt_ms_mean": _stat(shadow_rtt_ms, "mean"),
        "shadow_rtt_ms_max": _stat(shadow_rtt_ms, "max"),
        "chunks_path": str(chunks_path) if chunks_path else None,
        "video_path": str(video_path) if video_written else None,
        "video_error": video_error,
        "error": error_text,
    }


def _append_snapshot(image_history, state_history, obs: dict, resize_size: int) -> None:
    from openpi_client import image_tools  # noqa: PLC0415

    for source in image_history:
        if source not in obs:
            raise KeyError(f"Camera source {source!r} is missing from LIBERO observation keys {sorted(obs)}")
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
        image_history[source].append(image)
    state_history.append(_build_state(obs))


def _build_state(obs: dict) -> np.ndarray:
    """Build LIBERO's eight-dimensional state in the policy's expected order."""
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32, copy=False)


def _select_delayed_state(history: collections.deque, delay_steps: int) -> np.ndarray:
    if len(history) <= delay_steps:
        raise RuntimeError(
            f"State history has {len(history)} frames, but delay requires {delay_steps + 1}"
        )
    return np.array(history[-1 - delay_steps], copy=True)


def _build_delayed_observation(
    *,
    image_history,
    state_history: collections.deque,
    delay_target: str,
    delay_steps: int,
    prompt: str,
) -> dict[str, Any]:
    """Build the delayed-query observation for one delay target.

    Camera targets: the targeted camera's frame comes from ``delay_steps`` back
    in its history while the other camera and the state stay current. The
    "state" target keeps both cameras current and uses a historical state. The
    "__all__" target delays both cameras AND the state together, so the delayed
    query input at step t is identical to the shadow query input at step
    t-delay_steps — the strict control for the pure-replay hypothesis. The
    baseline target ("__none__", zero delay) leaves everything current.
    """
    images = {}
    for key, source in zip(CAMERA_KEYS, CAMERA_SOURCES):
        applied_delay = delay_steps if delay_target in (key, ALL_TARGET) else 0
        if len(image_history[source]) <= applied_delay:
            raise RuntimeError(
                f"Camera history for {source!r} has {len(image_history[source])} frames, "
                f"but delay requires {applied_delay + 1}"
            )
        images[key] = image_history[source][-1 - applied_delay]
    if delay_target in (STATE_TARGET, ALL_TARGET):
        state = _select_delayed_state(state_history, delay_steps)
    else:
        state = np.array(state_history[-1], copy=True)
    return {
        **images,
        "observation/state": state,
        "prompt": prompt,
    }


def _make_settings(args: Args, delay_targets: list[str], delay_levels: list[int]) -> list[DelaySetting]:
    settings = [DelaySetting(BASELINE_TARGET, 0)] if args.include_baseline else []
    settings.extend(DelaySetting(target, level) for target in delay_targets for level in delay_levels)
    return settings


def _write_summary(output_dir: pathlib.Path, args: Args, records: list[dict[str, Any]]) -> None:
    summary = {
        "experiment": "libero_shift_match_sweep",
        "config": dataclasses.asdict(args),
        "aggregate": _aggregate_results(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not records:
        return
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _aggregate_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[(record["delay_target"], record["delay_ms_requested"])].append(record)
    aggregate = []
    for (target, delay_ms), group in grouped.items():
        completed = [record for record in group if record["outcome"] in ("success", "failure")]
        successes = sum(record["success"] for record in completed)
        failures = len(completed) - successes
        aggregate.append(
            {
                "delay_target": target,
                "delay_ms_requested": delay_ms,
                "runs": len(group),
                "completed_runs": len(completed),
                "successes": successes,
                "failures": failures,
                "success_rate": round(successes / len(completed), 4) if completed else None,
                "failure_rate": round(failures / len(completed), 4) if completed else None,
            }
        )
    return aggregate


def _create_env(task, seed: int):
    from libero.libero import get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _split_csv(value: str) -> list[str]:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        raise ValueError("delay target list cannot be empty")
    if len(entries) != len(set(entries)):
        raise ValueError(f"delay target list contains duplicates: {entries}")
    return entries


def _parse_delay_levels(value: str) -> list[int]:
    entries = _split_csv(value)
    try:
        levels = [int(entry) for entry in entries]
    except ValueError:
        raise ValueError(f"delay_levels_ms must be a comma-separated list of integers, got {value!r}") from None
    if any(level < 0 for level in levels):
        raise ValueError(f"delay levels must be non-negative, got {levels}")
    if len(levels) != len(set(levels)):
        raise ValueError(f"delay levels contain duplicates: {levels}")
    return sorted(levels)


def _delay_to_steps(delay_ms: int, control_hz: float) -> int:
    return int(math.ceil(delay_ms * control_hz / 1000.0))


def _join_camera_frames(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def _make_video_frame(
    *,
    current_images: dict[str, np.ndarray],
    policy_images: dict[str, np.ndarray],
    current_state: np.ndarray,
    policy_state: np.ndarray,
    delay_target: str,
    effective_delay_ms: float,
    step: int,
    policy_query: bool,
) -> np.ndarray:
    """Create a video frame for the current delay target.

    Camera-delay targets (and the baseline) show current camera frames in the
    top row and the policy-input frames in the bottom row. The state target
    shows the current camera views plus an annotated state-mismatch panel. The
    "__all__" target shows the camera comparison plus the state panel.
    """
    if delay_target == STATE_TARGET:
        return _make_state_video_frame(
            current_images=current_images,
            current_state=current_state,
            policy_state=policy_state,
            effective_delay_ms=effective_delay_ms,
            step=step,
            policy_query=policy_query,
        )
    current_row = _join_camera_frames([current_images[key] for key in CAMERA_KEYS])
    policy_row = _join_camera_frames([policy_images[key] for key in CAMERA_KEYS])
    separator = np.full((4, current_row.shape[1], 3), 255, dtype=np.uint8)
    camera_frame = np.concatenate((current_row, separator, policy_row), axis=0)
    if delay_target != ALL_TARGET:
        return camera_frame
    panel = _state_mismatch_panel(
        width=camera_frame.shape[1],
        current_state=current_state,
        policy_state=policy_state,
        effective_delay_ms=effective_delay_ms,
        step=step,
        policy_query=policy_query,
    )
    return np.concatenate((camera_frame, panel), axis=0)


def _make_state_video_frame(
    *,
    current_images: dict[str, np.ndarray],
    current_state: np.ndarray,
    policy_state: np.ndarray,
    effective_delay_ms: float,
    step: int,
    policy_query: bool,
) -> np.ndarray:
    """Create a current-view frame with a readable state-mismatch panel."""
    image_row = _join_camera_frames([current_images[key] for key in CAMERA_KEYS])
    panel = _state_mismatch_panel(
        width=image_row.shape[1],
        current_state=current_state,
        policy_state=policy_state,
        effective_delay_ms=effective_delay_ms,
        step=step,
        policy_query=policy_query,
    )
    return np.concatenate((image_row, panel), axis=0)


def _state_mismatch_panel(
    *,
    width: int,
    current_state: np.ndarray,
    policy_state: np.ndarray,
    effective_delay_ms: float,
    step: int,
    policy_query: bool,
) -> np.ndarray:
    """Render the annotated current-vs-policy state panel as an image row."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    delta = current_state - policy_state
    panel_height = 122
    panel = np.full((panel_height, width, 3), 245, dtype=np.uint8)
    panel_image = Image.fromarray(panel)
    draw = ImageDraw.Draw(panel_image)
    draw.text(
        (6, 6),
        f"state delay={effective_delay_ms:g} ms | step={step} | query={'YES' if policy_query else 'no'} | "
        f"delta_l2={np.linalg.norm(delta):.4f} | "
        f"delta_max={np.max(np.abs(delta)):.4f}",
        fill=(20, 20, 20),
    )
    draw.text(
        (6, 34),
        "current: " + np.array2string(current_state, precision=3, suppress_small=True, separator=", "),
        fill=(20, 20, 20),
    )
    draw.text(
        (6, 62),
        "policy : " + np.array2string(policy_state, precision=3, suppress_small=True, separator=", "),
        fill=(20, 20, 20),
    )
    draw.text(
        (6, 90),
        "delta  : " + np.array2string(delta, precision=3, suppress_small=True, separator=", "),
        fill=(20, 20, 20),
    )
    return np.asarray(panel_image)


def _stat(values: list[float], name: str) -> Optional[float]:
    if not values:
        return None
    if name == "mean":
        return round(float(np.mean(values)), 3)
    if name == "max":
        return round(float(np.max(values)), 3)
    raise ValueError(f"Unknown statistic {name!r}")


def _save_video(video_path: pathlib.Path, frames: list[np.ndarray], fps: int) -> tuple[bool, Optional[str]]:
    if not frames:
        return False, None
    try:
        import imageio  # noqa: PLC0415

        imageio.mimwrite(video_path, frames, fps=fps)
    except Exception as error:
        logging.exception("Failed to save rollout video: %s", video_path)
        return False, repr(error)
    return True, None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "target"


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def _validate_args(args: Args) -> None:
    if args.task_suite_name not in MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unsupported task suite {args.task_suite_name!r}")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if args.resize_size <= 0 or args.video_fps <= 0:
        raise ValueError("resize_size and video_fps must be positive")
    unknown_targets = set(_split_csv(args.delay_targets)) - set(VALID_DELAY_TARGETS)
    if unknown_targets:
        raise ValueError(
            f"Unknown delay targets {sorted(unknown_targets)}; valid targets: {list(VALID_DELAY_TARGETS)}"
        )
    _parse_delay_levels(args.delay_levels_ms)
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if args.repeats_per_setting <= 0:
        raise ValueError("repeats_per_setting must be positive")


if __name__ == "__main__":
    import tyro  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    run_shift_match_sweep(tyro.cli(Args))
