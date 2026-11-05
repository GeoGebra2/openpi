"""Measure PI0 robustness to delayed LIBERO camera observations.

Each rollout injects a configurable delay into one camera stream while keeping the
robot state and the other camera streams current. The delayed frames are selected
from a per-camera history, so this tests stale-image/replay behavior rather than
sleeping the whole control loop. A policy-input video and JSON/CSV result files are
written for every rollout. Videos put current camera frames in the top row and
the frames supplied to the policy in the bottom row.

The stock LIBERO Spatial observation exposes two policy images:
``observation/image`` and ``observation/wrist_image``. Custom camera mappings are
only meaningful when the environment, policy input transform, model image slots,
and checkpoint were all built for those views.
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

DEFAULT_CAMERA_KEYS = "observation/image,observation/wrist_image"
DEFAULT_CAMERA_SOURCES = "agentview_image,robot0_eye_in_hand_image"


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

    # Delay experiment. ``one_at_a_time`` creates one run per camera per delay.
    camera_keys: str = DEFAULT_CAMERA_KEYS
    camera_sources: str = DEFAULT_CAMERA_SOURCES
    sweep_mode: str = "one_at_a_time"
    delay_start_ms: int = 100
    delay_end_ms: int = 1000
    delay_step_ms: int = 100
    control_hz: float = 20.0
    repeats_per_setting: int = 5
    include_baseline: bool = True

    # Rollout and output.
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7
    video_fps: int = 10
    output_dir: str = "rollout/libero_spatial_task_000_image_delay"


@dataclasses.dataclass(frozen=True)
class DelaySetting:
    delayed_camera: str
    delay_ms: int


def run_delay_sweep(args: Args) -> pathlib.Path:
    """Run all requested delay settings and return the summary JSON path."""
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    _validate_args(args)
    camera_keys = _split_csv(args.camera_keys)
    camera_sources = _split_csv(args.camera_sources)
    if len(camera_keys) != len(camera_sources):
        raise ValueError(
            "camera_keys and camera_sources must contain the same number of comma-separated entries"
        )

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
    settings = _make_settings(args, camera_keys)
    warmup_steps = max(args.num_steps_wait, _delay_to_steps(args.delay_end_ms, args.control_hz))
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running %d delay settings", len(settings) * args.repeats_per_setting)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
    logging.info("Camera keys: %s", ", ".join(camera_keys))
    logging.info("Policy prompt: %s", policy_prompt)

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
                    camera_keys=camera_keys,
                    camera_sources=camera_sources,
                    setting=setting,
                    repeat_index=repeat_index,
                    run_index=run_index,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    output_dir=output_dir,
                )
                records.append(record)
                _write_summary(output_dir, args, records)
                logging.info(
                    "[%d/%d] camera=%s delay=%d ms outcome=%s steps=%s",
                    run_index,
                    len(settings) * args.repeats_per_setting,
                    setting.delayed_camera,
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
    camera_keys: list[str],
    camera_sources: list[str],
    setting: DelaySetting,
    repeat_index: int,
    run_index: int,
    max_steps: int,
    warmup_steps: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    delay_steps = _delay_to_steps(setting.delay_ms, args.control_hz)
    effective_delay_ms = round(delay_steps * 1000.0 / args.control_hz, 3)
    env = _create_env(task, args.seed)
    frames: list[np.ndarray] = []
    action_plan = collections.deque()
    history = {source: collections.deque(maxlen=delay_steps + 1) for source in camera_sources}
    outcome = "failure"
    error_text = None
    steps_completed = 0
    inference_rtt_ms: list[float] = []

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        _append_snapshot(history, obs, camera_sources, args.resize_size)
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            _append_snapshot(history, obs, camera_sources, args.resize_size)

        for step in range(max_steps):
            delayed_images = _select_delayed_images(
                history=history,
                camera_keys=camera_keys,
                camera_sources=camera_sources,
                delayed_camera=setting.delayed_camera,
                delay_steps=delay_steps,
            )
            current_images = {key: history[camera_sources[index]][-1] for index, key in enumerate(camera_keys)}
            frames.append(_make_video_frame(current_images, delayed_images, camera_keys))

            if not action_plan:
                observation = {
                    **delayed_images,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": policy_prompt,
                }
                infer_started = time.monotonic()
                response = client.infer(observation)
                inference_rtt_ms.append((time.monotonic() - infer_started) * 1000.0)
                action_chunk = response["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            _append_snapshot(history, obs, camera_sources, args.resize_size)
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
            "Delay run failed: camera=%s delay=%d ms", setting.delayed_camera, setting.delay_ms
        )
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    video_path = output_dir / (
        f"run_{run_index:03d}_{_slug(setting.delayed_camera)}_{setting.delay_ms:04d}ms_"
        f"repeat_{repeat_index:02d}_{outcome}.mp4"
    )
    if frames:
        import imageio  # noqa: PLC0415

        imageio.mimwrite(video_path, frames, fps=args.video_fps)

    return {
        "run_index": run_index,
        "repeat_index": repeat_index,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "initial_state_id": args.initial_state_id,
        "task_description": task_description,
        "policy_prompt": policy_prompt,
        "delayed_camera": setting.delayed_camera,
        "delay_ms_requested": setting.delay_ms,
        "delay_steps": delay_steps,
        "delay_ms_effective": effective_delay_ms,
        "control_hz": args.control_hz,
        "replan_steps": args.replan_steps,
        "outcome": outcome,
        "success": outcome == "success",
        "steps": steps_completed,
        "inference_calls": len(inference_rtt_ms),
        "inference_rtt_ms_mean": _stat(inference_rtt_ms, "mean"),
        "inference_rtt_ms_p95": _stat(inference_rtt_ms, "p95"),
        "inference_rtt_ms_max": _stat(inference_rtt_ms, "max"),
        "video_path": str(video_path) if frames else None,
        "error": error_text,
    }


def _append_snapshot(history, obs: dict, camera_sources: list[str], resize_size: int) -> None:
    from openpi_client import image_tools  # noqa: PLC0415

    for source in camera_sources:
        if source not in obs:
            raise KeyError(f"Camera source {source!r} is missing from LIBERO observation keys {sorted(obs)}")
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
        history[source].append(image)


def _select_delayed_images(
    *, history, camera_keys: list[str], camera_sources: list[str], delayed_camera: str, delay_steps: int
) -> dict[str, np.ndarray]:
    result = {}
    for index, key in enumerate(camera_keys):
        source = camera_sources[index]
        applied_delay = delay_steps if delayed_camera in (key, "__all__") else 0
        if len(history[source]) <= applied_delay:
            raise RuntimeError(
                f"Camera history for {source!r} has {len(history[source])} frames, "
                f"but delay requires {applied_delay + 1}"
            )
        result[key] = history[source][-1 - applied_delay]
    return result


def _make_settings(args: Args, camera_keys: list[str]) -> list[DelaySetting]:
    levels = list(range(args.delay_start_ms, args.delay_end_ms + 1, args.delay_step_ms))
    settings = [DelaySetting("__none__", 0)] if args.include_baseline else []
    if args.sweep_mode == "one_at_a_time":
        settings.extend(DelaySetting(camera, level) for camera in camera_keys for level in levels)
        return settings
    if args.sweep_mode == "all_cameras":
        settings.extend(DelaySetting("__all__", level) for level in levels)
        return settings
    raise ValueError("sweep_mode must be 'one_at_a_time' or 'all_cameras'")


def _write_summary(output_dir: pathlib.Path, args: Args, records: list[dict[str, Any]]) -> None:
    summary = {
        "experiment": "libero_image_delay_sweep",
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
        raise ValueError("camera list cannot be empty")
    if len(entries) != len(set(entries)):
        raise ValueError(f"camera list contains duplicates: {entries}")
    return entries


def _delay_to_steps(delay_ms: int, control_hz: float) -> int:
    return int(math.ceil(delay_ms * control_hz / 1000.0))


def _join_camera_frames(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def _make_video_frame(
    current_images: dict[str, np.ndarray], policy_images: dict[str, np.ndarray], camera_keys: list[str]
) -> np.ndarray:
    current_row = _join_camera_frames([current_images[key] for key in camera_keys])
    policy_row = _join_camera_frames([policy_images[key] for key in camera_keys])
    separator = np.full((4, current_row.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate((current_row, separator, policy_row), axis=0)


def _stat(values: list[float], name: str) -> Optional[float]:
    if not values:
        return None
    if name == "mean":
        return round(float(np.mean(values)), 3)
    if name == "p95":
        return round(float(np.percentile(values, 95)), 3)
    if name == "max":
        return round(float(np.max(values)), 3)
    raise ValueError(f"Unknown statistic {name!r}")


def _aggregate_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[(record["delayed_camera"], record["delay_ms_requested"])].append(record)
    aggregate = []
    for (camera, delay_ms), group in grouped.items():
        completed = [record for record in group if record["outcome"] in ("success", "failure")]
        aggregate.append(
            {
                "delayed_camera": camera,
                "delay_ms_requested": delay_ms,
                "runs": len(group),
                "completed_runs": len(completed),
                "successes": sum(record["success"] for record in completed),
                "success_rate": (
                    round(sum(record["success"] for record in completed) / len(completed), 4) if completed else None
                ),
            }
        )
    return aggregate


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "camera"


def _quat2axisangle(quat):
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
    if args.delay_start_ms < 0 or args.delay_end_ms < args.delay_start_ms or args.delay_step_ms <= 0:
        raise ValueError("delay range must be non-negative, increasing, and have a positive step")
    if args.control_hz <= 0:
        raise ValueError("control_hz must be positive")
    if args.repeats_per_setting <= 0:
        raise ValueError("repeats_per_setting must be positive")
    if (args.delay_end_ms - args.delay_start_ms) % args.delay_step_ms:
        raise ValueError("delay_end_ms must be reachable from delay_start_ms using delay_step_ms")


if __name__ == "__main__":
    import tyro  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    run_delay_sweep(tyro.cli(Args))
