"""Measure PI0 robustness to delayed LIBERO self-state observations.

Each rollout keeps both LIBERO camera images current but supplies PI0 with a
historical eight-dimensional state at every policy query. The delayed state is
selected from a per-rollout history rather than sleeping the simulator. Each
rollout writes a video, and the sweep writes JSON/CSV summaries after every run.
The video shows the current camera views and an annotation panel containing the
current state, policy-input state, and their mismatch.
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

    # State-delay experiment. Values are quantized to simulator control steps.
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
    output_dir: str = "rollout/libero_spatial_task_000_state_delay"


@dataclasses.dataclass(frozen=True)
class DelaySetting:
    delay_ms: int


def run_delay_sweep(args: Args) -> pathlib.Path:
    """Run all requested state-delay settings and return the summary JSON path."""
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    _validate_args(args)
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
    settings = _make_settings(args)
    warmup_steps = max(args.num_steps_wait, _delay_to_steps(args.delay_end_ms, args.control_hz))
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(settings) * args.repeats_per_setting
    logging.info("Running %d state-delay rollouts", total_runs)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
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
                    "[%d/%d] state delay=%d ms outcome=%s steps=%s",
                    run_index,
                    total_runs,
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
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    delay_steps = _delay_to_steps(setting.delay_ms, args.control_hz)
    effective_delay_ms = round(delay_steps * 1000.0 / args.control_hz, 3)
    env = _create_env(task, args.seed)
    frames: list[np.ndarray] = []
    action_plan = collections.deque()
    image_history = {
        "agentview_image": collections.deque(maxlen=1),
        "robot0_eye_in_hand_image": collections.deque(maxlen=1),
    }
    state_history = collections.deque(maxlen=delay_steps + 1)
    outcome = "failure"
    error_text = None
    steps_completed = 0
    inference_rtt_ms: list[float] = []
    state_delta_l2: list[float] = []
    state_delta_max_abs: list[float] = []
    position_delta_l2: list[float] = []
    rotation_delta_l2: list[float] = []
    gripper_delta_l2: list[float] = []

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        _append_snapshot(image_history, state_history, obs, args.resize_size)
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            _append_snapshot(image_history, state_history, obs, args.resize_size)

        for step in range(max_steps):
            current_state = state_history[-1]
            policy_state = _select_delayed_state(state_history, delay_steps)
            delta = current_state - policy_state
            current_images = {
                "observation/image": image_history["agentview_image"][-1],
                "observation/wrist_image": image_history["robot0_eye_in_hand_image"][-1],
            }
            will_query_policy = not action_plan
            frames.append(
                _make_video_frame(
                    current_images=current_images,
                    current_state=current_state,
                    policy_state=policy_state,
                    effective_delay_ms=effective_delay_ms,
                    step=step,
                    policy_query=will_query_policy,
                )
            )

            if will_query_policy:
                observation = _build_policy_observation(current_images, policy_state, policy_prompt)
                infer_started = time.monotonic()
                response = client.infer(observation)
                inference_rtt_ms.append((time.monotonic() - infer_started) * 1000.0)
                state_delta_l2.append(float(np.linalg.norm(delta)))
                state_delta_max_abs.append(float(np.max(np.abs(delta))))
                position_delta_l2.append(float(np.linalg.norm(delta[:3])))
                rotation_delta_l2.append(float(np.linalg.norm(delta[3:6])))
                gripper_delta_l2.append(float(np.linalg.norm(delta[6:])))
                action_chunk = response["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
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
        logging.exception("State-delay run failed: delay=%d ms", setting.delay_ms)
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    video_path = output_dir / (
        f"run_{run_index:03d}_state_{setting.delay_ms:04d}ms_repeat_{repeat_index:02d}_{outcome}.mp4"
    )
    video_written, video_error = _save_video(video_path, frames, args.video_fps)

    return {
        "run_index": run_index,
        "repeat_index": repeat_index,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "initial_state_id": args.initial_state_id,
        "task_description": task_description,
        "policy_prompt": policy_prompt,
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
        "state_delta_l2_mean": _stat(state_delta_l2, "mean"),
        "state_delta_l2_p95": _stat(state_delta_l2, "p95"),
        "state_delta_l2_max": _stat(state_delta_l2, "max"),
        "state_delta_max_abs_mean": _stat(state_delta_max_abs, "mean"),
        "state_delta_max_abs_max": _stat(state_delta_max_abs, "max"),
        "position_delta_l2_mean": _stat(position_delta_l2, "mean"),
        "position_delta_l2_max": _stat(position_delta_l2, "max"),
        "axis_angle_delta_l2_mean": _stat(rotation_delta_l2, "mean"),
        "axis_angle_delta_l2_max": _stat(rotation_delta_l2, "max"),
        "gripper_delta_l2_mean": _stat(gripper_delta_l2, "mean"),
        "gripper_delta_l2_max": _stat(gripper_delta_l2, "max"),
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


def _build_policy_observation(
    current_images: dict[str, np.ndarray], policy_state: np.ndarray, prompt: str
) -> dict[str, Any]:
    return {
        **current_images,
        "observation/state": policy_state,
        "prompt": prompt,
    }


def _make_settings(args: Args) -> list[DelaySetting]:
    levels = list(range(args.delay_start_ms, args.delay_end_ms + 1, args.delay_step_ms))
    return ([DelaySetting(0)] if args.include_baseline else []) + [DelaySetting(level) for level in levels]


def _write_summary(output_dir: pathlib.Path, args: Args, records: list[dict[str, Any]]) -> None:
    summary = {
        "experiment": "libero_state_delay_sweep",
        "config": dataclasses.asdict(args),
        "state_definition": [
            "robot0_eef_pos[0:3]",
            "robot0_eef_quat converted to axis-angle[0:3]",
            "robot0_gripper_qpos[0:2]",
        ],
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
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[record["delay_ms_requested"]].append(record)
    aggregate = []
    for delay_ms, group in grouped.items():
        completed = [record for record in group if record["outcome"] in ("success", "failure")]
        successes = sum(record["success"] for record in completed)
        failures = len(completed) - successes
        aggregate.append(
            {
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


def _delay_to_steps(delay_ms: int, control_hz: float) -> int:
    return int(math.ceil(delay_ms * control_hz / 1000.0))


def _join_camera_frames(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]


def _make_video_frame(
    *,
    current_images: dict[str, np.ndarray],
    current_state: np.ndarray,
    policy_state: np.ndarray,
    effective_delay_ms: float,
    step: int,
    policy_query: bool,
) -> np.ndarray:
    """Create a current-view frame with a readable state-mismatch panel."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    image_row = _join_camera_frames(
        [current_images["observation/image"], current_images["observation/wrist_image"]]
    )
    delta = current_state - policy_state
    panel_height = 122
    panel = np.full((panel_height, image_row.shape[1], 3), 245, dtype=np.uint8)
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
    return np.concatenate((image_row, np.asarray(panel_image)), axis=0)


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
