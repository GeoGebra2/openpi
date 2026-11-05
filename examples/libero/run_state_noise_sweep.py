"""Control experiment: is delay failure caused by temporal misalignment or perturbation magnitude?

Two hypotheses explain why observation delay kills PI0 task performance on
LIBERO:

- Hypothesis A (input perturbation magnitude): the model is simply sensitive to
  input accuracy. It does not matter whether the input state is wrong because
  it is stale (delay) or corrupted (noise); the delay failure-rate curve is a
  generic perturbation-magnitude-vs-performance curve with no temporal
  mechanism.
- Hypothesis B (temporal misalignment): a delayed observation is a perfectly
  legal, self-consistent, in-distribution observation of a *past* world state.
  The model has no cue to distrust it and confidently plans a coherent action
  chunk for the wrong world. Random noise, by contrast, is out-of-distribution
  and may be partially detected as anomalous.

This script is the control arm. Each rollout keeps both LIBERO camera images
current but adds fresh iid isotropic Gaussian noise to the eight-dimensional
self-state at every policy query. The per-group L2 magnitude of the noise
(position / axis-angle rotation / gripper) is calibrated to EXACTLY match the
mean delay-induced state delta measured by ``run_state_delay_sweep.py`` at the
same delay level (read from that sweep's summary.json). If noise is ~harmless
while delay kills at the same perturbation magnitude, temporal misalignment
(Hypothesis B) is the culprit.

Baseline (delay 0) runs are clean (zero noise), comparable to the delay
sweep's baseline. Each rollout writes a video with a state-mismatch panel, and
the sweep writes JSON/CSV summaries after every run.
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
STATE_DELAY_EXPERIMENT = "libero_state_delay_sweep"
# (group name, start dim, end dim) slices of the 8-dim LIBERO state.
NOISE_GROUPS = (("position", 0, 3), ("rotation", 3, 6), ("gripper", 6, 8))
# State-delay record fields averaged to calibrate each group's noise magnitude.
CALIBRATION_FIELDS = {
    "position": "position_delta_l2_mean",
    "rotation": "axis_angle_delta_l2_mean",
    "gripper": "gripper_delta_l2_mean",
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

    # State-noise experiment. Levels mirror the state-delay sweep so that the
    # calibrated noise magnitude at each level matches that sweep's state delta.
    delay_start_ms: int = 100
    delay_end_ms: int = 1000
    delay_step_ms: int = 100
    control_hz: float = 20.0
    repeats_per_setting: int = 5
    include_baseline: bool = True
    calibration_summary: str = ""
    noise_seed: int = 0

    # Rollout and output.
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7
    video_fps: int = 10
    output_dir: str = "rollout/libero_spatial_task_000_state_noise"


@dataclasses.dataclass(frozen=True)
class DelaySetting:
    delay_ms: int


def run_noise_sweep(args: Args) -> pathlib.Path:
    """Run all requested state-noise settings and return the summary JSON path."""
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    _validate_args(args)
    calibration = _load_calibration(args.calibration_summary)
    settings = _make_settings(args)
    targets_by_delay = _targets_for_settings(settings, calibration)
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
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(settings) * args.repeats_per_setting
    logging.info("Running %d state-noise rollouts", total_runs)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
    logging.info("Policy prompt: %s", policy_prompt)
    logging.info("Noise calibration from %s: %s", args.calibration_summary, targets_by_delay)

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
                    targets=targets_by_delay[setting.delay_ms],
                    repeat_index=repeat_index,
                    run_index=run_index,
                    max_steps=max_steps,
                    output_dir=output_dir,
                )
                records.append(record)
                _write_summary(output_dir, args, records, targets_by_delay)
                logging.info(
                    "[%d/%d] state noise (delay level)=%d ms outcome=%s steps=%s",
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
        _write_summary(output_dir, args, records, targets_by_delay)

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
    targets: dict[str, float],
    repeat_index: int,
    run_index: int,
    max_steps: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    delay_steps = _delay_to_steps(setting.delay_ms, args.control_hz)
    effective_delay_ms = round(delay_steps * 1000.0 / args.control_hz, 3)
    rng = np.random.default_rng(args.noise_seed * 100003 + run_index)
    env = _create_env(task, args.seed)
    frames: list[np.ndarray] = []
    action_plan = collections.deque()
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
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        current_images, current_state = _extract_snapshot(obs, args.resize_size)
        last_policy_state = current_state

        for step in range(max_steps):
            will_query_policy = not action_plan
            if will_query_policy:
                policy_state = current_state + _make_noise(rng, targets)
                last_policy_state = policy_state
            frames.append(
                _make_video_frame(
                    current_images=current_images,
                    current_state=current_state,
                    policy_state=last_policy_state,
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
                noise = policy_state - current_state
                state_delta_l2.append(float(np.linalg.norm(noise)))
                state_delta_max_abs.append(float(np.max(np.abs(noise))))
                position_delta_l2.append(float(np.linalg.norm(noise[:3])))
                rotation_delta_l2.append(float(np.linalg.norm(noise[3:6])))
                gripper_delta_l2.append(float(np.linalg.norm(noise[6:])))
                action_chunk = response["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            current_images, current_state = _extract_snapshot(obs, args.resize_size)
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
        logging.exception("State-noise run failed: delay level=%d ms", setting.delay_ms)
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    video_path = output_dir / (
        f"run_{run_index:03d}_statenoise_{setting.delay_ms:04d}ms_repeat_{repeat_index:02d}_{outcome}.mp4"
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
        "noise_targets": targets,
        "calibration_summary": str(args.calibration_summary),
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


def _load_calibration(path: str | pathlib.Path) -> dict[int, dict[str, float]]:
    """Average per-group delay-induced state deltas from a state-delay summary.

    Returns {delay_ms: {"position": x, "rotation": y, "gripper": z}}. Records
    missing any of the group statistics are skipped.
    """
    calibration_path = pathlib.Path(path)
    if not calibration_path.is_file():
        raise ValueError(f"Calibration summary not found: {calibration_path}")
    try:
        summary = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse calibration summary {calibration_path}: {error}") from error
    experiment = summary.get("experiment")
    if experiment != STATE_DELAY_EXPERIMENT:
        raise ValueError(
            f"Calibration summary {calibration_path} has experiment {experiment!r}; "
            f"expected {STATE_DELAY_EXPERIMENT!r}"
        )
    grouped: dict[int, list[list[float]]] = collections.defaultdict(list)
    for record in summary.get("records", []):
        values = [record.get(field) for field in CALIBRATION_FIELDS.values()]
        if any(value is None for value in values):
            continue
        grouped[int(record["delay_ms_requested"])].append([float(value) for value in values])
    calibration: dict[int, dict[str, float]] = {}
    for delay_ms, samples in grouped.items():
        means = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
        calibration[delay_ms] = {group: float(value) for group, value in zip(CALIBRATION_FIELDS, means)}
    if not calibration:
        raise ValueError(f"Calibration summary {calibration_path} contains no usable records")
    return calibration


def _targets_for_settings(
    settings: list[DelaySetting], calibration: dict[int, dict[str, float]]
) -> dict[int, dict[str, float]]:
    """Resolve noise targets per level; the delay-0 baseline is always clean."""
    zero_targets = {"position": 0.0, "rotation": 0.0, "gripper": 0.0}
    targets_by_delay: dict[int, dict[str, float]] = {}
    for setting in settings:
        if setting.delay_ms == 0:
            targets_by_delay[0] = dict(zero_targets)
            continue
        if setting.delay_ms not in calibration:
            raise ValueError(
                f"No calibration data for delay {setting.delay_ms} ms; "
                f"available levels: {sorted(calibration)}"
            )
        targets_by_delay[setting.delay_ms] = calibration[setting.delay_ms]
    return targets_by_delay


def _make_noise(rng: np.random.Generator, targets: dict[str, float]) -> np.ndarray:
    """Sample 8-dim noise whose per-group L2 norm equals the target exactly.

    Each group draws an isotropic Gaussian direction g ~ N(0, I_dim) and is
    scaled to ``target_l2 * g / ||g||``; a zero target yields an exact zero
    group.
    """
    noise = np.zeros(8, dtype=np.float32)
    for group, start, end in NOISE_GROUPS:
        target = float(targets.get(group, 0.0))
        if target == 0.0:
            continue
        direction = rng.standard_normal(end - start)
        norm = float(np.linalg.norm(direction))
        while norm == 0.0:
            direction = rng.standard_normal(end - start)
            norm = float(np.linalg.norm(direction))
        noise[start:end] = target * direction / norm
    return noise


def _extract_snapshot(obs: dict, resize_size: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract the current policy images and 8-dim state from a LIBERO observation."""
    from openpi_client import image_tools  # noqa: PLC0415

    images: dict[str, np.ndarray] = {}
    for source, key in (
        ("agentview_image", "observation/image"),
        ("robot0_eye_in_hand_image", "observation/wrist_image"),
    ):
        if source not in obs:
            raise KeyError(f"Camera source {source!r} is missing from LIBERO observation keys {sorted(obs)}")
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        images[key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    return images, _build_state(obs)


def _build_state(obs: dict) -> np.ndarray:
    """Build LIBERO's eight-dimensional state in the policy's expected order."""
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32, copy=False)


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


def _write_summary(
    output_dir: pathlib.Path,
    args: Args,
    records: list[dict[str, Any]],
    targets_by_delay: dict[int, dict[str, float]],
) -> None:
    summary = {
        "experiment": "libero_state_noise_sweep",
        "config": dataclasses.asdict(args),
        "state_definition": [
            "robot0_eef_pos[0:3]",
            "robot0_eef_quat converted to axis-angle[0:3]",
            "robot0_gripper_qpos[0:2]",
        ],
        "calibration": {
            "source": str(args.calibration_summary),
            "targets": {str(delay_ms): targets for delay_ms, targets in sorted(targets_by_delay.items())},
        },
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
    noise = policy_state - current_state
    panel_height = 122
    panel = np.full((panel_height, image_row.shape[1], 3), 245, dtype=np.uint8)
    panel_image = Image.fromarray(panel)
    draw = ImageDraw.Draw(panel_image)
    draw.text(
        (6, 6),
        f"state noise (delay level={effective_delay_ms:g} ms) | step={step} | query={'YES' if policy_query else 'no'} | "
        f"noise_l2={np.linalg.norm(noise):.4f} | "
        f"noise_max={np.max(np.abs(noise)):.4f}",
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
        "noise   : " + np.array2string(noise, precision=3, suppress_small=True, separator=", "),
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
    if not args.calibration_summary:
        raise ValueError("calibration_summary is required (path to a state-delay summary.json)")
    if not pathlib.Path(args.calibration_summary).is_file():
        raise ValueError(f"Calibration summary not found: {args.calibration_summary}")


if __name__ == "__main__":
    import tyro  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    run_noise_sweep(tyro.cli(Args))
