"""Compare four visual counterfactuals for one delayed LIBERO camera.

The experiment keeps the MuJoCo state unchanged and changes only what the
policy sees in the selected camera:

``clean``
    Current background and current target object.
``object_only``
    Current background with the visible target-object pixels from the delayed
    frame pasted into it.
``context_only``
    Delayed background with the current target-object pixels pasted into it.
``full``
    The complete delayed camera frame.

Segmentation masks are obtained from LIBERO's ``SegmentationRenderEnv``.  The
target object is removed from the base frame with image inpainting before the
other frame's target pixels are inserted.  The physical object is never moved.
Each condition is a separate closed-loop rollout from the same initial state,
with the same explicit policy noise at matched environment steps.
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
from typing import Any, Optional

import numpy as np

try:
    from examples.libero import run_shift_match_sweep as _shift
    from examples.libero import run_trajectory_delay as _trajectory
except ImportError:
    import run_shift_match_sweep as _shift
    import run_trajectory_delay as _trajectory


EXPERIMENT = "libero_counterfactual_image_delay"
SCHEMA_VERSION = 1
CONDITIONS = ("clean", "object_only", "context_only", "full")
CAMERA_TARGETS = tuple(_shift.CAMERA_KEYS)


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
    target_object: str = ""
    repeats_per_condition: int = 30
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    output_dir: str = "rollout/libero_counterfactual_image_delay"
    skip_analysis: bool = False
    interactive: bool = False
    threshold_cm: float = 2.0
    consecutive: int = 3
    pair_limit: int = 10


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    delay_target: str
    delay_ms: int
    delay_steps: int


@dataclasses.dataclass
class CameraFrame:
    rgb: np.ndarray
    target_mask: np.ndarray


def paired_noise(seed: int, repeat_index: int, step: int, shape: tuple[int, ...]) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([seed, repeat_index, step]))
    return generator.standard_normal(shape).astype(np.float32)


def noise_shape_from_metadata(metadata: dict[str, Any]) -> tuple[int, int]:
    for name, version in (("wrist_kv_patch", 1), ("activation_drift", 2), ("attention", 1)):
        capability = metadata.get(name, {})
        if capability.get("schema_version") == version:
            shape = capability.get("noise_shape", [])
            if len(shape) == 2 and all(type(value) is int and value > 0 for value in shape) and shape[1] >= 7:
                return tuple(shape)
    raise RuntimeError("Server lacks noise-shape metadata; use the updated explicit-noise policy server")


def validate_args(args: Args) -> None:
    if args.delay_target not in CAMERA_TARGETS:
        raise ValueError(f"delay_target must be one of {CAMERA_TARGETS}, got {args.delay_target!r}")
    if args.delay_ms <= 0 or args.repeats_per_condition <= 0 or args.seed < 0:
        raise ValueError("delay_ms, repeats_per_condition and seed must be positive/non-negative")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if args.replan_steps <= 0 or args.resize_size <= 0 or args.video_fps <= 0 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, resize, video rate or warmup value")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.threshold_cm <= 0 or args.consecutive < 1 or args.pair_limit < 0:
        raise ValueError("Invalid analysis threshold, consecutive count or pair limit")
    exact_steps = args.delay_ms * args.control_hz / 1000.0
    if not np.isclose(exact_steps, round(exact_steps), rtol=0, atol=1e-8):
        raise ValueError("delay_ms must map to an integer simulator control step")


def make_conditions(args: Args) -> list[Condition]:
    validate_args(args)
    delay_steps = int(round(args.delay_ms * args.control_hz / 1000.0))
    return [
        Condition("clean", args.delay_target, 0, 0),
        Condition("object_only", args.delay_target, args.delay_ms, delay_steps),
        Condition("context_only", args.delay_target, args.delay_ms, delay_steps),
        Condition("full", args.delay_target, args.delay_ms, delay_steps),
    ]


def _orient_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _orient_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"Expected mask shape {shape}, got {mask.shape}")
    return np.ascontiguousarray(mask[::-1, ::-1].astype(bool, copy=False))


def _segmentation_key(observation: dict[str, Any], source: str) -> Optional[str]:
    prefix = source[:-len("_image")] if source.endswith("_image") else source
    exact = f"{prefix}_segmentation_instance"
    if exact in observation:
        return exact
    candidates = sorted(
        key for key in observation
        if key.startswith(prefix + "_segmentation") and "instance" in key
    )
    return candidates[0] if candidates else None


def _target_mask(env: Any, observation: dict[str, Any], source: str, target_object: str,
                 image_shape: tuple[int, int]) -> np.ndarray:
    key = _segmentation_key(observation, source)
    if key is None:
        raise KeyError(f"No instance segmentation for {source!r}; observation keys: {sorted(observation)}")
    raw = np.asarray(observation[key])
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.shape != image_shape:
        raise ValueError(f"Expected a single instance-ID plane of shape {image_shape}, got {raw.shape}")
    target_id = getattr(env, "instance_to_id", {}).get(target_object)
    if target_id is None or int(target_id) <= 0:
        raise ValueError(f"Target {target_object!r} has no positive LIBERO instance ID")
    return _orient_mask(raw == int(target_id), image_shape)


def _inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[-1] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("Image/mask shapes are incompatible")
    if not np.any(mask):
        return image.copy()
    try:
        import cv2  # noqa: PLC0415
        return cv2.inpaint(image, (mask.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA)
    except ImportError:
        # Small deterministic fallback for environments without OpenCV.  It
        # propagates the nearest available 4-neighbour colour into the mask;
        # the real LIBERO environment normally has cv2 through robosuite.
        output = image.copy()
        remaining = mask.copy()
        for _ in range(max(image.shape[:2])):
            if not np.any(remaining):
                break
            filled = np.zeros_like(remaining)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                shifted = np.zeros_like(output)
                shifted_mask = np.zeros_like(remaining)
                if dy == -1:
                    shifted[:-1] = output[1:]
                    shifted_mask[:-1] = remaining[1:]
                elif dy == 1:
                    shifted[1:] = output[:-1]
                    shifted_mask[1:] = remaining[:-1]
                elif dx == -1:
                    shifted[:, :-1] = output[:, 1:]
                    shifted_mask[:, :-1] = remaining[:, 1:]
                else:
                    shifted[:, 1:] = output[:, :-1]
                    shifted_mask[:, 1:] = remaining[:, :-1]
                available = remaining & ~shifted_mask
                output[available] = shifted[available]
                filled |= available
            if not np.any(filled):
                break
            remaining &= ~filled
        if np.any(remaining):
            output[remaining] = np.median(image[~mask], axis=0).astype(np.uint8)
        return output


def compose_target_only(current: CameraFrame, delayed: CameraFrame) -> np.ndarray:
    """Current background plus delayed visible target-object pixels."""
    background = _inpaint(current.rgb, current.target_mask)
    output = background.copy()
    output[delayed.target_mask] = delayed.rgb[delayed.target_mask]
    return output


def compose_context_only(current: CameraFrame, delayed: CameraFrame) -> np.ndarray:
    """Delayed background plus current visible target-object pixels."""
    background = _inpaint(delayed.rgb, delayed.target_mask)
    output = background.copy()
    output[current.target_mask] = current.rgb[current.target_mask]
    return output


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    from openpi_client import image_tools  # noqa: PLC0415
    return np.ascontiguousarray(image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size)))


def _condition_images(history: dict[str, collections.deque], target_source: str,
                      condition: str, delay_steps: int, resize_size: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    current = {source: history[source][-1] for source in _shift.CAMERA_SOURCES}
    delayed = {source: history[source][-1 - delay_steps] for source in _shift.CAMERA_SOURCES}
    chosen = {source: frame.rgb for source, frame in current.items()}
    if condition == "full":
        chosen[target_source] = delayed[target_source].rgb
    elif condition == "object_only":
        chosen[target_source] = compose_target_only(current[target_source], delayed[target_source])
    elif condition == "context_only":
        chosen[target_source] = compose_context_only(current[target_source], delayed[target_source])
    elif condition != "clean":
        raise ValueError(f"Unknown counterfactual condition {condition!r}")
    resized = {source: _resize_image(image, resize_size) for source, image in chosen.items()}
    current_resized = {source: _resize_image(frame.rgb, resize_size) for source, frame in current.items()}
    return resized, current_resized


def _build_policy_observation(history: dict[str, collections.deque], states: collections.deque,
                              condition: Condition, target_source: str, prompt: str,
                              resize_size: int) -> tuple[dict[str, Any], dict[str, float]]:
    if len(history[target_source]) <= condition.delay_steps:
        raise RuntimeError("Camera history is shorter than the requested delay")
    images, current_images = _condition_images(
        history, target_source, condition.name, condition.delay_steps, resize_size
    )
    current = history[target_source][-1]
    delayed = history[target_source][-1 - condition.delay_steps]
    diagnostics = {
        "current_target_pixels": float(np.count_nonzero(current.target_mask)),
        "delayed_target_pixels": float(np.count_nonzero(delayed.target_mask)),
        "current_target_visible": float(np.any(current.target_mask)),
        "delayed_target_visible": float(np.any(delayed.target_mask)),
    }
    observation = {
        "observation/image": images["agentview_image"],
        "observation/wrist_image": images["robot0_eye_in_hand_image"],
        "observation/state": np.array(states[-1], copy=True),
        "prompt": prompt,
    }
    return observation, {**diagnostics, "current_images": current_images, "policy_images": images}


def _append_snapshot(history: dict[str, collections.deque], states: collections.deque,
                     observation: dict[str, Any], env: Any, target_object: str) -> None:
    for source in _shift.CAMERA_SOURCES:
        rgb = _orient_rgb(observation[source])
        mask = _target_mask(env, observation, source, target_object, rgb.shape[:2])
        history[source].append(CameraFrame(rgb=rgb, target_mask=mask))
    states.append(_shift._build_state(observation))


def _select_target_object(env: Any, requested: str) -> str:
    available = list(getattr(env, "obj_of_interest", []))
    target = requested.strip() if requested and requested.strip() else (available[0] if available else "")
    if not target:
        raise ValueError("No target object was provided and the task has no obj_of_interest")
    mapping = getattr(env, "instance_to_id", {})
    if target not in mapping:
        raise ValueError(f"Target object {target!r} is not in LIBERO instance mapping: {sorted(mapping)}")
    return target


def _create_segmentation_env(task: Any):
    from libero.libero import get_libero_path  # noqa: PLC0415
    from libero.libero.envs import SegmentationRenderEnv  # noqa: PLC0415

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=_shift.LIBERO_ENV_RESOLUTION,
        camera_widths=_shift.LIBERO_ENV_RESOLUTION,
        camera_segmentations="instance",
    )
    return env


def _state_sample(env: Any, observation: dict[str, Any], objects: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    return _trajectory.state_sample(env, observation, objects)


def _run_episode(*, args: Args, client: Any, task: Any, initial_state: np.ndarray, prompt: str,
                 condition: Condition, repeat_index: int, noise_shape: tuple[int, int],
                 warmup_steps: int, history_steps: int, max_steps: int,
                 target_object: str, output_dir: pathlib.Path) -> dict[str, Any]:
    env = None
    history = {source: collections.deque(maxlen=history_steps + 1) for source in _shift.CAMERA_SOURCES}
    states = collections.deque(maxlen=history_steps + 1)
    plan = collections.deque()
    samples: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    queries: list[int] = []
    source_steps: list[list[int]] = []
    policy_states: list[np.ndarray] = []
    mask_stats: list[list[float]] = []
    frames: list[np.ndarray] = []
    outcome, reason, error_text = "failure", "max_steps", None
    actual_hz = None
    objects: list[tuple[str, int]] = []
    missing_objects: list[str] = []
    start_sim = np.empty(0)
    resolved_target = target_object

    try:
        env = _create_segmentation_env(task)
        env.seed(args.seed + repeat_index)
        actual_hz = _trajectory.simulator_hz(env)
        if not np.isclose(actual_hz, args.control_hz, rtol=0, atol=1e-8):
            raise ValueError(f"control_hz={args.control_hz} differs from simulator control_freq={actual_hz}")
        env.reset()
        resolved_target = _select_target_object(env, target_object)
        observation = env.set_init_state(np.array(initial_state, copy=True))
        _append_snapshot(history, states, observation, env, resolved_target)
        for _ in range(warmup_steps):
            observation, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _append_snapshot(history, states, observation, env, resolved_target)
            if done or env.check_success():
                raise RuntimeError("Task ended during warmup; no valid trajectory comparison")
        objects, missing_objects = _trajectory.object_references(env, "")
        start_sim = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        samples.append(_state_sample(env, observation, objects))

        for step in range(max_steps):
            policy_observation, details = _build_policy_observation(
                history, states, condition, _source_for_target(args.delay_target), prompt, args.resize_size
            )
            query = not plan
            if args.save_video:
                frames.append(_shift._make_video_frame(
                    current_images={
                        key: details["current_images"][source]
                        for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)
                    },
                    policy_images={
                        key: details["policy_images"][source]
                        for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)
                    },
                    current_state=states[-1], policy_state=policy_observation["observation/state"],
                    delay_target=args.delay_target, effective_delay_ms=condition.delay_steps / actual_hz * 1000,
                    step=step, policy_query=query,
                ))
            if query:
                noise = paired_noise(args.seed, repeat_index, step, noise_shape)
                response = client.infer_with_noise(policy_observation, noise)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                expected_shape = (noise_shape[0], 7)
                if chunk.shape != expected_shape or not np.all(np.isfinite(chunk)):
                    raise ValueError(f"Expected finite actions {expected_shape}, got {chunk.shape}")
                queries.append(step)
                selected_column = _shift.CAMERA_KEYS.index(args.delay_target)
                source_steps.append([
                    step - condition.delay_steps if index == selected_column else step
                    for index in range(3)
                ])
                chunks.append(chunk.copy())
                noises.append(noise)
                policy_states.append(np.array(policy_observation["observation/state"], copy=True))
                mask_stats.append([
                    details["current_target_pixels"], details["delayed_target_pixels"],
                    details["current_target_visible"], details["delayed_target_visible"],
                ])
                plan.extend(chunk[:args.replan_steps])
            action = plan.popleft()
            observation, _, done, _ = env.step(action.tolist())
            actions.append(action.copy())
            samples.append(_state_sample(env, observation, objects))
            _append_snapshot(history, states, observation, env, resolved_target)
            if env.check_success():
                outcome, reason = "success", "task_success"
                break
            if done:
                reason = "environment_terminated_without_success"
                break
    except KeyboardInterrupt:
        outcome, reason, error_text = "interrupted", "interrupted", "KeyboardInterrupt"
    except Exception as error:
        outcome, reason, error_text = "error", "runtime_error", repr(error)
        logging.exception("Counterfactual image-delay run failed: %s repeat=%d", condition.name, repeat_index)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("Environment cleanup failed")

    slug = f"{condition.name}_repeat_{repeat_index:03d}"
    trace_path = output_dir / f"{slug}_trajectory.npz"
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]} if samples else {}
    np.savez_compressed(
        trace_path, **arrays,
        steps=np.arange(len(samples), dtype=np.int32),
        time_s=np.arange(len(samples), dtype=np.float64) / args.control_hz,
        executed_actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
        query_steps=np.asarray(queries, dtype=np.int32),
        input_source_steps=np.asarray(source_steps, dtype=np.int32).reshape(-1, 3),
        action_chunks=np.asarray(chunks, dtype=np.float32).reshape(-1, noise_shape[0], 7),
        noise=np.asarray(noises, dtype=np.float32).reshape(-1, *noise_shape),
        policy_states=np.asarray(policy_states, dtype=np.float32).reshape(-1, 8),
        mask_stats=np.asarray(mask_stats, dtype=np.float32).reshape(-1, 4),
        object_names=np.asarray([name for name, _ in objects], dtype=str),
        start_sim_state=start_sim,
    )
    video_path = output_dir / f"{slug}_{outcome}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps) if frames else (False, None)
    return {
        "condition": condition.name,
        "repeat_index": repeat_index,
        "seed": args.seed + repeat_index,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "initial_state_id": args.initial_state_id,
        "initial_state_sha256": hashlib.sha256(np.asarray(initial_state, dtype=np.float64).tobytes()).hexdigest(),
        "prompt": prompt,
        "delay_target": args.delay_target,
        "target_object": resolved_target,
        "delay_ms_requested": condition.delay_ms,
        "delay_steps": condition.delay_steps,
        "delay_ms_effective": condition.delay_steps / args.control_hz * 1000 if actual_hz else None,
        "control_hz": actual_hz,
        "replan_steps": args.replan_steps,
        "warmup_steps": warmup_steps,
        "outcome": outcome,
        "termination_reason": reason,
        "success": outcome == "success",
        "steps": len(actions),
        "n_samples": len(samples),
        "n_queries": len(queries),
        "trace_path": trace_path.name,
        "video_path": video_path.name if video_ok else None,
        "video_fps": args.video_fps,
        "video_frames": len(frames),
        "video_error": video_error,
        "object_names": [name for name, _ in objects],
        "unavailable_objects": missing_objects,
        "error": error_text,
    }


def _source_for_target(target: str) -> str:
    return {
        "observation/image": "agentview_image",
        "observation/wrist_image": "robot0_eye_in_hand_image",
    }[target]


def _aggregate(records: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    result = []
    for condition in CONDITIONS:
        group = grouped.get(condition, [])
        counts = collections.Counter(record["outcome"] for record in group)
        completed = counts["success"] + counts["failure"]
        result.append({
            "condition": condition,
            "runs": len(group),
            "expected_runs": repeats,
            "successes": counts["success"],
            "failures": counts["failure"],
            "errors": counts["error"],
            "interrupted": counts["interrupted"],
            "completed": completed,
            "success_rate": counts["success"] / completed if completed else None,
            "failure_rate": counts["failure"] / completed if completed else None,
        })
    return result


def save_summary(directory: pathlib.Path, args: Args, metadata: dict[str, Any],
                 conditions: list[Condition], records: list[dict[str, Any]]) -> None:
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "config": dataclasses.asdict(args),
        "server_metadata": metadata,
        "conditions": [dataclasses.asdict(condition) for condition in conditions],
        "protocol": {
            "four_inputs": {
                "clean": "current background + current target",
                "object_only": "current background + delayed target mask/pixels",
                "context_only": "delayed background + current target mask/pixels",
                "full": "complete delayed camera frame",
            },
            "selected_camera_only": args.delay_target,
            "target_object": args.target_object or "first LIBERO obj_of_interest",
            "physical_state": "unchanged by image composition; real MuJoCo object is never moved",
            "segmentation": "LIBERO SegmentationRenderEnv instance masks",
            "sampling": "all control steps including initial and terminal state; samples=actions+1",
            "pairing": "same task, initial state, repeat seed, warmup and explicit noise",
            "noise": "PCG64 SeedSequence([seed, repeat_index, env_step])",
            "success": "LIBERO environment task predicate",
        },
        "aggregate": _aggregate(records, args.repeats_per_condition),
        "records": records,
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _run_rollouts(args: Args) -> pathlib.Path:
    conditions = make_conditions(args)
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {suite.n_tasks - 1}]")
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(f"initial_state_id must be in [0, {len(initial_states) - 1}]")
    task = suite.get_task(args.task_id)
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        metadata = client.get_server_metadata()
        noise_shape = noise_shape_from_metadata(metadata)
        if not callable(getattr(client, "infer_with_noise", None)):
            raise ValueError("Updated infer_with_noise client required")
        if args.replan_steps > noise_shape[0]:
            raise ValueError("replan_steps exceeds the policy action horizon")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = pathlib.Path(args.output_dir) / stamp
        directory.mkdir(parents=True, exist_ok=False)
        max_delay = max(condition.delay_steps for condition in conditions)
        warmup_steps = max(args.num_steps_wait, max_delay)
        max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
        records: list[dict[str, Any]] = []
        save_summary(directory, args, metadata, conditions, records)
        logging.info(
            "Running %d counterfactual rollouts (%d conditions x %d repeats)",
            len(conditions) * args.repeats_per_condition, len(conditions), args.repeats_per_condition,
        )
        logging.info("Selected camera: %s; target object: %s", args.delay_target, args.target_object or "first obj_of_interest")
        for repeat in range(args.repeats_per_condition):
            for condition in conditions:
                record = _run_episode(
                    args=args, client=client, task=task, initial_state=initial_states[args.initial_state_id],
                    prompt=prompt, condition=condition, repeat_index=repeat, noise_shape=noise_shape,
                    warmup_steps=warmup_steps, history_steps=max_delay, max_steps=max_steps,
                    target_object=args.target_object, output_dir=directory,
                )
                records.append(record)
                save_summary(directory, args, metadata, conditions, records)
                logging.info(
                    "[%d/%d] condition=%s repeat=%d outcome=%s steps=%d",
                    len(records), len(conditions) * args.repeats_per_condition,
                    condition.name, repeat, record["outcome"], record["steps"],
                )
                if record["outcome"] == "error":
                    raise RuntimeError(f"Stopped after error; partial results in {directory}: {record['error']}")
                if record["outcome"] == "interrupted":
                    return directory / "summary.json"
        return directory / "summary.json"
    finally:
        try:
            client._ws.close()
        except Exception:
            pass


def run(args: Args) -> pathlib.Path:
    make_conditions(args)
    if args.skip_analysis:
        return _run_rollouts(args)
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Automatic analysis requires matplotlib; use --skip-analysis to collect only") from error
    try:
        from examples.libero import analyze_counterfactual_image_delay as analysis
    except ImportError:
        import analyze_counterfactual_image_delay as analysis
    summary_path = _run_rollouts(args)
    logging.info("Rollouts saved to %s; generating counterfactual trajectory figures", summary_path)
    analysis_path = analysis.analyze(
        summary_path, threshold_cm=args.threshold_cm, consecutive=args.consecutive,
        pair_limit=args.pair_limit, interactive=args.interactive,
    )
    logging.info("Analysis complete: %s", analysis_path.parent)
    return summary_path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        if field.name == "save_video":
            parser.add_argument("--no-video", "--no_video", dest="save_video", action="store_false", default=True)
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
