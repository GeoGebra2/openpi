"""Record spatial SigLIP token drift over a closed-loop LIBERO rollout.

The policy sees an ordinary current observation or a real delayed camera frame.
No pixels are edited.  Segmentation is used only to draw target/gripper
markers on the diagnostic video.  The server returns a compact per-token drift
map, which is held between diagnostic queries and rendered into every frame.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import logging
import pathlib
from typing import Any, Optional

import numpy as np

try:
    from examples.libero import run_counterfactual_image_delay as _counter
    from examples.libero import run_shift_match_sweep as _shift
except ImportError:
    import run_counterfactual_image_delay as _counter
    import run_shift_match_sweep as _shift


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
    repeats_per_setting: int = 3
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    video_fps: int = 10
    record_every_env_steps: int = 1
    layer: int = -1
    output_dir: str = "rollout/libero_token_drift"


@dataclasses.dataclass
class Frame:
    rgb: np.ndarray
    target: np.ndarray
    gripper: np.ndarray


def _orient_rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    ) > 127


def _gripper_geom_ids(model: Any) -> list[int]:
    """Find the Panda hand and finger geoms, excluding the rest of the arm."""
    result = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_names = []
        while body_id >= 0:
            body_names.append(model.body_id2name(body_id) or "")
            parent = int(model.body_parentid[body_id])
            if parent == body_id:
                break
            body_id = parent
        searchable = " ".join((name, *body_names)).lower()
        if any(word in searchable for word in ("finger", "gripper", "hand", "eef")):
            result.append(geom_id)
    if not result:
        raise ValueError("No finger/gripper/hand geoms found in the MuJoCo model")
    return result


def _geom_segmentation_mask(segmentation: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    """MuJoCo segmentation pixels encode [object type, object id]."""
    raw = np.asarray(segmentation)
    if raw.ndim != 3 or raw.shape[-1] != 2:
        raise ValueError(f"Expected MuJoCo [H,W,2] geom segmentation, got {raw.shape}")
    # mjOBJ_GEOM is 5. Do not mistake instance IDs for geom IDs.
    types, ids = raw[..., 0], raw[..., 1]
    if not np.any(types == 5) and np.any(ids == 5):
        types, ids = ids, types
    if not np.any(types == 5):
        raise ValueError("Segmentation has no MuJoCo geom pixels (mjOBJ_GEOM=5)")
    return (types == 5) & np.isin(ids, geom_ids)


def _gripper_mask(env: Any, source: str, shape: tuple[int, int]) -> np.ndarray:
    camera_name = source[:-len("_image")] if source.endswith("_image") else source
    raw = env.sim.render(
        camera_name=camera_name, width=shape[1], height=shape[0], segmentation=True,
    )
    mask = _geom_segmentation_mask(raw, _gripper_geom_ids(env.sim.model))
    if mask.shape != shape:
        raise ValueError(f"Gripper mask shape {mask.shape} does not match camera image {shape}")
    return np.ascontiguousarray(mask[::-1, ::-1])


def _frame(env: Any, observation: dict[str, Any], source: str, target_object: str,
           resize_size: int) -> Frame:
    from openpi_client import image_tools

    raw = _orient_rgb(observation[source])
    rgb = image_tools.convert_to_uint8(image_tools.resize_with_pad(raw, resize_size, resize_size))
    target = _counter._target_mask(env, observation, source, target_object, raw.shape[:2])
    target = _resize_mask(target, rgb.shape[:2])
    gripper = _resize_mask(_gripper_mask(env, source, raw.shape[:2]), rgb.shape[:2])
    return Frame(np.ascontiguousarray(rgb), target, gripper)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _panel(image: np.ndarray, target: np.ndarray, gripper: np.ndarray,
           drift: np.ndarray | None, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = np.asarray(image, dtype=np.uint8)
    if drift is not None:
        values = np.asarray(drift, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        vmax = max(float(np.percentile(values, 98)), 1e-6)
        normalized = np.clip(values / vmax, 0.0, 1.0)
        heat = Image.fromarray(np.asarray(normalized * 255, dtype=np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
        )
        import matplotlib.cm as cm
        rgba = (cm.get_cmap("magma")(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
        image = np.asarray(Image.blend(Image.fromarray(image), Image.fromarray(rgba), 0.48))
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.shape[1] - 1, image.shape[0] - 1), outline=(255, 255, 255), width=2)
    draw.rectangle((3, 3, 3 + max(80, len(title) * 7), 22), fill=(0, 0, 0))
    draw.text((7, 6), title, fill=(255, 255, 255))
    for mask, color, label in ((target, (40, 255, 40), "target"), (gripper, (0, 220, 255), "gripper")):
        box = _bbox(mask)
        if box is not None:
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, max(0, box[1] - 12)), label, fill=color)
    return np.asarray(canvas)


def _grid_map(values: np.ndarray, layer: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected [layer, token] drift, got {values.shape}")
    index = layer if layer >= 0 else values.shape[0] + layer
    if not 0 <= index < values.shape[0]:
        raise ValueError(f"layer {layer} is outside [0, {values.shape[0] - 1}]")
    token_count = values.shape[1]
    side = int(round(token_count ** 0.5))
    if side * side != token_count:
        raise ValueError(f"SigLIP token count {token_count} is not a square spatial grid")
    return values[index].reshape(side, side)


def _video_frame(current: Frame, delayed: Frame, drift: np.ndarray | None,
                 layer: int, step: int, action_l2: float | None) -> np.ndarray:
    heat = _grid_map(drift, layer) if drift is not None else None
    left = _panel(current.rgb, current.target, current.gripper, None, f"current | step {step}")
    right = _panel(delayed.rgb, delayed.target, delayed.gripper, None, "delayed input")
    heat_left = _panel(current.rgb, current.target, current.gripper, heat, "token drift on current")
    detail = f"token drift layer={layer}"
    if action_l2 is not None:
        detail += f" | action L2={action_l2:.4f}"
    heat_right = _panel(delayed.rgb, delayed.target, delayed.gripper, heat, detail)
    top = np.concatenate((left, right), axis=1)
    bottom = np.concatenate((heat_left, heat_right), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _paired_noise(seed: int, repeat: int, step: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, repeat, step])).standard_normal(shape).astype(np.float32)


def _run_episode(args: Args, client: Any, task: Any, initial_state: np.ndarray,
                 prompt: str, repeat: int, noise_shape: tuple[int, int], output_dir: pathlib.Path) -> dict[str, Any]:
    delay_steps = round(args.delay_ms * args.control_hz / 1000.0)
    warmup = max(args.num_steps_wait, delay_steps)
    env = None
    frames: list[np.ndarray] = []
    drift_maps: list[np.ndarray] = []
    drift_steps: list[int] = []
    action_l2: list[float] = []
    action_plan = collections.deque()
    outcome, error = "failure", None
    try:
        env = _counter._create_segmentation_env(task)
        env.seed(args.seed + repeat)
        env.reset()
        obs = env.set_init_state(np.array(initial_state, copy=True))
        image_history = {source: collections.deque(maxlen=delay_steps + 1) for source in _shift.CAMERA_SOURCES}
        mask_history: dict[str, collections.deque[Frame]] = {
            source: collections.deque(maxlen=delay_steps + 1) for source in _shift.CAMERA_SOURCES
        }
        state_history = collections.deque(maxlen=delay_steps + 1)
        target = args.target_object.strip() or str(getattr(env, "obj_of_interest", [""])[0])
        if not target or target not in getattr(env, "instance_to_id", {}):
            raise ValueError(f"Target object {target!r} is not available in segmentation instances")

        def append(current_obs):
            for source in _shift.CAMERA_SOURCES:
                item = _frame(env, current_obs, source, target, args.resize_size)
                image_history[source].append(item.rgb)
                mask_history[source].append(item)
            state_history.append(_shift._build_state(current_obs))

        append(obs)
        for _ in range(warmup):
            obs, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            append(obs)
            if done:
                raise RuntimeError("Task ended during warmup")

        selected_source = (
            "robot0_eye_in_hand_image"
            if args.delay_target == "observation/wrist_image"
            else "agentview_image"
        )
        selected_camera = "left_wrist_0_rgb" if selected_source == "robot0_eye_in_hand_image" else "base_0_rgb"
        last_drift = None
        last_action_l2 = None
        max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
        for step in range(max_steps):
            current = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target=_shift.BASELINE_TARGET, delay_steps=0, prompt=prompt,
            )
            delayed = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target=args.delay_target, delay_steps=delay_steps, prompt=prompt,
            )
            query = not action_plan
            if query:
                noise = _paired_noise(args.seed, repeat, step, noise_shape)
                response = client.infer_with_noise(delayed, noise=noise)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape[1] < 7:
                    raise ValueError(f"Expected action dimension >=7, got {actions.shape}")
                action_plan.extend(actions[:args.replan_steps])
            if step % args.record_every_env_steps == 0:
                drift_noise = _paired_noise(args.seed + 100003, repeat, step, noise_shape)
                drift = client.infer_token_drift(delayed, current, drift_noise, camera=selected_camera)
                last_drift = np.asarray(drift["token_relative_l2"], dtype=np.float32)
                last_action_l2 = float(drift["action_l2"])
                drift_maps.append(last_drift.copy())
                drift_steps.append(step)
                action_l2.append(last_action_l2)
            current_frame = mask_history[selected_source][-1]
            delayed_frame = mask_history[selected_source][-1 - delay_steps]
            frames.append(_video_frame(current_frame, delayed_frame, last_drift, args.layer, step, last_action_l2))
            action = action_plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            append(obs)
            if env.check_success():
                outcome = "success"
                break
            if done:
                break
    except KeyboardInterrupt:
        outcome, error = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        outcome, error = "error", repr(exc)
        logging.exception("Token drift rollout failed: repeat=%d", repeat)
    finally:
        if env is not None:
            env.close()

    slug = f"repeat_{repeat:03d}"
    trace_path = output_dir / f"{slug}_token_drift.npz"
    np.savez_compressed(trace_path, steps=np.asarray(drift_steps, dtype=np.int32),
                        token_relative_l2=np.asarray(drift_maps, dtype=np.float32),
                        action_l2=np.asarray(action_l2, dtype=np.float32))
    video_path = output_dir / f"{slug}_{outcome}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps)
    return {
        "repeat_index": repeat, "outcome": outcome, "success": outcome == "success",
        "steps": len(frames), "delay_steps": delay_steps,
        "effective_delay_ms": delay_steps * 1000.0 / args.control_hz,
        "trace_path": str(trace_path), "video_path": str(video_path) if video_ok else None,
        "video_error": video_error, "error": error,
    }


def run(args: Args) -> pathlib.Path:
    if args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError(f"delay_target must be one of {_shift.CAMERA_KEYS}")
    if args.delay_ms <= 0 or args.repeats_per_setting <= 0 or args.record_every_env_steps <= 0:
        raise ValueError("delay_ms, repeats_per_setting and record_every_env_steps must be positive")
    delay_float = args.delay_ms * args.control_hz / 1000.0
    if not np.isfinite(delay_float) or delay_float < 1 or not np.isclose(delay_float, round(delay_float)):
        raise ValueError("delay_ms must map to an integer positive control step")
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata().get("token_drift", {})
    if metadata.get("schema_version") != 1:
        raise RuntimeError("Restart the policy server with token_drift support and --diagnostics")
    shape = tuple(int(value) for value in metadata.get("noise_shape", []))
    output_dir = pathlib.Path(args.output_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for repeat in range(args.repeats_per_setting):
            record = _run_episode(args, client, task, states[args.initial_state_id], prompt, repeat, shape, output_dir)
            records.append(record)
            summary = {"config": dataclasses.asdict(args), "records": records}
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logging.info(
                "[%d/%d] outcome=%s success=%s",
                repeat + 1, args.repeats_per_setting, record["outcome"], record["success"],
            )
            if record["outcome"] in ("error", "interrupted"):
                break
    finally:
        client._ws.close()
    return output_dir / "summary.json"


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    print(run(tyro.cli(Args)))
