"""Closed-loop success-rate experiment for local image-token patching.

The recipient receives a real delayed camera image.  On selected queries, only
the SigLIP tokens whose image patches overlap the target, gripper, their union,
or the remaining background are replaced by the clean current-camera tokens.
No pixels are edited and no diagnostic activations are returned.
"""

from __future__ import annotations

import collections
import csv
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
    from examples.libero.run_token_drift_video import Frame, _frame
except ImportError:
    import run_counterfactual_image_delay as _counter
    import run_shift_match_sweep as _shift
    from run_token_drift_video import Frame, _frame


REGIONS = ("clean", "delay", "target", "gripper", "target_gripper", "background")


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
    site: str = "siglip"
    layer: int = 0
    token_fraction_threshold: float = 0.20
    repeats_per_setting: int = 10
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    output_dir: str = "rollout/libero_token_patch"
    save_mask_visualization: bool = True
    visualize_repeat: int = 0
    visualize_every_query: int = 1
    visualization_fps: int = 5


def _paired_noise(seed: int, repeat: int, step: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, repeat, step])).standard_normal(shape).astype(np.float32)


def _token_mask(frame: Frame, region: str, side: int, threshold: float) -> np.ndarray:
    if frame.rgb.shape[0] % side or frame.rgb.shape[1] % side:
        raise ValueError(f"resize_size={frame.rgb.shape[:2]} is not divisible by token grid side {side}")
    patch_h, patch_w = frame.rgb.shape[0] // side, frame.rgb.shape[1] // side
    target = np.asarray(frame.target, dtype=np.float32)
    gripper = np.asarray(frame.gripper, dtype=np.float32)
    target_fraction = target.reshape(side, patch_h, side, patch_w).mean(axis=(1, 3))
    gripper_fraction = gripper.reshape(side, patch_h, side, patch_w).mean(axis=(1, 3))
    target_tokens = target_fraction >= threshold
    # Panda fingers can occupy less than 20% of a 14x14 patch.
    gripper_tokens = gripper_fraction > 0
    if region == "target":
        mask = target_tokens
    elif region == "gripper":
        mask = gripper_tokens
    elif region == "target_gripper":
        mask = target_tokens | gripper_tokens
    elif region == "background":
        mask = (target_fraction == 0) & (gripper_fraction == 0)
    else:
        raise ValueError(f"Unknown token patch region {region!r}")
    return mask.reshape(-1)


def _mask_overlay(frame: Frame, token_mask: np.ndarray | None, side: int,
                  title: str, selected_colour: tuple[int, int, int] = (255, 168, 0)) -> np.ndarray:
    """Draw the exact selected token cells over an RGB frame.

    The grid uses the same row-major order sent to the policy server. Target and
    gripper bounding boxes are drawn as independent visual references, so a
    reviewer can tell whether the selected cells actually overlap the object.
    """
    from PIL import Image, ImageDraw

    image = np.asarray(frame.rgb, dtype=np.uint8)
    canvas = Image.fromarray(image).convert("RGBA")
    height, width = image.shape[:2]
    cell_h, cell_w = height / side, width / side
    selected = None if token_mask is None else np.asarray(token_mask, dtype=bool).reshape(side, side)
    semantic_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for mask, colour in (
        (frame.target, (50, 255, 80)),
        (frame.gripper, (0, 220, 255)),
    ):
        pixels = np.asarray(mask, dtype=bool)
        semantic_overlay[pixels, :3] = colour
        semantic_overlay[pixels, 3] = 78
    canvas = Image.alpha_composite(canvas, Image.fromarray(semantic_overlay))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if selected is not None:
        for row in range(side):
            for col in range(side):
                x0, y0 = round(col * cell_w), round(row * cell_h)
                x1, y1 = round((col + 1) * cell_w) - 1, round((row + 1) * cell_h) - 1
                if selected[row, col]:
                    draw.rectangle(
                        (x0, y0, x1, y1),
                        fill=(*selected_colour, 92),
                        outline=(*selected_colour, 235),
                        width=1,
                    )
                else:
                    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 42), width=1)
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    for mask, colour, label in (
        (frame.target, (50, 255, 80), "target"),
        (frame.gripper, (0, 220, 255), "gripper"),
    ):
        box = bbox(mask)
        if box is not None:
            draw.rectangle(box, outline=(*colour, 255), width=2)
            draw.text((box[0] + 2, max(0, box[1] - 12)), label, fill=(*colour, 255))

    banner_height = 24
    draw.rectangle((0, 0, width - 1, banner_height), fill=(0, 0, 0, 205))
    selected_count = int(selected.sum()) if selected is not None else 0
    suffix = f" | selected={selected_count}/{side * side}" if token_mask is not None else " | no replacement"
    draw.text((6, 6), title + suffix, fill=(255, 255, 255, 255))
    return np.asarray(canvas.convert("RGB"))


def _mask_visualization_frame(current: Frame, delayed: Frame, current_mask: np.ndarray | None,
                              delayed_mask: np.ndarray | None, selected_mask: np.ndarray | None,
                              side: int, condition: str, step: int, query_index: int) -> np.ndarray:
    """Create a four-panel diagnostic frame for one actual patch query."""
    if selected_mask is None:
        current_panel = _mask_overlay(current, None, side, f"{condition} / current | step {step}")
        delayed_panel = _mask_overlay(delayed, None, side, f"{condition} / delayed | step {step}")
        panels = [current_panel, delayed_panel]
    else:
        panels = [
            _mask_overlay(current, current_mask, side, f"{condition} / region current | step {step}", (70, 190, 255)),
            _mask_overlay(delayed, delayed_mask, side, f"{condition} / region delayed | step {step}", (70, 190, 255)),
            _mask_overlay(current, selected_mask, side, f"{condition} / patch current q{query_index}"),
            _mask_overlay(delayed, selected_mask, side, f"{condition} / patch delayed q{query_index}"),
        ]
    if len(panels) == 2:
        return np.concatenate(panels, axis=1)
    return np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)


def _save_mask_visualizations(directory: pathlib.Path, condition: str, frames: list[np.ndarray],
                              fps: int) -> tuple[str | None, str | None]:
    """Save a condition video and its first frame; return relative paths."""
    if not frames:
        return None, None
    condition_dir = directory / "token_masks"
    condition_dir.mkdir(parents=True, exist_ok=True)
    first_path = condition_dir / f"{condition}_first.png"
    try:
        from PIL import Image

        Image.fromarray(frames[0]).save(first_path)
    except Exception:
        logging.exception("Failed to save token mask preview: %s", first_path)
        first_path = None
    video_path = condition_dir / f"{condition}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, fps)
    if not video_ok:
        logging.warning("Failed to save token mask video for %s: %s", condition, video_error)
        video_path = None
    return (
        str(video_path.relative_to(directory)) if video_path is not None else None,
        str(first_path.relative_to(directory)) if first_path is not None else None,
    )


def _write_first_query_comparison(directory: pathlib.Path, records: list[dict[str, Any]],
                                  visualize_repeat: int, image_size: int) -> None:
    """Put all six first-query camera views on one page for mask inspection."""
    from PIL import Image, ImageDraw

    previews = {
        record["condition"]: directory / record["mask_preview"]
        for record in records
        if record["repeat_index"] == visualize_repeat and record.get("mask_preview")
    }
    if not previews:
        return
    margin, gap, header = 12, 12, 30
    sheet = Image.new("RGB", (3 * image_size + 2 * gap + 2 * margin,
                              2 * (image_size + header) + gap + 2 * margin), (32, 36, 41))
    draw = ImageDraw.Draw(sheet)
    for index, condition in enumerate(REGIONS):
        path = previews.get(condition)
        if path is None or not path.exists():
            continue
        with Image.open(path) as preview:
            if condition == "clean":
                selected = preview.crop((0, 0, image_size, image_size))
            elif condition == "delay":
                selected = preview.crop((image_size, 0, 2 * image_size, image_size))
            else:
                selected = preview.crop((image_size, image_size, 2 * image_size, 2 * image_size))
        x = margin + (index % 3) * (image_size + gap)
        y = margin + (index // 3) * (image_size + header + gap)
        draw.text((x + 4, y + 7), condition, fill=(255, 255, 255))
        sheet.paste(selected, (x, y + header))
    sheet.save(directory / "token_masks" / "first_query_comparison.png")


def _run_episode(args: Args, client: Any, task: Any, initial_state: np.ndarray,
                 prompt: str, region: str, repeat: int, noise_shape: tuple[int, int],
                 tokens_per_camera: int, output_dir: pathlib.Path) -> dict[str, Any]:
    delay_steps = round(args.delay_ms * args.control_hz / 1000.0)
    warmup = max(args.num_steps_wait, delay_steps)
    env = None
    plan = collections.deque()
    outcome, error = "failure", None
    patch_steps = 0
    empty_mask_steps = 0
    query_steps = 0
    gripper_visible_queries = 0
    target_visible_queries = 0
    selected_token_sum = 0
    current_region_token_sum = 0
    delayed_region_token_sum = 0
    current_target_pixel_sum = 0
    delayed_target_pixel_sum = 0
    current_gripper_pixel_sum = 0
    delayed_gripper_pixel_sum = 0
    visualization_frames: list[np.ndarray] = []
    visualization_query_index = 0
    try:
        env = _counter._create_segmentation_env(task)
        env.seed(args.seed + repeat)
        env.reset()
        obs = env.set_init_state(np.array(initial_state, copy=True))
        image_history = {source: collections.deque(maxlen=delay_steps + 1) for source in _shift.CAMERA_SOURCES}
        frame_history: dict[str, collections.deque[Frame]] = {
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
                frame_history[source].append(item)
            state_history.append(_shift._build_state(current_obs))

        append(obs)
        for _ in range(warmup):
            obs, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            append(obs)
            if done:
                raise RuntimeError("Task ended during warmup")

        source = "robot0_eye_in_hand_image" if args.delay_target == "observation/wrist_image" else "agentview_image"
        camera = "left_wrist_0_rgb" if source == "robot0_eye_in_hand_image" else "base_0_rgb"
        input_key = "observation/wrist_image" if camera == "left_wrist_0_rgb" else "observation/image"
        max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
        side = int(round(tokens_per_camera ** 0.5))
        if side * side != tokens_per_camera:
            raise ValueError(f"Server token count {tokens_per_camera} is not a square grid")

        for step in range(max_steps):
            current = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target=_shift.BASELINE_TARGET, delay_steps=0, prompt=prompt,
            )
            delayed = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target=args.delay_target, delay_steps=delay_steps, prompt=prompt,
            )
            query = not plan
            if query:
                query_steps += 1
                noise = _paired_noise(args.seed, repeat, step, noise_shape)
                current_mask = delayed_mask = selected_mask = None
                if region in ("clean", "delay"):
                    response = client.infer_with_noise(current if region == "clean" else delayed, noise=noise)
                else:
                    current_frame = frame_history[source][-1]
                    delayed_frame = frame_history[source][-1 - delay_steps]
                    if np.any(current_frame.gripper | delayed_frame.gripper):
                        gripper_visible_queries += 1
                    if np.any(current_frame.target | delayed_frame.target):
                        target_visible_queries += 1
                    current_mask = _token_mask(current_frame, region, side, args.token_fraction_threshold)
                    delayed_mask = _token_mask(delayed_frame, region, side, args.token_fraction_threshold)
                    current_region_token_sum += int(current_mask.sum())
                    delayed_region_token_sum += int(delayed_mask.sum())
                    current_target_pixel_sum += int(np.count_nonzero(current_frame.target))
                    delayed_target_pixel_sum += int(np.count_nonzero(delayed_frame.target))
                    current_gripper_pixel_sum += int(np.count_nonzero(current_frame.gripper))
                    delayed_gripper_pixel_sum += int(np.count_nonzero(delayed_frame.gripper))
                    # For foreground regions include both current and stale
                    # positions. For the background negative control require
                    # both frames to agree that a patch is background, so a
                    # moving target boundary cannot be repaired accidentally.
                    selected_mask = (
                        current_mask & delayed_mask if region == "background" else current_mask | delayed_mask
                    )
                    selected_token_sum += int(selected_mask.sum())
                    if not np.any(selected_mask):
                        empty_mask_steps += 1
                        response = client.infer_with_noise(delayed, noise=noise)
                    else:
                        response = client.infer_token_patch(
                            delayed, current[input_key], noise,
                            mode="clean", layers=[args.layer], site=args.site,
                            camera=camera, token_mask=selected_mask,
                        )
                        patch_steps += 1
                if (
                    args.save_mask_visualization
                    and repeat == args.visualize_repeat
                    and visualization_query_index % args.visualize_every_query == 0
                ):
                    current_frame = frame_history[source][-1]
                    delayed_frame = frame_history[source][-1 - delay_steps]
                    visualization_frames.append(
                        _mask_visualization_frame(
                            current_frame, delayed_frame, current_mask, delayed_mask,
                            selected_mask, side, region, step, visualization_query_index,
                        )
                    )
                visualization_query_index += 1
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] < 7:
                    raise ValueError(f"Expected action chunk with >=7 dimensions, got {actions.shape}")
                plan.extend(actions[:args.replan_steps])
            action = plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            append(obs)
            if env.check_success():
                outcome = "success"
                break
            if done:
                break
        if region in ("gripper", "target_gripper", "background") and gripper_visible_queries == 0:
            raise RuntimeError(
                "No gripper pixels were visible in either camera frame at any policy query; "
                "the gripper/background intervention is invalid"
            )
        if region in ("target", "target_gripper", "background") and target_visible_queries == 0:
            raise RuntimeError(
                "No target pixels were visible at any policy query; the target/background intervention is invalid"
            )
        if region not in ("clean", "delay") and patch_steps == 0:
            raise RuntimeError("No token patches were applied; the intervention is invalid")
    except KeyboardInterrupt:
        outcome, error = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        outcome, error = "error", repr(exc)
        logging.exception("Token patch rollout failed: region=%s repeat=%d", region, repeat)
    finally:
        if env is not None:
            env.close()
    mask_video, mask_preview = (None, None)
    if args.save_mask_visualization and repeat == args.visualize_repeat:
        mask_video, mask_preview = _save_mask_visualizations(
            output_dir, region, visualization_frames, args.visualization_fps,
        )
    return {
        "condition": region, "repeat_index": repeat, "outcome": outcome,
        "success": outcome == "success", "delay_steps": delay_steps,
        "query_steps": query_steps, "patch_steps": patch_steps,
        "empty_mask_steps": empty_mask_steps,
        "gripper_visible_queries": gripper_visible_queries,
        "target_visible_queries": target_visible_queries,
        "selected_tokens_mean": selected_token_sum / query_steps if query_steps else None,
        "current_region_tokens_mean": current_region_token_sum / query_steps if query_steps else None,
        "delayed_region_tokens_mean": delayed_region_token_sum / query_steps if query_steps else None,
        "current_target_pixels_mean": current_target_pixel_sum / query_steps if query_steps else None,
        "delayed_target_pixels_mean": delayed_target_pixel_sum / query_steps if query_steps else None,
        "current_gripper_pixels_mean": current_gripper_pixel_sum / query_steps if query_steps else None,
        "delayed_gripper_pixels_mean": delayed_gripper_pixel_sum / query_steps if query_steps else None,
        "mask_video": mask_video,
        "mask_preview": mask_preview,
        "error": error,
    }


def run(args: Args) -> pathlib.Path:
    if args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError(f"delay_target must be one of {_shift.CAMERA_KEYS}")
    if args.site not in ("siglip", "wrist_kv", "prefix_hidden"):
        raise ValueError("site must be siglip, wrist_kv, or prefix_hidden")
    if args.site == "siglip" and args.layer != 0:
        raise ValueError("SigLIP patching has only layer 0")
    if not 0 < args.token_fraction_threshold <= 1:
        raise ValueError("token_fraction_threshold must be in (0, 1]")
    if args.visualize_repeat < 0 or args.visualize_repeat >= args.repeats_per_setting:
        raise ValueError("visualize_repeat must be within repeats_per_setting")
    if args.visualize_every_query <= 0 or args.visualization_fps <= 0:
        raise ValueError("visualize_every_query and visualization_fps must be positive")
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    capability = metadata.get("token_patch", {})
    if capability.get("schema_version") != 1:
        raise RuntimeError("Restart the policy server with token_patch support and --diagnostics")
    noise_shape = tuple(int(value) for value in capability.get("noise_shape", []))
    tokens_per_camera = int(capability.get("tokens_per_camera", 256))
    if args.site != "siglip" and not 0 <= args.layer < int(capability.get("num_layers", 0)):
        raise ValueError(f"layer must be in [0, {int(capability.get('num_layers', 1)) - 1}]")
    delay_float = args.delay_ms * args.control_hz / 1000.0
    if not np.isfinite(delay_float) or delay_float < 1 or not np.isclose(delay_float, round(delay_float)):
        raise ValueError("delay_ms must map to an integer positive control step")
    output_dir = pathlib.Path(args.output_dir) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for region in REGIONS:
            for repeat in range(args.repeats_per_setting):
                record = _run_episode(
                    args, client, task, states[args.initial_state_id], prompt, region,
                    repeat, noise_shape, tokens_per_camera, output_dir,
                )
                records.append(record)
                _write_summary(output_dir, args, records)
                logging.info("[%d/%d] region=%s repeat=%d outcome=%s success=%s", len(records),
                             len(REGIONS) * args.repeats_per_setting, region, repeat,
                             record["outcome"], record["success"])
                if record["outcome"] in ("error", "interrupted"):
                    return output_dir / "summary.json"
    finally:
        client._ws.close()
    return output_dir / "summary.json"


def _write_summary(directory: pathlib.Path, args: Args, records: list[dict[str, Any]]) -> None:
    aggregate: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        rows = [row for row in records if row["condition"] == region]
        completed = [row for row in rows if row["outcome"] in ("success", "failure")]
        successes = sum(row["outcome"] == "success" for row in completed)
        aggregate[region] = {
            "attempted": len(rows),
            "completed": len(completed),
            "successes": int(successes),
            "failures": int(sum(row["outcome"] == "failure" for row in completed)),
            "errors": int(sum(row["outcome"] == "error" for row in rows)),
            "interrupted": int(sum(row["outcome"] == "interrupted" for row in rows)),
            "success_rate": successes / len(completed) if completed else None,
            "mean_patch_steps": float(np.mean([row["patch_steps"] for row in rows])) if rows else None,
            "mean_empty_mask_steps": float(np.mean([row["empty_mask_steps"] for row in rows])) if rows else None,
            "mean_selected_tokens": float(np.mean([
                row["selected_tokens_mean"] for row in rows
                if row.get("selected_tokens_mean") is not None
            ])) if any(row.get("selected_tokens_mean") is not None for row in rows) else None,
            "mean_current_region_tokens": float(np.mean([
                row["current_region_tokens_mean"] for row in rows
                if row.get("current_region_tokens_mean") is not None
            ])) if any(row.get("current_region_tokens_mean") is not None for row in rows) else None,
            "mean_delayed_region_tokens": float(np.mean([
                row["delayed_region_tokens_mean"] for row in rows
                if row.get("delayed_region_tokens_mean") is not None
            ])) if any(row.get("delayed_region_tokens_mean") is not None for row in rows) else None,
            "mean_current_target_pixels": float(np.mean([
                row["current_target_pixels_mean"] for row in rows
                if row.get("current_target_pixels_mean") is not None
            ])) if any(row.get("current_target_pixels_mean") is not None for row in rows) else None,
            "mean_delayed_target_pixels": float(np.mean([
                row["delayed_target_pixels_mean"] for row in rows
                if row.get("delayed_target_pixels_mean") is not None
            ])) if any(row.get("delayed_target_pixels_mean") is not None for row in rows) else None,
        }
    report = {"experiment": "libero_token_patch_success", "config": dataclasses.asdict(args),
              "aggregate": aggregate, "records": records}
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if records:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if args.save_mask_visualization:
        _write_first_query_comparison(directory, records, args.visualize_repeat, args.resize_size)
    _write_success_plots(directory, args, aggregate)


def _wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """Return a 95% Wilson interval for a binomial success rate."""
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _write_success_plots(directory: pathlib.Path, args: Args, aggregate: dict[str, dict[str, Any]]) -> None:
    """Write direct visual comparisons so no separate analysis command is needed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        logging.warning("Could not generate token patch plots: matplotlib is unavailable: %s", error)
        return

    names = list(REGIONS)
    completed = np.asarray([aggregate[name]["completed"] for name in names], dtype=int)
    successes = np.asarray([aggregate[name]["successes"] for name in names], dtype=int)
    failures = np.asarray([aggregate[name]["failures"] for name in names], dtype=int)
    errors = np.asarray([aggregate[name]["errors"] for name in names], dtype=int)
    interrupted = np.asarray([aggregate[name]["interrupted"] for name in names], dtype=int)
    rates = np.divide(successes, completed, out=np.full(len(names), np.nan, dtype=float), where=completed > 0)
    intervals = [_wilson_interval(int(s), int(n)) for s, n in zip(successes, completed)]
    lower = np.asarray([rate - interval[0] if interval and np.isfinite(rate) else 0.0
                        for rate, interval in zip(rates, intervals)])
    upper = np.asarray([interval[1] - rate if interval and np.isfinite(rate) else 0.0
                        for rate, interval in zip(rates, intervals)])

    colours = ["#168163", "#c4463d", "#267e99", "#d28b26", "#8a67a9", "#5f6b73"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": (3, 2)}, constrained_layout=True)
    x = np.arange(len(names))
    valid = completed > 0
    axes[0].bar(x[valid], rates[valid] * 100.0, color=np.asarray(colours)[valid], alpha=0.88,
                yerr=np.vstack((lower[valid] * 100.0, upper[valid] * 100.0)), capsize=5)
    for index, name in enumerate(names):
        if valid[index]:
            axes[0].text(index, min(103.0, rates[index] * 100.0 + upper[index] * 100.0 + 2.0),
                         f"{successes[index]}/{completed[index]}", ha="center", va="bottom", fontsize=9)
        else:
            axes[0].text(index, 50.0, "no completed\nruns", ha="center", va="center", fontsize=8)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title(
        f"Local token patching | {args.delay_target} delay={args.delay_ms} ms | "
        f"site={args.site}, layer={args.layer}\nBars show success rate; whiskers show Wilson 95% intervals"
    )
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.7
    bottom = np.zeros(len(names), dtype=float)
    for values, label, colour in (
        (successes, "success", "#168163"),
        (failures, "failure", "#c4463d"),
        (errors, "error", "#d28b26"),
        (interrupted, "interrupted", "#7d8790"),
    ):
        axes[1].bar(x, values, width=width, bottom=bottom, label=label, color=colour, alpha=0.9)
        bottom += values
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Episode outcomes (errors and interruptions are not in the success-rate denominator)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=25, ha="right")
    fig.savefig(directory / "success_rate.png", dpi=160)
    fig.savefig(directory / "token_patch_comparison.png", dpi=160)
    plt.close(fig)

    selected = np.asarray([
        aggregate[name]["mean_selected_tokens"]
        if aggregate[name]["mean_selected_tokens"] is not None else np.nan
        for name in names
    ], dtype=float)
    patch_queries = np.asarray([aggregate[name]["mean_patch_steps"] for name in names], dtype=float)
    empty_queries = np.asarray([aggregate[name]["mean_empty_mask_steps"] for name in names], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].bar(x, selected, color=np.asarray(colours), alpha=0.88)
    axes[0].set_ylabel("Selected tokens/query")
    axes[0].set_title("Actual spatial token intervention coverage")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - 0.18, patch_queries, width=0.36, label="patch queries", color="#267e99")
    axes[1].bar(x + 0.18, empty_queries, width=0.36, label="empty-mask queries", color="#c4463d")
    axes[1].set_ylabel("Mean queries / episode")
    axes[1].set_title("A valid foreground intervention must have patch queries and no silent fallbacks")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=25, ha="right")
    fig.suptitle(
        f"Token mask diagnostics | {args.delay_target} delay={args.delay_ms} ms | "
        f"threshold={args.token_fraction_threshold:g}"
    )
    fig.savefig(directory / "token_mask_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    print(run(tyro.cli(Args)))
