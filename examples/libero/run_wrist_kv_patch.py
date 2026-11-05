"""Persistent wrist-delay KV repair: pilot controls first, then an explicit layer scan."""

from __future__ import annotations

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
    from examples.libero import run_shift_match_sweep as _shift
except ImportError:
    import run_shift_match_sweep as _shift


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    phase: str = "pilot"
    layers: str = "all"
    site: str = "wrist_kv"
    delay_ms: int = 600
    control_hz: float = 20.0
    repeats_per_setting: int = 30
    seed: int = 7
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None
    replan_steps: int = 5
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    video_fps: int = 10
    save_video: bool = True
    output_dir: str = "rollout/libero_wrist_kv_patch"


@dataclasses.dataclass(frozen=True)
class Setting:
    name: str
    mode: str
    layers: tuple[int, ...]
    delayed: bool
    site: str


def make_settings(phase: str, layers: str, depth: int, site: str = "wrist_kv") -> list[Setting]:
    if phase not in ("pilot", "scan") or depth <= 0:
        raise ValueError("phase must be pilot or scan, with positive model depth")
    all_layers = tuple(range(depth))
    if site not in ("wrist_kv", "siglip", "prefix_hidden", "action_hidden"):
        raise ValueError("unknown activation patch site")
    if site == "siglip":
        all_layers = (0,)
    settings = [
        Setting("clean", "none", (), False, site),
        Setting("delay", "none", (), True, site),
        Setting("self", "self", all_layers, True, site),
        Setting("patch_all", "clean", all_layers, True, site),
    ]
    if phase == "scan":
        selected = all_layers if layers == "all" else tuple(int(value.strip()) for value in layers.split(","))
        if not selected or len(set(selected)) != len(selected) or any(layer not in all_layers for layer in selected):
            raise ValueError("layers must be all or unique valid zero-based layer numbers")
        settings.extend(Setting(f"layer_{layer:02d}", "clean", (layer,), True, site) for layer in selected)
    elif layers != "all":
        raise ValueError("Choose --phase scan to specify layers")
    return settings


def paired_noise(seed, repeat_index, step, shape):
    return np.random.default_rng(np.random.SeedSequence([seed, repeat_index, step])).standard_normal(shape).astype(
        np.float32
    )


def _control_hz(env):
    for candidate in (env, getattr(env, "env", None)):
        value = getattr(candidate, "control_freq", None)
        if value is not None:
            return float(value)
    raise ValueError("Cannot verify simulator control_freq; refusing to label an unknown delay as milliseconds")


def _run_episode(*, args, client, task, initial_state, prompt, setting, repeat_index, noise_shape,
                 delay_steps, max_steps, output_dir):
    env = None
    frames, traces, chunks, noises, executed, states, hashes = [], [], [], [], [], [], []
    image_history = {name: collections.deque(maxlen=delay_steps + 1) for name in _shift.CAMERA_SOURCES}
    state_history = collections.deque(maxlen=delay_steps + 1)
    plan = collections.deque()
    outcome, error_text, actual_hz = "failure", None, None
    warmup = max(args.num_steps_wait, delay_steps)
    try:
        env = _shift._create_env(task, args.seed + repeat_index)
        actual_hz = _control_hz(env)
        if not np.isclose(actual_hz, args.control_hz):
            raise ValueError(f"Configured control_hz={args.control_hz}, simulator control_freq={actual_hz}")
        env.reset()
        obs = env.set_init_state(initial_state)
        _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
        for _ in range(warmup):
            obs, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
            if done:
                raise RuntimeError("Task ended during warmup; no valid intervention episode")
        for step in range(max_steps):
            current = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target=_shift.BASELINE_TARGET, delay_steps=0, prompt=prompt,
            )
            recipient = _shift._build_delayed_observation(
                image_history=image_history, state_history=state_history,
                delay_target="observation/wrist_image" if setting.delayed else _shift.BASELINE_TARGET,
                delay_steps=delay_steps if setting.delayed else 0, prompt=prompt,
            )
            query = not plan
            if args.save_video:
                frames.append(_shift._make_video_frame(
                    current_images={key: current[key] for key in _shift.CAMERA_KEYS},
                    policy_images={key: recipient[key] for key in _shift.CAMERA_KEYS},
                    current_state=current["observation/state"], policy_state=recipient["observation/state"],
                    delay_target="observation/wrist_image" if setting.delayed else _shift.BASELINE_TARGET,
                    effective_delay_ms=delay_steps / actual_hz * 1000 if setting.delayed else 0,
                    step=step, policy_query=query,
                ))
            if query:
                noise = paired_noise(args.seed, repeat_index, step, noise_shape)
                response = client.infer_wrist_patch(
                    recipient, current["observation/wrist_image"], noise,
                    mode=setting.mode, layers=list(setting.layers), site=setting.site,
                )
                expected = {
                    "schema_version": 1, "mode": setting.mode,
                    "layers": list(setting.layers), "site": setting.site,
                }
                if response.get("wrist_kv_patch") != expected:
                    raise RuntimeError("Server did not acknowledge the requested KV intervention")
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (noise_shape[0], 7) or not np.all(np.isfinite(actions)):
                    raise ValueError(f"Expected finite LIBERO actions {(noise_shape[0], 7)}, got {actions.shape}")
                plan.extend(actions[:args.replan_steps])
                traces.append((step, step - delay_steps if setting.delayed else step))
                chunks.append(actions)
                noises.append(noise)
                states.append(current["observation/state"])
                hashes.append([
                    hashlib.sha256(np.ascontiguousarray(obs_image).tobytes()).hexdigest()
                    for obs_image in (recipient["observation/wrist_image"], current["observation/wrist_image"])
                ])
            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            executed.append(action)
            _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
            if done:
                outcome = "success"
                break
    except KeyboardInterrupt:
        outcome, error_text = "interrupted", "KeyboardInterrupt"
    except Exception as error:
        outcome, error_text = "error", repr(error)
        logging.exception("Wrist KV experiment failed: %s repeat=%d", setting.name, repeat_index)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("Environment cleanup failed")

    slug = f"{setting.name}_repeat_{repeat_index:03d}"
    trace_path = output_dir / f"{slug}_trace.npz"
    np.savez_compressed(
        trace_path, query_steps=np.asarray(traces, dtype=np.int32).reshape(-1, 2),
        action_chunks=np.asarray(chunks, dtype=np.float32), noise=np.asarray(noises, dtype=np.float32),
        executed_actions=np.asarray(executed, dtype=np.float32), states=np.asarray(states, dtype=np.float32),
        wrist_sha256=np.asarray(hashes), patch_layers=np.asarray(setting.layers, dtype=np.int32),
    )
    video_path = output_dir / f"{slug}_{outcome}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps) if frames else (False, None)
    return {
        "condition": setting.name, "patch_site": setting.site,
        "patch_mode": setting.mode, "patch_layers": list(setting.layers),
        "repeat_index": repeat_index, "seed": args.seed + repeat_index,
        "task_suite_name": args.task_suite_name, "task_id": args.task_id, "initial_state_id": args.initial_state_id,
        "prompt": prompt, "outcome": outcome, "success": outcome == "success", "steps": len(executed),
        "n_queries": len(traces), "warmup_steps": warmup, "control_hz": actual_hz,
        "delay_steps": delay_steps if setting.delayed else 0,
        "effective_delay_ms": delay_steps / args.control_hz * 1000 if setting.delayed else 0,
        "trace_path": str(trace_path), "video_path": str(video_path) if video_ok else None,
        "video_error": video_error, "error": error_text,
    }


def _save_summary(output_dir, args, capability, settings, records):
    report = {
        "experiment": "libero_wrist_kv_patch", "schema_version": 1,
        "config": dataclasses.asdict(args), "server_capability": capability,
        "conditions": [dataclasses.asdict(setting) for setting in settings], "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if records:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def run(args: Args):
    if args.phase not in ("pilot", "scan") or args.repeats_per_setting <= 0 or args.seed < 0:
        raise ValueError("Invalid phase, repeats, or seed")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0 or args.delay_ms <= 0:
        raise ValueError("Delay and control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.video_fps) <= 0 or args.num_steps_wait < 0:
        raise ValueError("Invalid rollout configuration")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    delay_float = args.delay_ms * args.control_hz / 1000
    delay_steps = round(delay_float)
    if delay_steps < 1 or not np.isclose(delay_steps, delay_float):
        raise ValueError("delay_ms must map to an integer number of control steps")
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError("Unsupported task suite")
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError("task_id out of range")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(states):
        raise ValueError("initial_state_id out of range")
    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(task.language)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        capability = client.get_server_metadata().get("wrist_kv_patch", {})
        if capability.get("schema_version") != 1:
            raise RuntimeError("Update server/client for wrist_kv_patch schema v1")
        shape = tuple(capability["noise_shape"])
        if len(shape) != 2 or min(shape) <= 0 or args.replan_steps > shape[0]:
            raise ValueError("Invalid noise shape or replan_steps exceeds action horizon")
        settings = make_settings(args.phase, args.layers, capability["num_layers"], args.site)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output_dir = pathlib.Path(args.output_dir) / f"{stamp}_{args.phase}"
        output_dir.mkdir(parents=True, exist_ok=False)
        records = []
        _save_summary(output_dir, args, capability, settings, records)
        logging.info("Running %s: %d conditions x %d repeats; wrist delay %d steps (%d ms)",
                     args.phase, len(settings), args.repeats_per_setting, delay_steps, args.delay_ms)
        # Rotate order within paired repeats to avoid always running one condition first.
        for repeat in range(args.repeats_per_setting):
            order = np.random.default_rng(np.random.SeedSequence([args.seed, repeat, 987])).permutation(len(settings))
            for index in order:
                record = _run_episode(
                    args=args, client=client, task=task, initial_state=states[args.initial_state_id], prompt=prompt,
                    setting=settings[index], repeat_index=repeat, noise_shape=shape, delay_steps=delay_steps,
                    max_steps=args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name], output_dir=output_dir,
                )
                records.append(record)
                _save_summary(output_dir, args, capability, settings, records)
                logging.info("[%d/%d] %s repeat=%d outcome=%s steps=%d", len(records),
                             len(settings) * args.repeats_per_setting, record["condition"], repeat,
                             record["outcome"], record["steps"])
                if record["outcome"] == "error":
                    raise RuntimeError(f"Sweep stopped; partial results saved in {output_dir}: {record['error']}")
                if record["outcome"] == "interrupted":
                    return output_dir / "summary.json"
        return output_dir / "summary.json"
    finally:
        client._ws.close()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    print(run(tyro.cli(Args)))
