"""Measure layer-wise representation drift of PI0 under delayed LIBERO observations.

Companion experiment to the failure-rate and shift-match delay sweeps. Instead of
asking *whether* delayed observations hurt, this sweep asks *where* the delay's
effect lives inside the network: it captures per-layer activations for a delayed
query and a fully current "shadow" query at the same environment step, then
compares them layer by layer (SigLIP blocks -> Gemma prefix layers -> action-expert
layers -> flow vector field). Cross-module distances alone do not establish amplification.

Rollout mechanics match ``run_shift_match_sweep.py`` (the delayed query drives the
robot every ``replan_steps`` env steps), EXCEPT the shadow query (fully current
observation, never executed) only runs at replan steps, not every step — drift is
only defined where both inferences exist.

REQUIRES a policy server started with ``--diagnostics`` (see scripts/serve_policy.py);
schema-v2 paired inference is mandatory. Automatic consistency checks are disabled.

Each pair shares explicit initial noise. Baseline runs (``__none__``, zero delay)
must therefore have near-zero drift. This is still a CLOSED-LOOP experiment:
different delay settings can visit different states, even with paired noise.

Every run writes a ``*_drift.npz`` with per-query drift arrays plus a policy-input
video, and the sweep rewrites JSON/CSV summaries after every run. Analyze with
``analyze_activation_drift.py``.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import datetime
import json
import logging
import pathlib
import time
from typing import Any, Optional

import numpy as np
from openpi_client.activation_drift import SCHEMA_VERSION
from openpi_client.activation_drift import compute_drift  # noqa: F401 - public analysis helper

try:
    from examples.libero import run_shift_match_sweep as _shift
except ImportError:  # running as a plain script with examples/libero on sys.path
    import run_shift_match_sweep as _shift


class MissingDiagnosticsError(RuntimeError):
    """Raised when the server lacks the required paired diagnostics schema."""


class PolicyInferenceError(RuntimeError):
    """Fatal policy request failure; subsequent rollouts must not reuse the connection."""


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

    # Activation-drift experiment. Values are quantized to simulator control steps.
    delay_targets: str = "observation/image,observation/wrist_image,state"
    delay_levels_ms: str = "200,400,600,800"
    control_hz: float = 20.0
    repeats_per_setting: int = 3
    include_baseline: bool = True

    # Rollout and output.
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7
    video_fps: int = 10
    output_dir: str = "rollout/libero_spatial_task_000_activation_drift"


# ---------------------------------------------------------------------------
# Persist compact per-query drift statistics.
# ---------------------------------------------------------------------------


def _flatten_drift(drift: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested drift dict into npz-friendly 'a/b/c' -> array entries."""
    flat: dict[str, np.ndarray] = {}
    for key, value in drift.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_drift(value, prefix=f"{name}/"))
        else:
            flat[name] = np.asarray(value)
    return flat


def _require_diagnostics(response: dict[str, Any], query_kind: str) -> None:
    if response.get("schema_version") != SCHEMA_VERSION or "drift" not in response:
        raise MissingDiagnosticsError(
            f"The policy server's {query_kind} response lacks schema-v2 paired token diagnostics. "
            "Update both server and client and restart the server with --diagnostics."
        )


def _infer_with_diagnostics(client: Any, observation: dict[str, Any], reference: dict[str, Any], **kwargs):
    try:
        response = client.infer_drift(observation, reference, **kwargs)
    except Exception as error:
        raise PolicyInferenceError("Paired policy query failed; aborting activation-drift sweep.") from error
    _require_diagnostics(response, "paired")
    return response


def _paired_noise(seed: int, repeat_index: int, step: int, shape: tuple[int, ...]) -> np.ndarray:
    # The schedule is independent of delay setting and of previous episode lengths.
    rng = np.random.default_rng(np.random.SeedSequence([seed, repeat_index, step]))
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Sweep driver.
# ---------------------------------------------------------------------------


def run_activation_drift_sweep(args: Args) -> pathlib.Path:
    """Run all requested drift settings and return the summary JSON path."""
    from libero.libero import benchmark  # noqa: PLC0415
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    _shift._validate_args(args)
    if args.seed < 0:
        raise ValueError("Seed must be nonnegative")
    delay_targets = _shift._split_csv(args.delay_targets)
    delay_levels = _shift._parse_delay_levels(args.delay_levels_ms)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite {args.task_suite_name!r}; available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[args.task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}], got {args.task_id}")
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(f"initial_state_id must be in [0, {len(initial_states) - 1}], got {args.initial_state_id}")

    task = task_suite.get_task(args.task_id)
    task_description = str(task.language)
    policy_prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else task_description
    max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
    settings = _shift._make_settings(args, delay_targets, delay_levels)
    delay_steps_max = _shift._delay_to_steps(max(delay_levels), args.control_hz)
    warmup_steps = max(args.num_steps_wait, delay_steps_max)
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(settings) * args.repeats_per_setting
    logging.info("Running %d activation-drift rollouts", total_runs)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
    logging.info("Delay targets: %s", ", ".join(delay_targets))
    logging.info("Policy prompt: %s", policy_prompt)
    logging.info("Closed-loop paired queries at replan steps; shared initial noise; consistency checks disabled")

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    capability = client.get_server_metadata().get("activation_drift", {})
    if capability.get("schema_version") != SCHEMA_VERSION:
        raise MissingDiagnosticsError("Update server/client and restart with --diagnostics; schema v2 is required")
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
    except (MissingDiagnosticsError, PolicyInferenceError):
        _write_summary(output_dir, args, records)
        raise
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
    setting: _shift.DelaySetting,
    repeat_index: int,
    run_index: int,
    max_steps: int,
    warmup_steps: int,
    delay_steps_max: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    delay_steps = _shift._delay_to_steps(setting.delay_ms, args.control_hz)
    effective_delay_ms = round(delay_steps * 1000.0 / args.control_hz, 3)
    env = _shift._create_env(task, args.seed)
    frames: list[np.ndarray] = []
    action_plan = collections.deque()
    image_history = {source: collections.deque(maxlen=delay_steps_max + 1) for source in _shift.CAMERA_SOURCES}
    state_history = collections.deque(maxlen=delay_steps_max + 1)
    outcome = "failure"
    error_text = None
    steps_completed = 0
    pair_rtt_ms: list[float] = []
    replan_step_indices: list[int] = []
    drift_entries: list[dict[str, Any]] = []
    chunk_shape: Optional[tuple[int, ...]] = None
    noise_shape = tuple(client.get_server_metadata()["activation_drift"]["noise_shape"])

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
        for _ in range(warmup_steps):
            obs, _, _, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _shift._append_snapshot(image_history, state_history, obs, args.resize_size)

        for step in range(max_steps):
            current_images = {
                key: image_history[source][-1] for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)
            }
            current_state = np.array(state_history[-1], copy=True)
            policy_observation = _shift._build_delayed_observation(
                image_history=image_history,
                state_history=state_history,
                delay_target=setting.delay_target,
                delay_steps=delay_steps,
                prompt=policy_prompt,
            )
            will_query_policy = not action_plan
            frames.append(
                _shift._make_video_frame(
                    current_images=current_images,
                    policy_images={key: policy_observation[key] for key in _shift.CAMERA_KEYS},
                    current_state=current_state,
                    policy_state=policy_observation["observation/state"],
                    delay_target=setting.delay_target,
                    effective_delay_ms=effective_delay_ms,
                    step=step,
                    policy_query=will_query_policy,
                )
            )

            if will_query_policy:
                # Shadow query: fully current observation, never executed. Drift is
                # only defined where both delayed and current activations exist.
                shadow_observation = {
                    **current_images,
                    "observation/state": current_state,
                    "prompt": policy_prompt,
                }
                noise = _paired_noise(args.seed, repeat_index, step, noise_shape)
                infer_started = time.monotonic()
                response = _infer_with_diagnostics(
                    client,
                    policy_observation,
                    shadow_observation,
                    noise=noise,
                    check_consistency=False,
                )
                pair_rtt_ms.append((time.monotonic() - infer_started) * 1000.0)

                drift_entries.append(response["drift"])
                replan_step_indices.append(step)

                action_chunk = np.asarray(response["actions"], dtype=np.float32)
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
                if chunk_shape is None:
                    chunk_shape = action_chunk.shape
                elif action_chunk.shape != chunk_shape:
                    raise ValueError(f"Policy returned chunk with shape {action_chunk.shape}, expected {chunk_shape}")
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
            steps_completed = step + 1
            if done:
                outcome = "success"
                break
    except KeyboardInterrupt:
        outcome = "interrupted"
        error_text = "KeyboardInterrupt"
    except (MissingDiagnosticsError, PolicyInferenceError):
        # Policy failures invalidate the sweep; don't reuse a possibly closed connection.
        raise
    except Exception as error:  # Keep the sweep running after one bad setting.
        outcome = "error"
        error_text = repr(error)
        logging.exception("Activation-drift run failed: target=%s delay=%d ms", setting.delay_target, setting.delay_ms)
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    run_slug = (
        f"run_{run_index:03d}_{_shift._slug(setting.delay_target)}_{setting.delay_ms:04d}ms_repeat_{repeat_index:02d}"
    )
    drift_path: Optional[pathlib.Path] = None
    if drift_entries:
        output_dir.mkdir(parents=True, exist_ok=True)
        drift_path = output_dir / f"{run_slug}_drift.npz"
        flattened = [_flatten_drift(entry) for entry in drift_entries]
        np.savez_compressed(
            drift_path,
            steps=np.asarray(replan_step_indices, dtype=np.int32),
            **{
                key: np.stack([entry[key] for entry in flattened]).astype(np.float32, copy=False)
                for key in flattened[0]
            },
        )

    video_path = output_dir / f"{run_slug}_{outcome}.mp4"
    video_written, video_error = _shift._save_video(video_path, frames, args.video_fps)

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
        "noise_shape": list(noise_shape),
        "outcome": outcome,
        "success": outcome == "success",
        "steps": steps_completed,
        "n_drift_queries": len(drift_entries),
        "pair_rtt_ms_mean": _shift._stat(pair_rtt_ms, "mean"),
        "pair_rtt_ms_max": _shift._stat(pair_rtt_ms, "max"),
        "drift_path": str(drift_path) if drift_path else None,
        "video_path": str(video_path) if video_written else None,
        "video_error": video_error,
        "error": error_text,
    }


def _write_summary(output_dir: pathlib.Path, args: Args, records: list[dict[str, Any]]) -> None:
    summary = {
        "experiment": "libero_activation_drift_sweep",
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "trajectory": "closed_loop",
            "noise": "paired_shared_initial",
            "noise_schedule": "numpy PCG64 / SeedSequence([seed, repeat_index, env_step])",
            "statistics": "masked_tokenwise_then_mean_and_p90",
            "effective_action_dim": 7,
            "gripper_slice": [6, 7],
            "aggregation": "query_weighted_mean; P90 values are averaged per-query token P90s",
            "checks": "disabled",
        },
        "config": dataclasses.asdict(args),
        "aggregate": _shift._aggregate_results(records),
        "drift": _aggregate_drift(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not records:
        return
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _aggregate_drift(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of every drift array over queries and runs, per (delay_target, delay_ms)."""
    # (target, delay_ms) -> flat drift key -> [running sum over queries, query count]
    accumulators: dict[tuple[str, int], dict[str, list[Any]]] = {}
    for record in records:
        drift_path = record.get("drift_path")
        if not drift_path:
            continue
        group = accumulators.setdefault((record["delay_target"], record["delay_ms_requested"]), {})
        with np.load(drift_path) as data:
            for key in data.files:
                if key == "steps":
                    continue
                values = np.asarray(data[key], dtype=np.float64)
                if key not in group:
                    group[key] = [np.zeros(values.shape[1:], dtype=np.float64), 0]
                group[key][0] += values.sum(axis=0)
                group[key][1] += values.shape[0]
    result: dict[str, Any] = {}
    for (target, delay_ms), group in accumulators.items():
        result.setdefault(target, {})[str(delay_ms)] = {
            key: (total / count).tolist() for key, (total, count) in sorted(group.items())
        }
    return result


if __name__ == "__main__":
    import tyro  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    run_activation_drift_sweep(tyro.cli(Args))
