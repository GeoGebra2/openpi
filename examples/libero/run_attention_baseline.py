"""Record action-expert attention during normal, zero-delay LIBERO rollouts.

The robot follows the ordinary closed-loop policy.  Attention is queried at
fixed environment-step intervals (``record_every_env_steps``), including an
extra diagnostic query when the action chunk is still active.  Diagnostic
actions are always discarded, so the recorded attention cannot change the
control trajectory.

Start the policy server with ``--diagnostics``.  Each completed episode writes
an ``*_attention.npz`` file containing the environment step, denoising step,
layer, head and source-group axes, plus a compact JSON summary.
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

try:
    from examples.libero import run_shift_match_sweep as _shift
except ImportError:  # running as a plain script with examples/libero on sys.path
    import run_shift_match_sweep as _shift


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    record_every_env_steps: int = 5

    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    prompt: Optional[str] = None

    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    repeats_per_setting: int = 3
    seed: int = 7
    video_fps: int = 10
    output_dir: str = "rollout/libero_spatial_task_000_attention_baseline"


def run_attention_baseline(args: Args) -> pathlib.Path:
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
        raise ValueError(f"initial_state_id must be in [0, {len(initial_states) - 1}], got {args.initial_state_id}")

    task = task_suite.get_task(args.task_id)
    task_description = str(task.language)
    policy_prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else task_description
    max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    output_dir = pathlib.Path(args.output_dir) / started_at.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    attention_meta = metadata.get("attention", {})
    if attention_meta.get("schema_version") != 1:
        raise RuntimeError("The policy server does not expose attention schema v1; restart it with --diagnostics")
    source_groups = list(attention_meta.get("source_groups", []))
    noise_shape = tuple(int(value) for value in attention_meta.get("noise_shape", []))
    if len(noise_shape) != 2:
        raise RuntimeError(f"Invalid attention noise_shape in server metadata: {noise_shape}")

    logging.info("Running %d normal attention rollouts", args.repeats_per_setting)
    logging.info("Task: %s / %d / initial state %d", args.task_suite_name, args.task_id, args.initial_state_id)
    logging.info("Recording attention every %d environment steps", args.record_every_env_steps)
    records: list[dict[str, Any]] = []
    try:
        for repeat_index in range(args.repeats_per_setting):
            record = _run_episode(
                args=args,
                client=client,
                task=task,
                initial_state=initial_states[args.initial_state_id],
                task_description=task_description,
                policy_prompt=policy_prompt,
                repeat_index=repeat_index,
                run_index=repeat_index + 1,
                max_steps=max_steps,
                noise_shape=noise_shape,
                source_groups=source_groups,
                output_dir=output_dir,
            )
            records.append(record)
            _write_summary(output_dir, args, records, source_groups)
            logging.info(
                "[%d/%d] outcome=%s success=%s steps=%s attention_records=%s",
                len(records), args.repeats_per_setting, record["outcome"], record["success"],
                record["steps"], record["n_attention_records"],
            )
            if record["outcome"] in ("error", "interrupted"):
                break
    except KeyboardInterrupt:
        logging.warning("Attention baseline interrupted after %d completed runs", len(records))
        _write_summary(output_dir, args, records, source_groups)

    summary_path = output_dir / "summary.json"
    logging.info("Saved attention baseline summary: %s", summary_path)
    return summary_path


def _run_episode(
    *, args: Args, client: Any, task, initial_state, task_description: str, policy_prompt: str,
    repeat_index: int, run_index: int, max_steps: int, noise_shape: tuple[int, ...],
    source_groups: list[str], output_dir: pathlib.Path,
) -> dict[str, Any]:
    env = _shift._create_env(task, args.seed + repeat_index)
    image_history = {source: collections.deque(maxlen=1) for source in _shift.CAMERA_SOURCES}
    state_history = collections.deque(maxlen=1)
    action_plan = collections.deque()
    frames: list[np.ndarray] = []
    attention_steps: list[int] = []
    source_mass: list[np.ndarray] = []
    first_action_mass: list[np.ndarray] = []
    token_counts: list[np.ndarray] = []
    rtt_ms: list[float] = []
    n_policy_queries = 0
    n_attention_queries = 0
    n_attention_only_queries = 0
    outcome = "failure"
    error_text = None
    steps_completed = 0

    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        _shift._append_snapshot(image_history, state_history, obs, args.resize_size)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            _shift._append_snapshot(image_history, state_history, obs, args.resize_size)

        for step in range(max_steps):
            current_images = {
                key: image_history[source][-1] for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)
            }
            current_state = np.array(state_history[-1], copy=True)
            observation = {
                **current_images,
                "observation/state": current_state,
                "prompt": policy_prompt,
            }
            will_query = not action_plan
            frames.append(
                _shift._make_video_frame(
                    current_images=current_images,
                    policy_images=current_images,
                    current_state=current_state,
                    policy_state=current_state,
                    delay_target=_shift.BASELINE_TARGET,
                    effective_delay_ms=0.0,
                    step=step,
                    policy_query=will_query,
                )
            )

            record_attention = step % args.record_every_env_steps == 0
            if will_query:
                n_policy_queries += 1
                noise = _paired_noise(args.seed, repeat_index, step, noise_shape)
                started = time.monotonic()
                response = client.infer_with_noise(observation, noise=noise)
                rtt_ms.append((time.monotonic() - started) * 1000.0)
                actions = np.asarray(response["actions"], dtype=np.float32)
                if len(actions) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(actions)} actions, fewer than "
                        f"replan_steps={args.replan_steps}"
                    )
                action_plan.extend(actions[: args.replan_steps])

                if record_attention:
                    n_attention_queries += 1
                    started = time.monotonic()
                    attention_response = client.infer_attention(observation, noise=noise)
                    rtt_ms.append((time.monotonic() - started) * 1000.0)
                    _append_attention_response(
                        attention_response,
                        attention_steps,
                        source_mass,
                        first_action_mass,
                        token_counts,
                        step,
                        source_groups,
                    )
            elif record_attention:
                # Keep the executed trajectory unchanged. This query samples attention for the
                # current observation at a fixed environment-step interval, but its actions are
                # intentionally discarded because the existing action chunk is still active.
                n_attention_queries += 1
                n_attention_only_queries += 1
                started = time.monotonic()
                response = client.infer_attention(
                    observation, noise=_paired_noise(args.seed, repeat_index, step, noise_shape)
                )
                rtt_ms.append((time.monotonic() - started) * 1000.0)
                _append_attention_response(
                    response,
                    attention_steps,
                    source_mass,
                    first_action_mass,
                    token_counts,
                    step,
                    source_groups,
                )

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
    except Exception as error:
        outcome = "error"
        error_text = repr(error)
        logging.exception("Attention baseline run failed")
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO environment")

    run_slug = f"run_{run_index:03d}_baseline_repeat_{repeat_index:02d}"
    attention_path: Optional[pathlib.Path] = None
    if source_mass:
        attention_path = output_dir / f"{run_slug}_attention.npz"
        np.savez_compressed(
            attention_path,
            env_steps=np.asarray(attention_steps, dtype=np.int32),
            is_policy_query=np.asarray([step % args.replan_steps == 0 for step in attention_steps]),
            source_mass=np.stack(source_mass).astype(np.float32, copy=False),
            first_action_source_mass=np.stack(first_action_mass).astype(np.float32, copy=False),
            source_token_counts=np.stack(token_counts).astype(np.float32, copy=False),
            source_groups=np.asarray(source_groups),
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
        "delay_target": _shift.BASELINE_TARGET,
        "delay_ms": 0,
        "replan_steps": args.replan_steps,
        "record_every_env_steps": args.record_every_env_steps,
        "outcome": outcome,
        "success": outcome == "success",
        "steps": steps_completed,
        "n_policy_queries": n_policy_queries,
        "n_attention_queries": n_attention_queries,
        "n_attention_only_queries": n_attention_only_queries,
        "n_attention_records": len(attention_steps),
        "infer_rtt_ms_mean": _stat(rtt_ms, "mean"),
        "infer_rtt_ms_max": _stat(rtt_ms, "max"),
        "attention_path": str(attention_path) if attention_path else None,
        "video_path": str(video_path) if video_written else None,
        "video_error": video_error,
        "error": error_text,
    }


def _append_attention_response(
    response: dict[str, Any],
    attention_steps: list[int],
    source_mass: list[np.ndarray],
    first_action_mass: list[np.ndarray],
    token_counts: list[np.ndarray],
    step: int,
    source_groups: list[str],
) -> None:
    attention = response.get("attention")
    required = ("source_mass", "first_action_source_mass", "source_token_counts")
    if not isinstance(attention, dict) or any(key not in attention for key in required):
        raise RuntimeError("Attention response is missing source_mass/first_action_source_mass/source_token_counts")
    response_groups = response.get("attention_source_groups")
    if response_groups is None or list(response_groups) != source_groups:
        raise RuntimeError(
            f"Attention source groups differ from server metadata: response={response_groups}, "
            f"metadata={source_groups}"
        )
    mass = np.asarray(attention["source_mass"], dtype=np.float32)
    first_mass = np.asarray(attention["first_action_source_mass"], dtype=np.float32)
    counts = np.asarray(attention["source_token_counts"], dtype=np.float32)
    if (
        mass.ndim != 5
        or first_mass.shape != mass.shape
        or counts.shape != (len(source_groups),)
        or mass.shape[-1] != len(source_groups)
    ):
        raise RuntimeError(
            "Invalid attention shapes: expected source_mass/first_action_source_mass "
            "[denoise, layer, kv, query, source] "
            f"and source_token_counts [source], got {mass.shape}, {first_mass.shape}, {counts.shape}"
        )
    if any(not np.all(np.isfinite(value)) for value in (mass, first_mass, counts)):
        raise RuntimeError("Attention response contains non-finite values")
    source_mass.append(mass)
    first_action_mass.append(first_mass)
    token_counts.append(counts)
    attention_steps.append(step)


def _paired_noise(seed: int, repeat_index: int, step: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, repeat_index, step]))
    return rng.standard_normal(shape).astype(np.float32)


def _stat(values: list[float], name: str) -> Optional[float]:
    if not values:
        return None
    if name == "mean":
        return round(float(np.mean(values)), 3)
    if name == "max":
        return round(float(np.max(values)), 3)
    raise ValueError(f"Unknown statistic {name!r}")


def _aggregate_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["outcome"] in ("success", "failure")]
    successes = sum(bool(record["success"]) for record in completed)
    return {
        "runs": len(records),
        "completed_runs": len(completed),
        "successes": successes,
        "failures": len(completed) - successes,
        "errors": sum(record["outcome"] == "error" for record in records),
        "interrupted": sum(record["outcome"] == "interrupted" for record in records),
        "success_rate": round(successes / len(completed), 4) if completed else None,
    }


def _write_summary(
    output_dir: pathlib.Path, args: Args, records: list[dict[str, Any]], source_groups: list[str]
) -> None:
    summary = {
        "experiment": "libero_attention_baseline",
        "config": dataclasses.asdict(args),
        "source_groups": source_groups,
        "attention_axes": ["record", "denoise_step", "layer", "kv_head", "query_head", "source_group"],
        "aggregate": _aggregate_results(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def _validate_args(args: Args) -> None:
    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE:
        raise ValueError(f"Unsupported task suite {args.task_suite_name!r}")
    if args.replan_steps <= 0 or args.record_every_env_steps <= 0:
        raise ValueError("replan_steps and record_every_env_steps must be positive")
    if args.repeats_per_setting <= 0:
        raise ValueError("repeats_per_setting must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if args.resize_size <= 0 or args.video_fps <= 0:
        raise ValueError("resize_size and video_fps must be positive")


if __name__ == "__main__":
    import tyro  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    run_attention_baseline(tyro.cli(Args))
