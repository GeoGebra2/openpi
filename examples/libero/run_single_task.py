"""Run one LIBERO task against an openpi policy server and save its rollout."""

import collections
import dataclasses
import datetime
import logging
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

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
    # Policy server settings.
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Scene selection. Task and initial-state IDs are zero-based.
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0

    # Set this in code or pass --prompt. None/empty uses the task's original instruction.
    prompt: Optional[str] = None

    # Rollout settings. None uses the suite-specific limit from the benchmark runner.
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7

    # Videos are named with suite, task ID, save time, and outcome.
    video_out_path: str = "rollout"
    video_fps: int = 10


def run_single_task(args: Args) -> pathlib.Path:
    _validate_basic_args(args)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        available = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"Unknown task suite {args.task_suite_name!r}. Available suites: {available}")

    task_suite = benchmark_dict[args.task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(
            f"task_id must be in [0, {task_suite.n_tasks - 1}] for {args.task_suite_name}, got {args.task_id}"
        )

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_id < len(initial_states):
        raise ValueError(
            f"initial_state_id must be in [0, {len(initial_states) - 1}] for task {args.task_id}, "
            f"got {args.initial_state_id}"
        )

    task_description = str(task.language)
    policy_prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else task_description
    max_steps = args.max_steps or MAX_STEPS_BY_SUITE[args.task_suite_name]

    logging.info("Task suite: %s", args.task_suite_name)
    logging.info("Task ID: %d", args.task_id)
    logging.info("Initial-state ID: %d", args.initial_state_id)
    logging.info("LIBERO task description: %s", task_description)
    logging.info("Policy prompt: %s", policy_prompt)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = _create_env(task, args.seed)
    frames = []
    outcome = "failure"
    pending_error = None

    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.initial_state_id])
        action_plan = collections.deque()

        for step in range(max_steps + args.num_steps_wait):
            if step < args.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
            )
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
            )
            frames.append(image)

            if not action_plan:
                observation = {
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": policy_prompt,
                }
                action_chunk = client.infer(observation)["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, fewer than replan_steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            if done:
                outcome = "success"
                break
    except KeyboardInterrupt as error:
        outcome = "interrupted"
        pending_error = error
        logging.warning("Rollout interrupted; saving collected frames before exiting")
    except Exception as error:
        outcome = "error"
        pending_error = error
        logging.exception("Rollout failed; saving collected frames before exiting")
    finally:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close the LIBERO environment")

    logging.info("Outcome: %s", outcome)
    video_path = None
    if frames:
        video_path = _save_video(args, frames, outcome)
        logging.info("Saved rollout video: %s", video_path)
    else:
        logging.warning("No rollout frames were captured; no video was saved")

    if pending_error is not None:
        raise pending_error
    if video_path is None:
        raise RuntimeError("The rollout completed without capturing any video frames")
    return video_path


def _validate_basic_args(args: Args) -> None:
    if args.task_suite_name not in MAX_STEPS_BY_SUITE:
        available = ", ".join(MAX_STEPS_BY_SUITE)
        raise ValueError(f"Unsupported task suite {args.task_suite_name!r}. Expected one of: {available}")
    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {args.replan_steps}")
    if args.num_steps_wait < 0:
        raise ValueError(f"num_steps_wait must be non-negative, got {args.num_steps_wait}")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError(f"max_steps must be positive when set, got {args.max_steps}")
    if args.resize_size <= 0:
        raise ValueError(f"resize_size must be positive, got {args.resize_size}")
    if args.video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {args.video_fps}")


def _create_env(task, seed: int) -> OffScreenRenderEnv:
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _save_video(args: Args, frames, outcome: str) -> pathlib.Path:
    output_dir = pathlib.Path(args.video_out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.task_suite_name}_task_{args.task_id:03d}_{saved_at}_{outcome}.mp4"
    video_path = output_dir / filename
    imageio.mimwrite(video_path, [np.asarray(frame) for frame in frames], fps=args.video_fps)
    return video_path


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_single_task(tyro.cli(Args))
