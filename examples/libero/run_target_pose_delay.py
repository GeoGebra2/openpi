"""Measure the effect of a stale target pose without image compositing.

The selected camera receives one of three inputs at each policy query:

``clean``
    The current simulator render.
``target_pose_stale``
    A current-scene render in which only the target object is temporarily
    placed at its delayed **world-frame** pose.  The current robot and camera
    remain fixed, so the target does not follow the current end effector.  The
    real object and simulator state are restored before the action is executed.
``full_delay``
    The complete camera image from ``delay_steps`` earlier.

The second condition is a renderer-level counterfactual.  It keeps the current
robot, camera, table and all other objects, so it does not create the robot
silhouettes caused by pixel inpainting.  This is still an image intervention:
the executed action interacts with the real current object, not the temporary
rendered object.
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


EXPERIMENT = "libero_target_pose_delay"
SCHEMA_VERSION = 2
CONDITIONS = ("clean", "target_pose_stale", "full_delay")
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
    render_size: int = 256
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    output_dir: str = "rollout/libero_target_pose_delay"
    skip_analysis: bool = False
    interactive: bool = False
    threshold_cm: float = 2.0
    consecutive: int = 3
    pair_limit: int = 10
    object_names: str = ""
    target_pose_mode: str = "table_stale"
    table_z_tolerance_m: float = 0.025
    debug_render_pose: bool = False


@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    delay_ms: int
    delay_steps: int


@dataclasses.dataclass
class Snapshot:
    images: dict[str, np.ndarray]
    eef_pos: np.ndarray
    eef_quat_xyzw: np.ndarray
    object_pos: np.ndarray
    object_quat_xyzw: np.ndarray


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
        raise ValueError(f"delay_target must be one of {CAMERA_TARGETS}")
    if args.delay_ms <= 0 or args.repeats_per_condition <= 0 or args.seed < 0:
        raise ValueError("delay_ms/repeats_per_condition must be positive and seed non-negative")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if min(args.replan_steps, args.resize_size, args.render_size, args.video_fps) <= 0 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size, render size, video rate or warmup")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.threshold_cm <= 0 or args.consecutive < 1 or args.pair_limit < 0:
        raise ValueError("Invalid analysis threshold, consecutive count or pair limit")
    if args.target_pose_mode not in ("table_stale", "world_delay"):
        raise ValueError("target_pose_mode must be 'table_stale' or 'world_delay'")
    if not np.isfinite(args.table_z_tolerance_m) or args.table_z_tolerance_m <= 0:
        raise ValueError("table_z_tolerance_m must be positive and finite")
    exact_steps = args.delay_ms * args.control_hz / 1000.0
    if not np.isclose(exact_steps, round(exact_steps), rtol=0, atol=1e-8):
        raise ValueError("delay_ms must map to an integer simulator control step")


def make_conditions(args: Args) -> list[Condition]:
    validate_args(args)
    delay_steps = int(round(args.delay_ms * args.control_hz / 1000.0))
    return [
        Condition("clean", 0, 0),
        Condition("target_pose_stale", args.delay_ms, delay_steps),
        Condition("full_delay", args.delay_ms, delay_steps),
    ]


def _source_for_target(target: str) -> str:
    return {
        "observation/image": "agentview_image",
        "observation/wrist_image": "robot0_eye_in_hand_image",
    }[target]


def _camera_name(source: str) -> str:
    return source[:-len("_image")] if source.endswith("_image") else source


def _orient_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    from openpi_client import image_tools  # noqa: PLC0415
    return np.ascontiguousarray(image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size)))


def _object_pose(env: Any, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()
    # MuJoCo body_xquat is wxyz; use xyzw internally to match LIBERO EEF obs.
    wxyz = np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64).copy()
    if wxyz.shape != (4,) or not np.all(np.isfinite(wxyz)):
        raise ValueError("Invalid target body quaternion")
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
    return position, _quat_normalize(xyzw)


def _eef_pose(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    quaternion = _quat_normalize(np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy())
    return position, quaternion


def _quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid quaternion")
    return quaternion / norm


def stale_object_world_pose(current: Snapshot, delayed: Snapshot, *,
                            mode: str = "world_delay",
                            table_pose: Optional[tuple[np.ndarray, np.ndarray]] = None,
                            table_z_tolerance_m: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    """Return a delayed target pose in the fixed world frame.

    The previous implementation transformed the delayed object-to-EEF pose
    onto the current EEF. That makes a tabletop object follow the robot and
    can lift it into the air. The world-delay mode uses the target's old world
    pose directly. The default table-stale mode additionally freezes a target
    once its delayed snapshot indicates that it has been carried away from
    the tabletop. Current EEF pose is never used to construct the target.
    """
    del current  # Explicit: current EEF pose must not affect the target pose.
    if mode not in ("table_stale", "world_delay"):
        raise ValueError(f"Unknown target pose mode {mode!r}")
    if mode == "table_stale" and table_pose is not None:
        table_position, table_quaternion = table_pose
        # Once the target has been lifted, its delayed world pose is a pose
        # along the historical grasp trajectory.  That is a valid full-world
        # delay, but it makes this intervention look as if the object is glued
        # to the gripper.  The table-stale intervention intentionally keeps a
        # detached target at its last tabletop pose so it isolates the visual
        # target/robot mismatch.
        lifted = delayed.object_pos[2] > float(table_position[2]) + table_z_tolerance_m
        # Some object XMLs place the visual mesh below/above the free-joint
        # origin, so z alone is not a sufficient held-object test. A target
        # that has moved close to the delayed EEF is also treated as carried.
        carried = (
            np.linalg.norm(delayed.object_pos - delayed.eef_pos) < 0.12
            and np.linalg.norm(delayed.object_pos - np.asarray(table_position)) > 0.04
        )
        if lifted or carried:
            return np.asarray(table_position, dtype=np.float64).copy(), _quat_normalize(
                np.asarray(table_quaternion, dtype=np.float64).copy()
            )
    return delayed.object_pos.copy(), _quat_normalize(delayed.object_quat_xyzw.copy())


def _body_name(model: Any, body_id: int) -> str:
    try:
        return str(model.body_id2name(int(body_id)))
    except Exception:
        return f"body_{int(body_id)}"


def _free_joint_for_body(env: Any, body_id: int) -> tuple[int, int]:
    """Return the root free joint for exactly this body.

    LIBERO's object mapping points at the root body. Silently selecting a
    different free joint is dangerous: the rendered object can then appear to
    move with another body (often the gripper). Require an exact body match
    and include a useful model dump if the mapping is not as expected.
    """
    model = env.sim.model
    # Prefer MuJoCo's per-body joint range. This avoids accidentally finding
    # a free joint from another body when a wrapper exposes a reordered joint
    # table. Fall back to the global arrays for older mujoco_py versions.
    body_jntadr = getattr(model, "body_jntadr", None)
    body_jntnum = getattr(model, "body_jntnum", None)
    if body_jntadr is not None and body_jntnum is not None:
        first = int(np.asarray(body_jntadr)[int(body_id)])
        count = int(np.asarray(body_jntnum)[int(body_id)])
        candidate_ids = np.arange(first, first + count, dtype=np.int64)
    else:
        candidate_ids = np.arange(len(np.asarray(model.jnt_bodyid)), dtype=np.int64)
    body_ids = np.asarray(model.jnt_bodyid)
    joint_types = np.asarray(model.jnt_type)
    joint_qpos = np.asarray(model.jnt_qposadr)
    # MuJoCo's mjJNT_FREE enum is 0 for both mujoco_py and mujoco bindings.
    matches = candidate_ids[(body_ids[candidate_ids] == int(body_id)) & (joint_types[candidate_ids] == 0)]
    if len(matches) == 0:
        details = []
        for joint_id, (owner, jtype, qadr) in enumerate(zip(body_ids, joint_types, joint_qpos)):
            details.append({
                "joint": int(joint_id), "body": int(owner),
                "body_name": _body_name(model, owner), "type": int(jtype),
                "qposadr": int(qadr),
            })
        raise RuntimeError(
            f"Target body {body_id} ({_body_name(model, body_id)}) has no free joint; "
            f"joints={details}"
        )
    if len(matches) > 1:
        logging.warning("Target body %d has multiple free joints %s; using first", body_id, matches.tolist())
    joint = int(matches[0])
    return joint, int(joint_qpos[joint])


def _body_geom_ids(model: Any, body_id: int) -> list[int]:
    geom_bodyid = np.asarray(getattr(model, "geom_bodyid", []))
    return [int(index) for index, owner in enumerate(geom_bodyid) if int(owner) == int(body_id)]


def _assert_body_pose(sim: Any, body_id: int, requested: np.ndarray, *, label: str,
                      atol: float = 1e-5) -> None:
    actual = np.asarray(sim.data.body_xpos[int(body_id)], dtype=np.float64)
    error = float(np.max(np.abs(actual - requested)))
    if not np.all(np.isfinite(actual)) or error > atol:
        model = sim.model
        raise RuntimeError(
            f"{label}: MuJoCo body pose mismatch for body {body_id} "
            f"({_body_name(model, body_id)}): requested={requested.tolist()}, "
            f"actual={actual.tolist()}, max_abs={error}, atol={atol}, "
            f"geom_ids={_body_geom_ids(model, body_id)}"
        )


def render_with_object_pose(env: Any, source: str, body_id: int, position: np.ndarray,
                            quaternion_xyzw: np.ndarray, render_size: int) -> np.ndarray:
    """Render one camera after a temporary target pose, restoring all qpos."""
    sim = env.sim
    _, qpos_address = _free_joint_for_body(env, body_id)
    if not hasattr(sim, "get_state") or not hasattr(sim, "set_state_from_flattened"):
        raise RuntimeError("The active simulator cannot save/restore a temporary render state")
    # Synchronize derived kinematics before taking the reference.  The body
    # pose cache can otherwise be one forward pass behind the qpos state after
    # a contact/constraint update.
    sim.forward()
    saved_state = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    saved_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    saved_body_position = np.asarray(sim.data.body_xpos[int(body_id)], dtype=np.float64).copy()
    position = np.asarray(position, dtype=np.float64)
    quaternion_xyzw = _quat_normalize(quaternion_xyzw)
    quaternion_wxyz = np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
    try:
        # A MuJoCo free-joint qpos is [world xyz, quaternion wxyz]. Verify
        # body_xpos immediately after forward so a wrong body/joint mapping
        # cannot silently produce a misleading video.
        sim.data.qpos[qpos_address:qpos_address + 3] = position
        sim.data.qpos[qpos_address + 3:qpos_address + 7] = quaternion_wxyz
        sim.forward()
        _assert_body_pose(sim, body_id, position, label="temporary render")
        try:
            image = sim.render(camera_name=_camera_name(source), width=render_size, height=render_size)
        except TypeError:
            image = sim.render(camera_name=_camera_name(source), width=render_size, height=render_size, depth=False)
        image = _orient_rgb(image)
    finally:
        sim.set_state_from_flattened(saved_state)
        # Only qpos is intentionally changed above.  Assigning the saved qpos
        # once more avoids depending on wrapper-specific flatten/unflatten
        # handling and gives us an exact state check below.
        sim.data.qpos[:] = saved_qpos
        sim.forward()
        qpos_error = float(np.max(np.abs(np.asarray(sim.data.qpos) - saved_qpos)))
        if qpos_error > 1e-12:
            raise RuntimeError(
                f"render restore: qpos was not restored exactly; max_abs={qpos_error}"
            )
        # body_xpos is a derived floating-point cache.  A sub-millimetre
        # difference after forward is normal in contact-heavy scenes; qpos is
        # the authoritative restoration check above.
        _assert_body_pose(
            sim, body_id, saved_body_position, label="render restore", atol=1e-3
        )
    return image


def _snapshot(env: Any, observation: dict[str, Any], body_id: int) -> Snapshot:
    images = {source: _orient_rgb(observation[source]) for source in _shift.CAMERA_SOURCES}
    eef_pos, eef_quat = _eef_pose(observation)
    object_pos, object_quat = _object_pose(env, body_id)
    return Snapshot(images, eef_pos, eef_quat, object_pos, object_quat)


def _create_segmentation_env(task: Any):
    from libero.libero import get_libero_path  # noqa: PLC0415
    from libero.libero.envs import SegmentationRenderEnv  # noqa: PLC0415

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return SegmentationRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=_shift.LIBERO_ENV_RESOLUTION,
        camera_widths=_shift.LIBERO_ENV_RESOLUTION,
        camera_segmentations="instance",
    )


def _target_object(env: Any, requested: str) -> str:
    inner = getattr(env, "env", env)
    names = list(getattr(env, "obj_of_interest", []))
    body_ids = getattr(inner, "obj_body_id", {})
    target = requested.strip() if requested and requested.strip() else (names[0] if names else "")
    if target not in body_ids:
        raise ValueError(f"Target object {target!r} is not available; choose one of {sorted(body_ids)}")
    return target


def _policy_observation(*, env: Any, history: collections.deque, states: collections.deque,
                        condition: Condition, target_source: str, body_id: int, prompt: str,
                        resize_size: int, render_size: int, target_pose_mode: str,
                        table_pose: Optional[tuple[np.ndarray, np.ndarray]],
                        table_z_tolerance_m: float) -> tuple[dict[str, Any], dict[str, Any]]:
    current = history[-1]
    delayed = history[-1 - condition.delay_steps]
    selected_image = current.images[target_source]
    stale_pose = None
    if condition.name == "target_pose_stale":
        stale_pose = stale_object_world_pose(
            current, delayed, mode=target_pose_mode, table_pose=table_pose,
            table_z_tolerance_m=table_z_tolerance_m,
        )
        selected_image = render_with_object_pose(
            env, target_source, body_id, stale_pose[0], stale_pose[1], render_size
        )
    elif condition.name == "full_delay":
        selected_image = delayed.images[target_source]
    images = dict(current.images)
    images[target_source] = selected_image
    policy = {
        "observation/image": _resize_image(images["agentview_image"], resize_size),
        "observation/wrist_image": _resize_image(images["robot0_eye_in_hand_image"], resize_size),
        "observation/state": np.array(states[-1], copy=True),
        "prompt": prompt,
    }
    detail = {
        "current_images": {source: _resize_image(image, resize_size) for source, image in current.images.items()},
        "policy_images": {source: _resize_image(image, resize_size) for source, image in images.items()},
        "stale_object_pose": stale_pose,
        "current_object_pos": current.object_pos.copy(),
        "delayed_object_pos": delayed.object_pos.copy(),
        "current_eef_pos": current.eef_pos.copy(),
        "delayed_eef_pos": delayed.eef_pos.copy(),
    }
    return policy, detail


def _run_episode(*, args: Args, client: Any, task: Any, initial_state: np.ndarray, prompt: str,
                 condition: Condition, repeat_index: int, noise_shape: tuple[int, int],
                 warmup_steps: int, history_steps: int, max_steps: int,
                 target_name: str, output_dir: pathlib.Path) -> dict[str, Any]:
    env = None
    snapshots = collections.deque(maxlen=history_steps + 1)
    states = collections.deque(maxlen=history_steps + 1)
    plan = collections.deque()
    samples, frames, actions, chunks, noises, queries, source_steps, policy_states = [], [], [], [], [], [], [], []
    pose_diagnostics = []
    outcome, reason, error_text = "failure", "max_steps", None
    actual_hz, objects, missing_objects, start_sim = None, [], [], np.empty(0)
    resolved_target = target_name

    try:
        env = _create_segmentation_env(task)
        env.seed(args.seed + repeat_index)
        actual_hz = _trajectory.simulator_hz(env)
        if not np.isclose(actual_hz, args.control_hz, rtol=0, atol=1e-8):
            raise ValueError(f"control_hz={args.control_hz} differs from simulator control_freq={actual_hz}")
        env.reset()
        resolved_target = _target_object(env, target_name)
        body_id = int(getattr(getattr(env, "env", env), "obj_body_id")[resolved_target])
        joint_id, qpos_address = _free_joint_for_body(env, body_id)
        logging.info(
            "Target render mapping: name=%s body_id=%d body_name=%s free_joint=%d qposadr=%d geom_ids=%s",
            resolved_target, body_id, _body_name(env.sim.model, body_id), joint_id, qpos_address,
            _body_geom_ids(env.sim.model, body_id),
        )
        observation = env.set_init_state(np.array(initial_state, copy=True))
        snapshots.append(_snapshot(env, observation, body_id))
        states.append(_shift._build_state(observation))
        for _ in range(warmup_steps):
            observation, _, done, _ = env.step(_shift.LIBERO_DUMMY_ACTION)
            snapshots.append(_snapshot(env, observation, body_id))
            states.append(_shift._build_state(observation))
            if done or env.check_success():
                raise RuntimeError("Task ended during warmup; no valid trajectory comparison")
        # The post-warmup target is the tabletop reference.  In the default
        # table_stale mode it is used only after a delayed snapshot shows the
        # object has been lifted; world_delay keeps the exact historical pose.
        table_pose = (
            snapshots[-1].object_pos.copy(), snapshots[-1].object_quat_xyzw.copy()
        )
        objects, missing_objects = _trajectory.object_references(env, args.object_names)
        start_sim = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        samples.append(_trajectory.state_sample(env, observation, objects))
        target_source = _source_for_target(args.delay_target)

        for step in range(max_steps):
            policy_obs, detail = _policy_observation(
                env=env, history=snapshots, states=states, condition=condition,
                target_source=target_source, body_id=body_id, prompt=prompt,
                resize_size=args.resize_size, render_size=args.render_size,
                target_pose_mode=args.target_pose_mode, table_pose=table_pose,
                table_z_tolerance_m=args.table_z_tolerance_m,
            )
            if args.debug_render_pose and condition.name == "target_pose_stale" and detail["stale_object_pose"] is not None:
                logging.info(
                    "target pose query step=%d mode=%s current=%s delayed=%s stale=%s eef=%s delayed_eef=%s",
                    step, args.target_pose_mode,
                    np.round(detail["current_object_pos"], 4).tolist(),
                    np.round(detail["delayed_object_pos"], 4).tolist(),
                    np.round(detail["stale_object_pose"][0], 4).tolist(),
                    np.round(detail["current_eef_pos"], 4).tolist(),
                    np.round(detail["delayed_eef_pos"], 4).tolist(),
                )
            query = not plan
            if args.save_video:
                frames.append(_shift._make_video_frame(
                    current_images={key: detail["current_images"][source]
                                    for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)},
                    policy_images={key: detail["policy_images"][source]
                                  for key, source in zip(_shift.CAMERA_KEYS, _shift.CAMERA_SOURCES)},
                    current_state=states[-1], policy_state=policy_obs["observation/state"],
                    delay_target=f"{args.delay_target}:{condition.name}",
                    effective_delay_ms=condition.delay_steps / actual_hz * 1000,
                    step=step, policy_query=query,
                ))
            if query:
                noise = paired_noise(args.seed, repeat_index, step, noise_shape)
                response = client.infer_with_noise(policy_obs, noise)
                chunk = np.asarray(response["actions"], dtype=np.float32)
                if chunk.shape != (noise_shape[0], 7) or not np.all(np.isfinite(chunk)):
                    raise ValueError(f"Expected finite actions {(noise_shape[0], 7)}, got {chunk.shape}")
                queries.append(step)
                selected_column = _shift.CAMERA_KEYS.index(args.delay_target)
                source_steps.append([
                    step - condition.delay_steps if index == selected_column else step
                    for index in range(3)
                ])
                chunks.append(chunk.copy())
                noises.append(noise)
                policy_states.append(np.array(policy_obs["observation/state"], copy=True))
                if detail["stale_object_pose"] is None:
                    stale_pose = (np.full(3, np.nan), np.full(4, np.nan))
                else:
                    stale_pose = detail["stale_object_pose"]
                pose_diagnostics.append(np.concatenate([
                    detail["current_object_pos"], detail["delayed_object_pos"],
                    detail["current_eef_pos"], detail["delayed_eef_pos"],
                    stale_pose[0], stale_pose[1],
                ]).astype(np.float32))
                plan.extend(chunk[:args.replan_steps])
            action = plan.popleft()
            observation, _, done, _ = env.step(action.tolist())
            actions.append(action.copy())
            samples.append(_trajectory.state_sample(env, observation, objects))
            snapshots.append(_snapshot(env, observation, body_id))
            states.append(_shift._build_state(observation))
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
        logging.exception("Target-pose delay run failed: %s repeat=%d", condition.name, repeat_index)
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
        pose_diagnostics=np.asarray(pose_diagnostics, dtype=np.float32).reshape(-1, 19),
        object_names=np.asarray([name for name, _ in objects], dtype=str),
        start_sim_state=start_sim,
    )
    video_path = output_dir / f"{slug}_{outcome}.mp4"
    video_ok, video_error = _shift._save_video(video_path, frames, args.video_fps) if frames else (False, None)
    return {
        "condition": condition.name, "repeat_index": repeat_index, "seed": args.seed + repeat_index,
        "task_suite_name": args.task_suite_name, "task_id": args.task_id, "initial_state_id": args.initial_state_id,
        "initial_state_sha256": hashlib.sha256(np.asarray(initial_state, dtype=np.float64).tobytes()).hexdigest(),
        "prompt": prompt, "delay_target": args.delay_target, "target_object": resolved_target,
        "target_pose_mode": args.target_pose_mode,
        "table_z_tolerance_m": args.table_z_tolerance_m,
        "delay_ms_requested": condition.delay_ms, "delay_steps": condition.delay_steps,
        "delay_ms_effective": condition.delay_steps / args.control_hz * 1000 if actual_hz else None,
        "control_hz": actual_hz, "replan_steps": args.replan_steps, "warmup_steps": warmup_steps,
        "outcome": outcome, "termination_reason": reason, "success": outcome == "success",
        "steps": len(actions), "n_samples": len(samples), "n_queries": len(queries),
        "trace_path": trace_path.name, "video_path": video_path.name if video_ok else None,
        "video_fps": args.video_fps, "video_frames": len(frames), "video_error": video_error,
        "object_names": [name for name, _ in objects], "unavailable_objects": missing_objects, "error": error_text,
    }


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
            "condition": condition, "runs": len(group), "expected_runs": repeats,
            "successes": counts["success"], "failures": counts["failure"],
            "errors": counts["error"], "interrupted": counts["interrupted"], "completed": completed,
            "success_rate": counts["success"] / completed if completed else None,
            "failure_rate": counts["failure"] / completed if completed else None,
        })
    return result


def save_summary(directory: pathlib.Path, args: Args, metadata: dict[str, Any],
                 conditions: list[Condition], records: list[dict[str, Any]]) -> None:
    report = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "config": dataclasses.asdict(args),
        "server_metadata": metadata, "conditions": [dataclasses.asdict(c) for c in conditions],
        "protocol": {
            "clean": "current selected-camera render",
            "target_pose_stale": "current-scene render with the target at its delayed world-frame pose; in table_stale mode, freeze at the last tabletop pose after lift",
            "full_delay": "complete selected-camera frame from delay_steps earlier",
            "selected_camera": args.delay_target,
            "physical_state": "temporary render pose is restored before env.step; real target remains current",
            "target_pose_columns": "position and quaternion are world-frame; quaternion xyzw",
            "target_pose_mode": args.target_pose_mode,
            "table_stale_rule": "if delayed target z exceeds the post-warmup tabletop z by table_z_tolerance_m, render the fixed tabletop reference pose; world_delay preserves the exact delayed pose",
            "pairing": "same task, initial state, repeat seed, warmup and explicit policy noise",
            "sampling": "all control steps including initial and terminal state; samples=actions+1",
            "success": "LIBERO environment task predicate",
        },
        "aggregate": _aggregate(records, args.repeats_per_condition), "records": records,
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "summary.json")
    if records:
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
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
        max_delay = max(c.delay_steps for c in conditions)
        warmup_steps = max(args.num_steps_wait, max_delay)
        max_steps = args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]
        records: list[dict[str, Any]] = []
        save_summary(directory, args, metadata, conditions, records)
        logging.info("Running %d target-pose rollouts (%d conditions x %d repeats)",
                     len(conditions) * args.repeats_per_condition, len(conditions), args.repeats_per_condition)
        logging.info("Selected camera: %s; target object: %s", args.delay_target, args.target_object or "first obj_of_interest")
        for repeat in range(args.repeats_per_condition):
            for condition in conditions:
                record = _run_episode(
                    args=args, client=client, task=task, initial_state=initial_states[args.initial_state_id],
                    prompt=prompt, condition=condition, repeat_index=repeat, noise_shape=noise_shape,
                    warmup_steps=warmup_steps, history_steps=max_delay, max_steps=max_steps,
                    target_name=args.target_object, output_dir=directory,
                )
                records.append(record)
                save_summary(directory, args, metadata, conditions, records)
                logging.info("[%d/%d] condition=%s repeat=%d outcome=%s steps=%d",
                             len(records), len(conditions) * args.repeats_per_condition,
                             condition.name, repeat, record["outcome"], record["steps"])
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
        from examples.libero import analyze_target_pose_delay as analysis
    except ImportError:
        import analyze_target_pose_delay as analysis
    summary_path = _run_rollouts(args)
    logging.info("Rollouts saved to %s; generating target-pose trajectory figures", summary_path)
    analysis_path = analysis.analyze(summary_path, threshold_cm=args.threshold_cm,
                                     consecutive=args.consecutive, pair_limit=args.pair_limit,
                                     interactive=args.interactive)
    logging.info("Analysis complete: %s", analysis_path.parent)
    return summary_path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        if field.name == "save_video":
            parser.add_argument("--no-video", "--no_video", dest=field.name, action="store_false", default=True)
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
