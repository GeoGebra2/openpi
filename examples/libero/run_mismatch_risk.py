"""Bounded LIBERO experiment for image/self-state temporal mismatch risk.

The command runs clean calibration, a small automatic time-window search,
recovery-gap probes, and held-out validation. Only the selected camera is
delayed; the policy always receives the current self state. Every rollout
stores the real self state at the image's source time for metric auditing.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import pathlib
from typing import Optional

import numpy as np

try:
    from examples.libero import analyze_mismatch_risk as _analysis
    from examples.libero import critical_window_workers as _workers
    from examples.libero import mismatch_risk_metrics as _metrics
    from examples.libero import run_critical_window_search as _critical
    from examples.libero import run_trajectory_delay as _trajectory
except ImportError:
    import analyze_mismatch_risk as _analysis
    import critical_window_workers as _workers
    import mismatch_risk_metrics as _metrics
    import run_critical_window_search as _critical
    import run_trajectory_delay as _trajectory


EXPERIMENT = "libero_image_state_mismatch_risk"
SCHEMA_VERSION = 1
VALIDATION_OFFSET = 1_000_000


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    servers: str = ""
    workers: int = 1
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    initial_state_id: int = 0
    initial_state_ids: str = ""
    prompt: Optional[str] = None
    policy_label: str = "pi0_libero"
    delay_target: str = "observation/wrist_image"
    delay_levels_ms: str = "200,400,600"
    max_coverage: float = 0.5
    grid_bins: int = 10
    discovery_repeats: int = 6
    validation_repeats: int = 50
    finalists: int = 3
    matched_controls: int = 1
    control_hz: float = 20.0
    replan_steps: int = 1
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    seed: int = 7
    save_video: bool = False
    video_fps: int = 10
    object_names: str = ""
    output_dir: str = "rollout/libero_image_state_mismatch_risk"
    resume: Optional[str] = None
    dry_run: bool = False
    analyze_only: Optional[str] = None


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def state_ids(args):
    values = [int(item.strip()) for item in args.initial_state_ids.split(",") if item.strip()]
    values = values or [args.initial_state_id]
    if min(values) < 0 or len(values) != len(set(values)):
        raise ValueError("initial state IDs must be distinct nonnegative integers")
    return values


def delay_levels(args):
    values = sorted({int(value.strip()) for value in args.delay_levels_ms.split(",") if value.strip()})
    if not values or any(value < 1 or value > 600 for value in values):
        raise ValueError("delay_levels_ms must be positive integers no greater than 600")
    for value in values:
        exact = value * args.control_hz / 1000
        if not np.isclose(exact, round(exact), rtol=0, atol=1e-8):
            raise ValueError(f"delay {value} ms does not align to control_hz={args.control_hz}")
    return values


def validate_args(args, minimum_validation_repeats=3):
    if args.delay_target not in _critical._shift.CAMERA_KEYS:
        raise ValueError("delay_target must be exactly one supported camera")
    if args.workers < 1 or args.grid_bins != 10:
        raise ValueError("workers must be positive; this bounded pilot uses grid_bins=10")
    if args.discovery_repeats < 3 or args.validation_repeats < minimum_validation_repeats or not 1 <= args.finalists <= 3:
        raise ValueError(f"Need at least three discovery repeats, {minimum_validation_repeats} validation repeats, and 1..3 finalists")
    if args.matched_controls not in (0, 1):
        raise ValueError("matched_controls is bounded to 0 or 1 for this pilot")
    if args.replan_steps < 1 or args.resize_size < 1 or args.num_steps_wait < 0:
        raise ValueError("Invalid replanning, image size or warmup")
    if not 0.1 <= args.max_coverage <= 0.5:
        raise ValueError("This pilot requires max_coverage in [0.1, 0.5]")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0 or args.video_fps < 1:
        raise ValueError("control_hz and video_fps must be positive")
    if args.task_suite_name not in _critical._shift.MAX_STEPS_BY_SUITE:
        raise ValueError("Unsupported LIBERO suite")
    if max(args.discovery_repeats, args.validation_repeats) >= VALIDATION_OFFSET:
        raise ValueError("Discovery/validation repeat ranges would overlap")
    delay_levels(args)
    state_ids(args)
    endpoints = _workers.parse_servers(args.servers, args.host, args.port)
    if args.workers < len(endpoints):
        raise ValueError("workers must be at least the number of policy servers")
    return int(round(max(delay_levels(args)) * args.control_hz / 1000)), args.max_steps or _critical._shift.MAX_STEPS_BY_SUITE[args.task_suite_name]


def candidate_id(delay_ms, windows):
    digest = hashlib.sha256(json.dumps([delay_ms, windows], separators=(",", ":")).encode()).hexdigest()[:12]
    return f"d{delay_ms:03d}_w{digest}"


def make_candidate(delay_ms, windows, grid, origin):
    windows = _critical.merge_windows(windows)
    if not windows or windows[0][0] < 0 or windows[-1][1] > grid:
        return None
    coverage = _critical.window_length(windows) / grid
    if coverage > 0.5 + 1e-9:
        return None
    return {"id": candidate_id(delay_ms, windows), "kind": "windows", "windows": windows,
            "coverage": coverage, "delay_ms": delay_ms, "origins": [origin]}


def unique_candidates(candidates):
    found = {}
    for candidate in candidates:
        if candidate is None:
            continue
        old = found.get(candidate["id"])
        if old is None:
            found[candidate["id"]] = candidate
        else:
            old["origins"] = sorted(set(old["origins"] + candidate["origins"]))
    return list(found.values())


def search_candidates(args):
    """Small deterministic coarse/fine grid plus separated recovery probes."""
    grid = args.grid_bins
    values = []
    for delay in delay_levels(args):
        for length in (1, 3, 5):
            if length / grid > args.max_coverage + 1e-9:
                continue
            starts = range(0, grid - length + 1, 2)
            values.extend(make_candidate(delay, [[start, start + length]], grid, "fine_single")
                          for start in starts)
    return unique_candidates(values)


def refine_candidates(args, ranking):
    result = []
    singles = [row for row in ranking if len(row["spec"]["windows"]) == 1]
    for row in singles[: min(3, len(singles))]:
        spec = row["spec"]
        delay = spec["delay_ms"]
        for index, (start, end) in enumerate(spec["windows"]):
            for delta in (-1, 1):
                shifted = [window[:] for window in spec["windows"]]
                shifted[index] = [start + delta, end + delta]
                result.append(make_candidate(delay, shifted, args.grid_bins, "local_refinement"))
            if end - start > 1:
                shortened = [window[:] for window in spec["windows"]]
                shortened[index] = [start, end - 1]
                result.append(make_candidate(delay, shortened, args.grid_bins, "local_refinement"))
    return [spec for spec in unique_candidates(result) if spec["coverage"] <= args.max_coverage][:6]


def recovery_plan(args, ranking):
    """Fix the second window, move the first, and include both single controls."""
    eligible = [row for row in ranking if len(row["spec"]["windows"]) == 1
                and row["spec"]["windows"][0][1] >= 6]
    if not eligible or args.max_coverage < 0.2:
        return {"groups": [], "candidates": [], "reason": "No eligible late window or coverage below 20%"}
    anchor = eligible[0]["spec"]
    end = min(10, anchor["windows"][0][1])
    second = [end - 1, end]
    specs, groups = [], []
    for gap in (1, 2, 3):
        first = [second[0] - gap - 1, second[0] - gap]
        if first[0] < 0:
            continue
        a = make_candidate(anchor["delay_ms"], [first], 10, "recovery_first")
        b = make_candidate(anchor["delay_ms"], [second], 10, "recovery_second")
        both = make_candidate(anchor["delay_ms"], [first, second], 10, "recovery_both")
        specs.extend([a, b, both])
        groups.append({"gap_frac": gap / 10, "first": a["id"], "second": b["id"], "both": both["id"]})
    continuous = make_candidate(anchor["delay_ms"], [[end - 2, end]], 10, "recovery_contiguous")
    specs.append(continuous)
    return {"anchor_id": anchor["id"], "fixed_second_window": second,
            "groups": groups, "continuous_id": continuous["id"], "candidates": unique_candidates(specs)}


def mapped_windows(args, spec, reference_steps):
    q, grid = args.replan_steps, args.grid_bins
    usable = reference_steps // q
    windows = []
    for start, end in spec["windows"]:
        left = (start * usable // grid) * q
        right = left + ((end - start) * usable // grid) * q
        if right > left:
            windows.append([left, right])
    windows = _critical.merge_windows(windows)
    budget = math.floor(min(args.max_coverage, spec["coverage"]) * reference_steps / q + 1e-9) * q
    while _critical.window_length(windows) > budget and windows:
        start, end = windows[-1]
        if end - start <= q:
            windows.pop()
        else:
            windows[-1][1] -= q
    return windows


def calibration_from_records(records):
    clean = [row for row in records if row["candidate_id"] == "clean" and row["split"] == "discovery"
             and row["outcome"] in ("success", "failure")]
    successful = np.asarray([row["steps"] for row in clean if row["success"]], dtype=float)
    if len(successful) < 3:
        raise RuntimeError("Need at least three successful discovery clean episodes for robust timing calibration; "
                           "check the clean task/policy or start a run with more --discovery-repeats")
    median = float(np.median(successful))
    mad = float(np.median(np.abs(successful - median)))
    spread = max(3 * 1.4826 * mad, 0.5 * median, 2.0)
    lower, upper = max(1.0, median - spread), median + spread
    typical = successful[(successful >= lower) & (successful <= upper)]
    median = float(np.median(typical))
    calibration = {"successful_clean_n": int(len(successful)), "median_steps": median,
                   "successful_mad_steps": mad, "typical_success_n": int(len(typical)),
                   "valid_lower_steps": lower, "valid_upper_steps": upper,
                   "duration_rule": "own successful duration within median +/- max(3*1.4826*MAD,0.5*median,2); "
                                    "failed/atypical clean duration replaced, outcome retained",
                   "per_state": {}}
    for initial_id in sorted({row["initial_state_id"] for row in clean}):
        group = [row for row in clean if row["initial_state_id"] == initial_id]
        if sum(row["success"] for row in group) >= 3 and len(group) != len(clean):
            calibration["per_state"][str(initial_id)] = calibration_from_records(group)
    return calibration


def calibration_reference(record, calibration):
    calibration = calibration.get("per_state", {}).get(str(record.get("initial_state_id")), calibration)
    steps = float(record.get("steps", 0))
    valid = (record.get("outcome") == "success" and
             calibration["valid_lower_steps"] <= steps <= calibration["valid_upper_steps"] and steps > 0)
    # A max-step clean failure is commonly an unfinished rollout; it must not
    # stretch normalized windows. Its outcome remains in paired statistics.
    if valid:
        return int(round(steps)), "paired_clean_duration"
    return int(round(calibration["median_steps"])), "frozen_success_clean_median"


def audit_trace(args, record, trace, windows, reference=None):
    query = np.arange(0, record["steps"], args.replan_steps, dtype=np.int32)
    if not np.array_equal(query, trace["query_steps"]):
        raise ValueError("Unexpected policy query times")
    delay = int(round(record["effective_delay_ms"] * args.control_hz / 1000))
    expected = np.array([delay if any(a <= t < b for a, b in windows) else 0 for t in query])
    if not np.array_equal(expected, trace["query_delay_steps"]):
        raise ValueError("Image age does not match the frozen windows")
    sources = np.repeat(query[:, None], 3, axis=1)
    sources[:, _critical._shift.CAMERA_KEYS.index(args.delay_target)] -= expected
    if not np.array_equal(sources, trace["input_source_steps"]):
        raise ValueError("Image/state source step mismatch")
    current = trace["query_current_states"]
    if not np.allclose(current, trace["policy_states"], rtol=0, atol=1e-6):
        raise ValueError("Policy did not receive current self state")
    if not np.allclose(current[:, :3], trace["eef_pos"][query], rtol=0, atol=1e-6):
        raise ValueError("Recorded query state does not match physical state")
    source_steps = query - expected
    nonnegative = source_steps >= 0
    if not np.allclose(trace["query_image_source_states"][nonnegative, :3],
                       trace["eef_pos"][source_steps[nonnegative]], rtol=0, atol=1e-6):
        raise ValueError("Image-source state came from a different trajectory")
    if not np.allclose(trace["query_image_source_states"][nonnegative, 6:],
                       trace["gripper_qpos"][source_steps[nonnegative]], rtol=0, atol=1e-6):
        raise ValueError("Image-source gripper state does not match history")
    if np.any(_metrics.rotation_distance(trace["query_current_quats"], trace["eef_quat_xyzw"][query]) > 1e-6):
        raise ValueError("Query rotation does not match physical state")
    if np.any(_metrics.rotation_distance(trace["query_image_source_quats"][nonnegative],
                                         trace["eef_quat_xyzw"][source_steps[nonnegative]]) > 1e-6):
        raise ValueError("Image-source rotation does not match history")
    action_query = np.arange(record["steps"]) // args.replan_steps
    if (not np.array_equal(trace["action_query_steps"], query[action_query])
            or not np.array_equal(trace["action_delay_steps"], expected[action_query])):
        raise ValueError("An action plan crossed an intervention boundary")
    _metrics.components(trace)
    if reference is not None:
        if not np.allclose(trace["start_sim_state"], reference["start_sim_state"], rtol=0, atol=1e-9):
            raise ValueError("Paired post-warmup simulator states differ")
        _, a, b = np.intersect1d(query, reference["query_steps"], return_indices=True)
        if not np.array_equal(_critical.noise_hashes(trace)[a], _critical.noise_hashes(reference)[b]):
            raise ValueError("Paired explicit policy noise differs")


def compact_trace(path, trace):
    names = ("steps", "time_s", "eef_pos", "eef_quat_xyzw", "gripper_qpos", "executed_actions",
             "query_steps", "query_delay_steps", "action_query_steps", "action_delay_steps", "input_source_steps",
             "start_sim_state", "policy_states", "query_current_states", "query_image_source_states",
             "query_current_quats", "query_image_source_quats")
    arrays = {name: trace[name] for name in names}
    arrays["query_noise_hashes"] = _critical.noise_hashes(trace)
    temporary = pathlib.Path(path).with_suffix(".compact.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class MismatchSession:
    def __init__(self, args, directory, client, task, initial_states, metadata, episode_runner=None,
                 client_factory=None):
        self.args, self.directory, self.client, self.task = args, pathlib.Path(directory), client, task
        self.initial_states, self.metadata = initial_states, metadata
        self.max_delay_steps, self.max_steps = validate_args(args)
        self.server_endpoints = _workers.parse_servers(args.servers, args.host, args.port)
        self.noise_shape = _trajectory.noise_shape_from_metadata(metadata)
        if args.replan_steps > self.noise_shape[0] or not callable(getattr(client, "infer_with_noise", None)):
            raise ValueError("Explicit-noise client required; replan_steps cannot exceed action horizon")
        if max(state_ids(args)) >= len(initial_states):
            raise ValueError("initial state ID out of range")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.registry = {}
        self._pool = None
        self.episode_runner = episode_runner or _trajectory.run_episode
        self.client_factory = client_factory or _workers.create_client
        self.calibration = None
        manifest_path = self.directory / "manifest.json"
        scientific = dataclasses.asdict(args).copy()
        state_hashes = {str(index): hashlib.sha256(np.asarray(initial_states[index], dtype=np.float64).tobytes()).hexdigest()
                        for index in state_ids(args)}
        for key in ("resume", "dry_run", "analyze_only", "output_dir", "host", "port", "workers", "servers"):
            scientific.pop(key, None)
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (self.manifest.get("scientific_config") != scientific or self.manifest.get("experiment") != EXPERIMENT
                    or self.manifest.get("schema_version") != SCHEMA_VERSION
                    or self.manifest.get("initial_state_hashes") != state_hashes
                    or self.manifest.get("server_metadata") != json.loads(json.dumps(metadata))):
                raise ValueError("Resume configuration does not match the saved mismatch experiment")
            self.calibration = self.manifest.get("timing_calibration")
            self.registry = json.loads((self.directory / "candidates.json").read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("episodes/*/*/pair_*/record.json")):
                row = json.loads(path.read_text(encoding="utf-8"))
                self.records[(row["split"], row["candidate_id"], row["pair_index"])] = row
        else:
            self.manifest = {"experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION,
                             "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "config": dataclasses.asdict(args), "scientific_config": scientific,
                             "initial_state_hashes": state_hashes,
                             "server_metadata": metadata, "protocol": self.protocol(), "status": "created"}
        for key in ("host", "port", "workers", "servers"):
            self.manifest["config"][key] = getattr(args, key)
        implementation = {pathlib.Path(path).name: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                          for path in (__file__, _metrics.__file__, _trajectory.__file__, _workers.__file__,
                                       _analysis.__file__, _critical.__file__, _critical._shift.__file__)}
        self.manifest.setdefault("execution_history", []).append({
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "workers": args.workers,
            "servers": [_workers.server_address(*endpoint) for endpoint in self.server_endpoints],
            "implementation_sha256": implementation})
        self.register({"id": "clean", "kind": "clean", "windows": [], "coverage": 0.0, "delay_ms": 0,
                       "origins": ["control"]})

    def protocol(self):
        return {"outcome": "Final LIBERO success/failure; runtime errors separate.",
                "input": "Selected camera delayed in windows; self state is always current.",
                "mismatch": "Compare current self state with real self state at the delayed image source step.",
                "timing": "Use paired successful clean duration inside frozen MAD fence; replace failed/atypical durations "
                          "with frozen typical-success median, retaining every outcome. Sparse states use pooled calibration.",
                "search": "Bounded deterministic single-window and two-window recovery probes; validation plan locked before validation outcomes.",
                "scope": "Pilot evidence for the requested task/state/delay; not a universal safety threshold.",
                "delay_cap_ms": 600}

    def register(self, spec):
        old = self.registry.get(spec["id"])
        if old is not None:
            if old.get("windows") != spec.get("windows") or old.get("delay_ms") != spec.get("delay_ms"):
                raise ValueError("Candidate identity collision")
            old["origins"] = sorted(set(old.get("origins", []) + spec.get("origins", [])))
        else:
            self.registry[spec["id"]] = spec

    def checkpoint(self, phase, status="running"):
        self.manifest.update(phase=phase, status=status)
        if self.calibration is not None:
            self.manifest["timing_calibration"] = self.calibration
        atomic_json(self.directory / "manifest.json", self.manifest)
        atomic_json(self.directory / "candidates.json", self.registry)

    def complete(self, key):
        row = self.records.get(key)
        if row and row.get("outcome") in ("success", "failure"):
            if not (self.directory / row["trace_path"]).exists():
                raise ValueError(f"Completed episode lost its trace: {row['trace_path']}")
            return row
        return None

    def prepare(self, split, candidate_id, pair_index):
        key = (split, candidate_id, pair_index)
        old = self.records.get(key)
        if old:
            directory = self.directory / "episodes" / split / candidate_id / f"pair_{pair_index:04d}"
            with (directory / "previous_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(old, allow_nan=False) + "\n")
        spec = self.registry[candidate_id]
        repeat = pair_index + (VALIDATION_OFFSET if split == "validation" else 0)
        initial_id = state_ids(self.args)[pair_index % len(state_ids(self.args))]
        reference = None if candidate_id == "clean" else self.records.get((split, "clean", pair_index))
        if candidate_id != "clean" and reference is None:
            raise RuntimeError(f"Missing paired clean record for {key}")
        if candidate_id != "clean" and reference.get("outcome") not in ("success", "failure"):
            raise RuntimeError(f"Paired clean record is not completed for {key}")
        if candidate_id == "clean":
            reference_steps, source = None, "clean"
            windows, delay_ms, delay_steps, target = [], 0, 0, _critical._shift.BASELINE_TARGET
        else:
            reference_steps, source = calibration_reference(reference, self.calibration)
            if spec["kind"] == "matched_random":
                original = mapped_windows(self.args, self.registry[spec["source_id"]], reference_steps)
                salt = int(hashlib.sha256(spec["id"].encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, repeat, salt]))
                q = self.args.replan_steps
                windows = [[a * q, b * q] for a, b in _critical.random_shape_windows(
                    [[a // q, b // q] for a, b in original], reference_steps // q, rng)] if original else []
            else:
                windows = mapped_windows(self.args, spec, reference_steps)
            delay_ms = spec["delay_ms"]
            delay_steps = int(round(delay_ms * self.args.control_hz / 1000))
            target = self.args.delay_target
        directory = self.directory / "episodes" / split / candidate_id / f"pair_{pair_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        prompt = self.args.prompt.strip() if self.args.prompt and self.args.prompt.strip() else str(self.task.language)
        episode_args = dataclasses.replace(self.args, initial_state_id=initial_id)
        setting = _trajectory.Setting(candidate_id, target, delay_ms, delay_steps)
        episode_kwargs = dict(args=episode_args, task=self.task, initial_state=self.initial_states[initial_id],
                              prompt=prompt, setting=setting, repeat_index=repeat, noise_shape=self.noise_shape,
                              warmup_steps=max(self.args.num_steps_wait, self.max_delay_steps),
                              history_steps=self.max_delay_steps, max_steps=self.max_steps, output_dir=directory,
                              capture_state_provenance=True)
        return {"key": key, "spec": spec, "reference": reference, "reference_steps": reference_steps,
                "reference_source": source, "windows": windows,
                "job": {"episode_kwargs": episode_kwargs, "windows": windows, "delay_steps": delay_steps}}

    def finish(self, prepared, record):
        key = prepared["key"]
        directory = self.directory / "episodes" / key[0] / key[1] / f"pair_{key[2]:04d}"
        record["trace_path"] = (directory / record["trace_path"]).relative_to(self.directory).as_posix()
        if record.get("video_path"):
            record["video_path"] = (directory / record["video_path"]).relative_to(self.directory).as_posix()
        reference_steps = prepared["reference_steps"] or record["steps"]
        record.update(split=key[0], candidate_id=key[1], pair_index=key[2],
                      delay_ms=prepared["spec"].get("delay_ms", 0),
                      windows_steps=prepared["windows"],
                      windows_seconds=[[a / self.args.control_hz, b / self.args.control_hz] for a, b in prepared["windows"]],
                      reference_steps=reference_steps, reference_duration_source=prepared["reference_source"],
                      planned_coverage=(sum(b - a for a, b in prepared["windows"]) / reference_steps
                                        if reference_steps else 0.0),
                      effective_delay_ms=(prepared["spec"].get("delay_ms", 0)))
        if record["outcome"] in ("success", "failure"):
            try:
                trace = _critical.read_trace(self.directory, record)
                reference = _critical.read_trace(self.directory, prepared["reference"]) if prepared["reference"] else None
                audit_trace(self.args, record, trace, prepared["windows"], reference)
                if "mismatch_calibration" in self.manifest:
                    values, _, _ = _metrics.exposure(trace, self.manifest["mismatch_calibration"], self.args.control_hz)
                    record.update(values)
                delayed = int(np.count_nonzero(trace["action_delay_steps"]))
                record.update(actual_coverage=delayed / max(1, record["steps"]),
                              actual_budget_valid=delayed <= self.args.max_coverage * record["steps"] + 1e-9)
                if record["planned_coverage"] > self.args.max_coverage + 1e-9:
                    raise ValueError("Mapped window exceeds coverage budget")
                compact_trace(self.directory / record["trace_path"], trace)
            except Exception as error:
                record.update(outcome="error", success=False, error=f"Provenance audit failed: {error!r}")
        atomic_json(directory / "record.json", record)
        self.records[key] = record
        atomic_json(self.directory / "progress.json", {"phase": self.manifest.get("phase"), "last_trial": list(key),
                                                          "completed_trials": sum(row["outcome"] in ("success", "failure") for row in self.records.values())})
        if record["outcome"] == "error":
            raise RuntimeError(f"Runtime error, not a task failure: {record.get('error')}; resume {self.directory}")
        if record["outcome"] == "interrupted":
            raise KeyboardInterrupt()
        logging.info("[%s] %s pair=%d %s actions=%d reference=%s", self.manifest.get("phase"), key[1], key[2],
                     record["outcome"], record["steps"], prepared["reference_source"])
        return record

    def run_stage(self, phase, candidates, split, repeats):
        self.checkpoint(phase)
        candidates = list(dict.fromkeys(candidates))
        salt = int(hashlib.sha256(phase.encode()).hexdigest()[:8], 16)
        keys = []
        for pair in range(repeats):
            generator = np.random.default_rng(np.random.SeedSequence([self.args.seed, salt, pair]))
            keys.extend((split, candidates[int(i)], pair) for i in generator.permutation(len(candidates)))
        prepared = {}
        pending = []
        for key in keys:
            if self.complete(key):
                continue
            value = self.prepare(*key)
            prepared[key] = value
            pending.append((key, value["job"]))
        if not pending:
            return
        if self.args.workers == 1:
            for key, job in pending:
                record = _workers.run_rollout_job(job, self.client, self.episode_runner)
                record["policy_server"] = _workers.server_address(*self.server_endpoints[0])
                self.finish(prepared[key], record)
            return
        if self._pool is None:
            self._pool = _workers.RolloutPool(self.args.workers, self.args.host, self.args.port, self.metadata,
                                            self.episode_runner, self.client_factory,
                                            server_endpoints=self.server_endpoints)
            self.manifest["execution_history"][-1]["worker_assignments"] = self._pool.assignments
            self.checkpoint(phase)
        try:
            for key, record in self._pool.iter_results(pending):
                self.finish(prepared[key], record)
        except BaseException:
            self._pool.close(terminate=True)
            self._pool = None
            raise

    def ranking(self):
        clean = [row for row in self.records.values() if row["split"] == "discovery" and row["candidate_id"] == "clean"
                 and row["outcome"] in ("success", "failure")]
        by_candidate = collections.defaultdict(list)
        for row in self.records.values():
            if row["split"] == "discovery" and row["candidate_id"] != "clean" and row["outcome"] in ("success", "failure"):
                by_candidate[row["candidate_id"]].append(row)
        baseline = {row["pair_index"]: row for row in clean}
        rows = []
        for key, group in by_candidate.items():
            pairs = [(baseline[row["pair_index"]], row) for row in group if row["pair_index"] in baseline]
            if len(pairs) < self.args.discovery_repeats:
                continue
            harmed = sum(a["success"] and not b["success"] for a, b in pairs)
            rescued = sum(not a["success"] and b["success"] for a, b in pairs)
            spec = self.registry[key]
            rows.append({"candidate_id": key, "spec": spec, "failure_increase": (harmed - rescued) / len(pairs),
                         "failure_rate": sum(not b["success"] for _, b in pairs) / len(pairs),
                         "harmed": harmed, "rescued": rescued, "n": len(pairs)})
        return sorted(rows, key=lambda row: (-row["failure_increase"], row["spec"]["coverage"], row["candidate_id"]))

    def validation_plan(self, ranking, recovery):
        path = self.directory / "validation_plan.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        selected, representatives = [], []
        for delay in delay_levels(self.args):
            group = [row for row in ranking if row["spec"]["delay_ms"] == delay
                     and len(row["spec"]["windows"]) == 1]
            if group:
                selected.append(group[0]["candidate_id"])
                representatives.append(group[len(group) // 2]["candidate_id"])
        selected.extend(row["candidate_id"] for row in ranking if len(row["spec"]["windows"]) == 1)
        finalists = list(dict.fromkeys(selected))[: self.args.finalists]
        extra_ids = list(dict.fromkeys(representatives + [spec["id"] for spec in recovery["candidates"]]))
        specs, controls, ablations, comparisons = [], [], [], []
        for key in finalists:
            spec = self.registry[key]
            specs.append(spec)
            comparisons.append({"id": f"{key}:clean", "candidate_id": key, "reference_id": "clean", "purpose": "vs_clean"})
            if self.args.matched_controls:
                control = {"id": f"random_{key}", "kind": "matched_random", "source_id": key,
                           "control_index": 0, "windows": [], "coverage": spec["coverage"], "delay_ms": spec["delay_ms"],
                           "origins": ["held_out_random_control"]}
                specs.append(control)
                controls.append({"candidate_id": key, "control_id": control["id"]})
                comparisons.append({"id": f"{key}:{control['id']}", "candidate_id": key,
                                    "reference_id": control["id"], "purpose": "vs_random"})
            for index in range(len(spec["windows"])):
                child = make_candidate(spec["delay_ms"], spec["windows"][:index] + spec["windows"][index + 1:],
                                       self.args.grid_bins, "held_out_ablation")
                if child:
                    specs.append(child)
                ablations.append({"candidate_id": key, "removed_window_index": index,
                                  "ablation_id": child["id"] if child else "clean"})
        for key in extra_ids:
            specs.append(self.registry[key])
        for spec in unique_candidates(specs):
            if spec["kind"] == "windows" and spec["id"] not in finalists:
                comparisons.append({"id": f"{spec['id']}:clean", "candidate_id": spec["id"],
                                    "reference_id": "clean", "purpose": "vs_clean"})
        plan = {"locked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "finalists": finalists,
                "candidates": unique_candidates(specs), "controls": controls, "ablations": ablations,
                "recovery": recovery, "representatives": representatives,
                "primary_comparisons": comparisons, "validation_repeat_start": VALIDATION_OFFSET,
                "selection": "Best and middle-ranked single window per delay, plus ALL recovery quartet controls; "
                             "frozen before held-out clean outcomes. Model fitted on discovery only."}
        atomic_json(path, plan)
        return plan

    def execute(self):
        try:
            self.run_stage("00_discovery_clean", ["clean"], "discovery", self.args.discovery_repeats)
            if self.calibration is None:
                self.calibration = calibration_from_records(list(self.records.values()))
            if "mismatch_calibration" not in self.manifest:
                normal_clean = [row for row in self.records.values() if row["candidate_id"] == "clean"
                                and row["split"] == "discovery" and row["outcome"] == "success"
                                and calibration_reference(row, self.calibration)[1] == "paired_clean_duration"]
                clean_traces = [_critical.read_trace(self.directory, row) for row in normal_clean]
                self.manifest["mismatch_calibration"] = _metrics.calibrate_scales(clean_traces)
            self.checkpoint("calibrated")
            base = search_candidates(self.args)
            for spec in base:
                self.register(spec)
            self.checkpoint("01_coarse")
            self.run_stage("01_coarse", [spec["id"] for spec in base], "discovery", self.args.discovery_repeats)
            refinement_path = self.directory / "stage_plans" / "refinement.json"
            if refinement_path.exists():
                refinement = json.loads(refinement_path.read_text(encoding="utf-8"))["candidates"]
            else:
                refinement = refine_candidates(self.args, self.ranking())
                atomic_json(refinement_path, {"candidates": refinement})
            for spec in refinement:
                self.register(spec)
            self.checkpoint("02_refinement")
            self.run_stage("02_refinement", [spec["id"] for spec in refinement], "discovery", self.args.discovery_repeats)
            recovery_path = self.directory / "stage_plans" / "recovery.json"
            if recovery_path.exists():
                recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            else:
                recovery = recovery_plan(self.args, self.ranking())
                atomic_json(recovery_path, recovery)
            for spec in recovery["candidates"]:
                self.register(spec)
            self.run_stage("03_recovery", [spec["id"] for spec in recovery["candidates"]], "discovery", self.args.discovery_repeats)
            reserve_path = self.directory / "stage_plans" / "event_reserve.json"
            if reserve_path.exists():
                reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
            else:
                completed = [row for row in self.records.values() if row["split"] == "discovery"
                             and row["outcome"] in ("success", "failure")]
                low_events = min(sum(row["success"] for row in completed), sum(not row["success"] for row in completed)) < 20
                reserve = {"extra_pairs": 6 if low_events else 0,
                           "candidates": [row["candidate_id"] for row in self.ranking()[:4]] if low_events else [],
                           "reason": "One bounded reserve for class scarcity; never inspect validation or run indefinitely."}
                atomic_json(reserve_path, reserve)
            if reserve["extra_pairs"]:
                extended = self.args.discovery_repeats + reserve["extra_pairs"]
                self.run_stage("04_reserve_clean", ["clean"], "discovery", extended)
                self.run_stage("05_event_reserve", reserve["candidates"], "discovery", extended)
            plan = self.validation_plan(self.ranking(), recovery)
            for spec in plan["candidates"]:
                self.register(spec)
            self.checkpoint("validation_locked")
            _analysis.freeze_models(self.directory)
            self.run_stage("06_validation_clean", ["clean"], "validation", self.args.validation_repeats)
            self.run_stage("07_validation", [spec["id"] for spec in plan["candidates"]], "validation", self.args.validation_repeats)
            self.checkpoint("complete", "complete")
        except KeyboardInterrupt:
            self.checkpoint(self.manifest.get("phase", "unknown"), "interrupted")
            raise
        except Exception:
            self.checkpoint(self.manifest.get("phase", "unknown"), "error")
            raise
        finally:
            if self._pool is not None:
                self._pool.close(terminate=self.manifest.get("status") != "complete")
                self._pool = None
        return self.directory


def estimate(args):
    candidates = len(search_candidates(args))
    refinement, recovery = 6, 8
    validation = args.validation_repeats * (1 + args.finalists * (1 + args.matched_controls)
                                           + len(delay_levels(args)) + recovery)
    return {"base_candidates": candidates, "base_discovery_episodes": candidates * args.discovery_repeats,
            "refinement_candidate_upper_bound": refinement, "refinement_discovery_episodes": refinement * args.discovery_repeats,
            "recovery_candidate_upper_bound": recovery, "validation_episode_upper_bound": validation,
            "event_reserve_episode_upper_bound": 30,
            "total_episode_upper_bound": args.discovery_repeats * (1 + candidates + refinement + recovery) + validation + 30,
            "delay_cap_ms": 600}


def run(args):
    if args.analyze_only:
        return _analysis.analyze(pathlib.Path(args.analyze_only))
    if args.resume:
        directory = pathlib.Path(args.resume)
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        overrides = getattr(args, "_connection_overrides", set())
        defaults = Args()
        updates = {name: getattr(args, name) for name in ("host", "port", "workers", "servers")
                   if name in overrides or getattr(args, name) != getattr(defaults, name)}
        if {"host", "port"} & updates.keys() and "servers" not in updates:
            updates["servers"] = ""
        args = dataclasses.replace(Args(**saved["config"]), resume=str(directory), dry_run=False, analyze_only=None, **updates)
        if saved.get("experiment") != EXPERIMENT or saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Requires a mismatch-risk manifest")
        if saved.get("status") == "complete":
            return _analysis.analyze(directory)
    else:
        directory = pathlib.Path(args.output_dir) / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    validate_args(args, minimum_validation_repeats=3 if args.resume else 50)
    logging.info("Bounded experiment estimate: %s", json.dumps(estimate(args)))
    logging.info("Results and resumable checkpoint: %s", directory)
    if args.workers > 1 and len(_workers.parse_servers(args.servers, args.host, args.port)) == 1:
        logging.info("One server serializes inference; workers overlap rendering/simulation. "
                     "Use --servers for independent policy replicas.")
    if args.dry_run:
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "search_plan.json"
        endpoints = _workers.parse_servers(args.servers, args.host, args.port)
        atomic_json(path, {"experiment": EXPERIMENT, "config": dataclasses.asdict(args), "estimate": estimate(args),
                           "worker_servers": [_workers.server_address(*endpoints[index % len(endpoints)])
                                              for index in range(args.workers)],
                           "candidates": search_candidates(args),
                           "protocol": {"delay_cap_ms": 600, "workers": args.workers}})
        print(json.dumps(estimate(args), indent=2))
        return path
    import matplotlib

    matplotlib.use("Agg")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError("task_id out of range")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    endpoints = _workers.parse_servers(args.servers, args.host, args.port)
    client = _workers.create_client(*endpoints[0])
    try:
        session = MismatchSession(args, directory, client, task, states, client.get_server_metadata())
        session.execute()
    finally:
        client._ws.close()
    return _analysis.analyze(directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Args()
    for field in dataclasses.fields(Args):
        default = getattr(defaults, field.name)
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=argparse.SUPPRESS)
        else:
            kind = int if field.name == "max_steps" else str if default is None else type(default)
            parser.add_argument(*flags, dest=field.name, type=kind, default=argparse.SUPPRESS)
    values = vars(parser.parse_args(argv))
    if values.get("servers") and {"host", "port"} & values.keys():
        parser.error("Use either --servers or --host/--port")
    if "resume" in values:
        conflicting = set(values) - {"resume", "host", "port", "workers", "servers"}
        if conflicting:
            parser.error("--resume restores scientific configuration; only connection/worker options may be overridden")
    args = Args(**values)
    args._connection_overrides = set(values) & {"host", "port", "workers", "servers"}
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print(run(parse_args()))
    except KeyboardInterrupt:
        logging.warning("Interrupted; resume with --resume <saved-run-directory>")
        raise SystemExit(130)
