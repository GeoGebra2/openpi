"""Constant camera delay for the entire task, followed by three empirical failure-rate bar charts."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import pathlib
from typing import Optional

import numpy as np

try:
    from examples.libero import analyze_full_task_mismatch as analysis
    from examples.libero import run_mismatch_risk as common
except ImportError:
    import analyze_full_task_mismatch as analysis
    import run_mismatch_risk as common


workers = common._workers
trajectory = common._trajectory
metrics = common._metrics
EXPERIMENT = "libero_full_task_mismatch"
SCHEMA_VERSION = 1
CALIBRATION_SEED_OFFSET = 1_000_000


def args_from_config(config):
    """Load manifests written before optional multi-task fields were added."""
    names = {field.name for field in dataclasses.fields(Args)}
    return Args(**{name: config[name] for name in names if name in config})


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    servers: str = ""
    workers: int = 1
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    task_ids: Optional[str] = None
    all_tasks: bool = False
    initial_state_id: int = 0
    prompt: Optional[str] = None
    delay_target: str = "observation/image"
    delay_levels_ms: str = "100,200,300,400,500"
    repeats: int = 50
    bins: int = 10
    calibration_repeats: int = 6
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 1
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = True
    video_fps: int = 20
    object_names: str = ""
    output_dir: str = "rollout/libero_full_task_mismatch"
    resume: Optional[str] = None
    dry_run: bool = False
    analyze_only: Optional[str] = None


def validate_args(args):
    if args.delay_target not in common._critical._shift.CAMERA_KEYS:
        raise ValueError("delay_target must select exactly one camera")
    if min(args.repeats, args.bins, args.workers, args.replan_steps, args.resize_size, args.video_fps) < 1:
        raise ValueError("repeats, bins, workers, replan_steps, resize_size and video_fps must be positive")
    if not 3 <= args.calibration_repeats < CALIBRATION_SEED_OFFSET or args.repeats >= CALIBRATION_SEED_OFFSET:
        raise ValueError("Need at least three calibration repeats, and repeat counts below 1000000")
    if args.seed < 0 or args.num_steps_wait < 0 or args.initial_state_id < 0 or args.task_id < 0:
        raise ValueError("seed, warmup, task and initial state IDs must be nonnegative")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    if args.save_video and args.video_fps > args.control_hz:
        raise ValueError("video_fps must not exceed control_hz; lower fps subsamples without slowing playback")
    if args.task_suite_name not in common._critical._shift.MAX_STEPS_BY_SUITE:
        raise ValueError("Unsupported LIBERO task suite")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    levels = common.delay_levels(args)
    endpoints = workers.parse_servers(args.servers, args.host, args.port)
    if args.workers < len(endpoints):
        raise ValueError("workers must be at least the number of policy servers")
    return levels, args.max_steps or common._critical._shift.MAX_STEPS_BY_SUITE[args.task_suite_name]


def estimate(args):
    levels, maximum = validate_args(args)
    return {"delay_levels_ms": levels, "repeats_per_delay": args.repeats,
            "calibration_episodes": args.calibration_repeats, "sweep_episodes": len(levels) * args.repeats,
            "total_episodes": args.calibration_repeats + len(levels) * args.repeats,
            "bins_per_metric": args.bins, "max_steps": maximum,
            "task_selection": "all tasks in suite" if args.all_tasks else args.task_ids or str(args.task_id),
            "input": "Selected camera delayed for every executed task step; other camera and self state current."}


def select_task_ids(args, suite):
    """Resolve one task, an explicit task list, or every task in a LIBERO suite."""
    if args.all_tasks and args.task_ids:
        raise ValueError("Use only one of --all-tasks and --task-ids")
    if args.all_tasks:
        values = list(range(suite.n_tasks))
    elif args.task_ids:
        try:
            values = [int(value.strip()) for value in args.task_ids.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError("task_ids must be a comma-separated list of integers") from error
        if not values or len(values) != len(set(values)):
            raise ValueError("task_ids must contain at least one distinct task ID")
    else:
        values = [args.task_id]
    if min(values) < 0 or max(values) >= suite.n_tasks:
        raise ValueError(f"task IDs must be in [0, {suite.n_tasks - 1}]")
    return values


class FullTaskSession:
    def __init__(self, args, directory, client, task, initial_states, metadata,
                 episode_runner=None, client_factory=None):
        self.args, self.directory, self.client, self.task = args, pathlib.Path(directory), client, task
        self.levels, self.max_steps = validate_args(args)
        if args.initial_state_id >= len(initial_states):
            raise ValueError("initial_state_id out of range")
        self.initial_state = np.asarray(initial_states[args.initial_state_id])
        self.history_steps = int(round(max(self.levels) * args.control_hz / 1000))
        self.noise_shape = trajectory.noise_shape_from_metadata(metadata)
        if args.replan_steps > self.noise_shape[0] or not callable(getattr(client, "infer_with_noise", None)):
            raise ValueError("Explicit-noise client required; replan_steps cannot exceed action horizon")
        self.endpoints = workers.parse_servers(args.servers, args.host, args.port)
        self.episode_runner = episode_runner or trajectory.run_episode
        self.client_factory = client_factory or workers.create_client
        self._pool, self.records = None, {}
        self.directory.mkdir(parents=True, exist_ok=True)
        scientific = dataclasses.asdict(args)
        for key in ("host", "port", "servers", "workers", "bins", "output_dir", "resume", "dry_run", "analyze_only",
                    "save_video", "video_fps"):
            scientific.pop(key)
        state_hash = hashlib.sha256(np.asarray(self.initial_state, dtype=np.float64).tobytes()).hexdigest()
        path = self.directory / "manifest.json"
        if path.exists():
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            saved_scientific = dict(self.manifest.get("scientific_config", {}))
            for name in ("save_video", "video_fps"):
                saved_scientific.pop(name, None)
            # Manifests from the single-task runner predate task_ids/all_tasks.
            for name in ("task_ids", "all_tasks"):
                saved_scientific.setdefault(name, getattr(Args(), name))
            if (self.manifest.get("experiment") != EXPERIMENT
                    or self.manifest.get("schema_version") != SCHEMA_VERSION
                or saved_scientific != scientific
                    or self.manifest["initial_state_sha256"] != state_hash
                    or self.manifest["server_metadata"] != json.loads(json.dumps(metadata))):
                raise ValueError("Resume configuration/state/server metadata differs from the saved experiment")
            for record_path in sorted(self.directory.glob("episodes/*/*/pair_*/record.json")):
                row = json.loads(record_path.read_text(encoding="utf-8"))
                self.records[(row["phase"], row["condition"], row["pair_index"])] = row
        else:
            self.manifest = {"experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION,
                             "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "scientific_config": scientific, "initial_state_sha256": state_hash,
                             "server_metadata": metadata, "status": "created",
                             "protocol": {"delay_scope": "Whole executed task, including time beyond clean duration.",
                                          "state": "Current self state and other camera; image-source state is measurement only.",
                                          "pairing": "Same seed, initial state, warmup and explicit policy noise across delays.",
                                          "duration": "Executed actions / control_hz; excludes warmup, retains failed timeouts.",
                                          "bins": "Separate equal-width observed ranges, sweep outcomes only; empty bins are not zero risk.",
                                          "fit": "Empirical counts only; no fitted risk model or window search."}}
        self.manifest["config"] = dataclasses.asdict(args)
        self.manifest["protocol"]["video"] = (
            "Top=current agentview/wrist, bottom=actual last policy input, held between queries. "
            "Source time and age use simulation time including negative warmup sources. "
            "Streaming MP4 at video_fps with control-step subsampling; timing JSONL retains every captured step.")
        self.manifest.setdefault("execution_history", []).append({
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "workers": args.workers, "servers": [workers.server_address(*e) for e in self.endpoints],
            "video": {"enabled": args.save_video, "fps": args.video_fps,
                      "layout": "current_vs_actual_policy_v1"},
            "implementation_sha256": {pathlib.Path(p).name: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
                                      for p in (__file__, analysis.__file__, trajectory.__file__, metrics.__file__,
                                                workers.__file__, common.__file__, trajectory._video.__file__)}})

    def checkpoint(self, phase, status="running"):
        self.manifest.update(phase=phase, status=status)
        common.atomic_json(self.directory / "manifest.json", self.manifest)

    def prepare(self, key):
        phase, condition, pair = key
        delay = 0 if phase == "calibration" else int(condition[1:])
        steps = int(round(delay * self.args.control_hz / 1000))
        directory = self.directory / "episodes" / phase / condition / f"pair_{pair:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        if key in self.records:
            with (directory / "previous_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.records[key], allow_nan=False) + "\n")
        repeat = pair + (CALIBRATION_SEED_OFFSET if phase == "calibration" else 0)
        target = self.args.delay_target if delay else common._critical._shift.BASELINE_TARGET
        windows = [[0, self.max_steps]] if delay else []
        prompt = self.args.prompt.strip() if self.args.prompt and self.args.prompt.strip() else str(self.task.language)
        kwargs = dict(args=self.args, task=self.task, initial_state=self.initial_state, prompt=prompt,
                      setting=trajectory.Setting(condition, target, delay, steps), repeat_index=repeat,
                      noise_shape=self.noise_shape, warmup_steps=max(self.args.num_steps_wait, self.history_steps),
                      history_steps=self.history_steps, max_steps=self.max_steps, output_dir=directory,
                      capture_state_provenance=True, record_comparison_video=True)
        return {"episode_kwargs": kwargs, "windows": windows, "delay_steps": steps}

    def finish(self, key, record):
        phase, condition, pair = key
        directory = self.directory / "episodes" / phase / condition / f"pair_{pair:04d}"
        record["trace_path"] = (directory / record["trace_path"]).relative_to(self.directory).as_posix()
        for name in ("video_path", "video_timing_path"):
            if record.get(name):
                record[name] = (directory / record[name]).relative_to(self.directory).as_posix()
        record.update(phase=phase, pair_index=pair)
        if record["outcome"] in ("success", "failure"):
            try:
                trace = common._critical.read_trace(self.directory, record)
                if record["steps"] != len(trace["executed_actions"]) or record["steps"] < 1:
                    raise ValueError("Invalid executed task duration")
                reference = next((r for (p, c, i), r in self.records.items()
                                  if p == "sweep" and phase == "sweep" and i == pair
                                  and c != condition and r["outcome"] in ("success", "failure")), None)
                reference_trace = common._critical.read_trace(self.directory, reference) if reference else None
                windows = [[0, self.max_steps]] if phase == "sweep" else []
                common.audit_trace(self.args, record, trace, windows, reference_trace)
                if phase == "sweep":
                    if record["n_delayed_queries"] != record["n_queries"]:
                        raise ValueError("Delay was not applied to every policy query")
                    record.update(metrics.exposure(trace, self.manifest["mismatch_calibration"], self.args.control_hz)[0])
                    record["task_s"] = record["steps"] / self.args.control_hz
                    record["m_mean"] = record["m_area_s"] / record["task_s"]
                common.compact_trace(self.directory / record["trace_path"], trace)
            except Exception as error:
                record.update(outcome="error", success=False, error=f"Provenance/metric audit failed: {error!r}")
        common.atomic_json(directory / "record.json", record)
        self.records[key] = record
        common.atomic_json(self.directory / "progress.json", {"last_trial": list(key),
                           "completed_trials": sum(r["outcome"] in ("success", "failure") for r in self.records.values())})
        if record["outcome"] == "error":
            raise RuntimeError(f"Runtime error, not a task failure: {record.get('error')}; resume {self.directory}")
        if record["outcome"] == "interrupted":
            raise KeyboardInterrupt()
        logging.info("%s pair=%d %s steps=%d M_max=%s A_M=%s M_d=%s", condition, pair, record["outcome"],
                     record["steps"], record.get("m_peak"), record.get("m_area_s"), record.get("m_mean"))
        if record.get("video_path"):
            logging.info("Video: %s", self.directory / record["video_path"])
        if record.get("video_error"):
            logging.warning("%s pair=%d video_error=%s", condition, pair, record["video_error"])

    def run_stage(self, phase, conditions, repeats):
        self.checkpoint(phase)
        pending = []
        for pair in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, 314159, pair]))
            for index in rng.permutation(len(conditions)):
                key = (phase, conditions[int(index)], pair)
                old = self.records.get(key)
                if old and old["outcome"] in ("success", "failure"):
                    if not (self.directory / old["trace_path"]).exists():
                        raise ValueError(f"Completed episode lost its trace: {old['trace_path']}")
                    continue
                pending.append((key, self.prepare(key)))
        if not pending:
            return
        if self.args.workers == 1:
            for key, job in pending:
                record = workers.run_rollout_job(job, self.client, self.episode_runner)
                record["policy_server"] = workers.server_address(*self.endpoints[0])
                self.finish(key, record)
            return
        if self._pool is None:
            self._pool = workers.RolloutPool(self.args.workers, self.args.host, self.args.port,
                                            self.manifest["server_metadata"], self.episode_runner,
                                            self.client_factory, server_endpoints=self.endpoints)
            self.manifest["execution_history"][-1]["worker_assignments"] = self._pool.assignments
            self.checkpoint(phase)
        for key, record in self._pool.iter_results(pending):
            self.finish(key, record)

    def execute(self):
        try:
            self.run_stage("calibration", ["clean"], self.args.calibration_repeats)
            if "mismatch_calibration" not in self.manifest:
                clean = [dict(r, candidate_id="clean", split="discovery") for r in self.records.values()
                         if r["phase"] == "calibration"]
                duration_filter = common.calibration_from_records(clean)
                normal = [r for r in clean if common.calibration_reference(r, duration_filter)[1] == "paired_clean_duration"]
                self.manifest["mismatch_calibration"] = metrics.calibrate_scales(
                    [common._critical.read_trace(self.directory, r) for r in normal])
                self.manifest["clean_calibration_filter"] = duration_filter
                self.checkpoint("calibrated")
            self.run_stage("sweep", [f"d{d:03d}" for d in self.levels], self.args.repeats)
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


def _run_single(args, suite, directory):
    """Run and analyze exactly one task in the supplied output directory."""
    endpoints = workers.parse_servers(args.servers, args.host, args.port)
    client = workers.create_client(*endpoints[0])
    try:
        session = FullTaskSession(args, directory, client, suite.get_task(args.task_id),
                                  suite.get_task_init_states(args.task_id), client.get_server_metadata())
        session.execute()
    finally:
        client._ws.close()
    return analysis.analyze(directory, args.bins)


def _run_suite(args, suite, task_ids, directory):
    """Run each selected task independently, then produce one aggregate report."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tasks = []
    manifest = {
        "experiment": EXPERIMENT + "_suite", "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_suite_name": args.task_suite_name, "task_ids": task_ids,
        "config": dataclasses.asdict(args), "status": "running", "tasks": tasks,
    }
    common.atomic_json(directory / "suite_manifest.json", manifest)
    try:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            task_directory = directory / f"task_{task_id:03d}"
            child_args = dataclasses.replace(
                args, task_id=task_id, task_ids=None, all_tasks=False,
                output_dir=str(task_directory), resume=None, analyze_only=None,
            )
            # A batch child uses the exact task directory so the aggregate manifest
            # remains stable and can be resumed by rerunning the unfinished task.
            _run_single(child_args, suite, task_directory)
            tasks.append({"task_id": task_id, "task_name": str(task.language),
                          "directory": task_directory.name})
            manifest["tasks"] = tasks
            common.atomic_json(directory / "suite_manifest.json", manifest)
        manifest["status"] = "complete"
        common.atomic_json(directory / "suite_manifest.json", manifest)
    except Exception:
        manifest["status"] = "error"
        common.atomic_json(directory / "suite_manifest.json", manifest)
        raise
    try:
        from examples.libero import analyze_full_task_mismatch_suite as suite_analysis
    except ImportError:
        import analyze_full_task_mismatch_suite as suite_analysis
    return suite_analysis.analyze(directory)


def _resume_suite(args, suite, directory, manifest):
    """Continue completed/partial task subruns recorded in a suite manifest."""
    directory = pathlib.Path(directory)
    manifest["config"] = dataclasses.asdict(args)
    tasks_by_id = {int(task["task_id"]): dict(task) for task in manifest.get("tasks", [])}
    for task_id in manifest["task_ids"]:
        task = suite.get_task(task_id)
        task_directory = directory / f"task_{task_id:03d}"
        child_manifest_path = task_directory / "manifest.json"
        child_args = dataclasses.replace(
            args, task_id=task_id, task_ids=None, all_tasks=False,
            output_dir=str(task_directory), resume=None, analyze_only=None,
        )
        if child_manifest_path.exists():
            child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
            if child_manifest.get("status") == "complete":
                tasks_by_id.setdefault(task_id, {"task_id": task_id, "task_name": str(task.language),
                                                  "directory": task_directory.name})
                continue
        _run_single(child_args, suite, task_directory)
        tasks_by_id[task_id] = {"task_id": task_id, "task_name": str(task.language),
                                "directory": task_directory.name}
        manifest["tasks"] = [tasks_by_id[key] for key in sorted(tasks_by_id)]
        common.atomic_json(directory / "suite_manifest.json", manifest)
    manifest["tasks"] = [tasks_by_id[key] for key in sorted(tasks_by_id)]
    manifest["status"] = "complete"
    common.atomic_json(directory / "suite_manifest.json", manifest)
    try:
        from examples.libero import analyze_full_task_mismatch_suite as suite_analysis
    except ImportError:
        import analyze_full_task_mismatch_suite as suite_analysis
    return suite_analysis.analyze(directory)


def run(args):
    if args.analyze_only:
        bins = args.bins if "bins" in getattr(args, "_overrides", set()) or args.bins != Args().bins else None
        return analysis.analyze(args.analyze_only, bins)
    if args.resume:
        directory = pathlib.Path(args.resume)
        suite_manifest_path = directory / "suite_manifest.json"
        if suite_manifest_path.exists():
            saved = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
            if (saved.get("experiment") != EXPERIMENT + "_suite"
                    or saved.get("schema_version") != SCHEMA_VERSION):
                raise ValueError("Requires a full-task-mismatch suite manifest")
            explicit = getattr(args, "_overrides", set())
            defaults = Args()
            changes = {n: getattr(args, n) for n in ("host", "port", "servers", "workers", "bins", "save_video", "video_fps")
                       if n in explicit or getattr(args, n) != getattr(defaults, n)}
            if {"host", "port"} & changes.keys() and "servers" not in changes:
                changes["servers"] = ""
            args = dataclasses.replace(args_from_config(saved["config"]), resume=str(directory), **changes)
            import matplotlib
            matplotlib.use("Agg")
            from libero.libero import benchmark
            suite = benchmark.get_benchmark_dict()[saved["task_suite_name"]]()
            return _resume_suite(args, suite, directory, saved)
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if saved.get("experiment") != EXPERIMENT or saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Requires a full-task-mismatch manifest")
        explicit = getattr(args, "_overrides", set())
        defaults = Args()
        changes = {n: getattr(args, n) for n in ("host", "port", "servers", "workers", "bins", "save_video", "video_fps")
                   if n in explicit or getattr(args, n) != getattr(defaults, n)}
        if {"host", "port"} & changes.keys() and "servers" not in changes:
            changes["servers"] = ""
        args = dataclasses.replace(args_from_config(saved["config"]), resume=str(directory), **changes)
        if saved.get("status") == "complete":
            return analysis.analyze(directory, args.bins)
    else:
        directory = pathlib.Path(args.output_dir) / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    plan = estimate(args)
    logging.info("Plan: %s; resumable output: %s", json.dumps(plan), directory)
    if args.dry_run:
        path = directory / "experiment_plan.json"
        common.atomic_json(path, {"experiment": EXPERIMENT, "config": dataclasses.asdict(args), "estimate": plan})
        print(json.dumps(plan, indent=2))
        return path
    import matplotlib
    matplotlib.use("Agg")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = select_task_ids(args, suite)
    if len(task_ids) == 1:
        args = dataclasses.replace(args, task_id=task_ids[0])
        return _run_single(args, suite, directory)
    return _run_suite(args, suite, task_ids, directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Args):
        default = getattr(Args(), field.name)
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if field.name == "save_video":
            parser.add_argument(*flags, dest=field.name, action="store_true", default=argparse.SUPPRESS,
                                help="Record annotated current-vs-policy videos (default: enabled)")
            parser.add_argument("--no-video", "--no_video", dest=field.name, action="store_false",
                                default=argparse.SUPPRESS, help="Disable recording for new/unfinished episodes")
        elif isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=argparse.SUPPRESS)
        else:
            kind = int if field.name == "max_steps" else str if default is None else type(default)
            parser.add_argument(*flags, dest=field.name, type=kind, default=argparse.SUPPRESS)
    values = vars(parser.parse_args(argv))
    if values.get("servers") and {"host", "port"} & values.keys():
        parser.error("Use either --servers or --host/--port")
    if values.get("resume") and values.get("analyze_only"):
        parser.error("Use either --resume or --analyze-only")
    if values.get("resume") and set(values) - {"resume", "host", "port", "servers", "workers", "bins", "save_video", "video_fps"}:
        parser.error("--resume restores scientific configuration; only connections, workers, bins and video may change")
    if values.get("analyze_only") and set(values) - {"analyze_only", "bins"}:
        parser.error("--analyze-only accepts only --bins as an additional option")
    args = Args(**values)
    args._overrides = set(values)
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print(run(parse_args()))
    except KeyboardInterrupt:
        logging.warning("Interrupted; resume with --resume <saved-run-directory>")
        raise SystemExit(130)
