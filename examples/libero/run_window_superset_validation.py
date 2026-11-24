"""Independently compare a delayed base window set with every measured strict superset.

One command locks candidates from an earlier critical-window search, runs new
paired closed-loop episodes, then writes plots and paired statistical results.
Compatible with the LIBERO Python 3.8 environment; no diagnostics required.
"""
# Keep optional annotations usable by Python 3.8 consumers; defer simulator imports for dry runs.
# ruff: noqa: PLC0415, UP045

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import pathlib
from typing import Optional

import numpy as np

try:
    from examples.libero import run_critical_window_search as _search
except ImportError:
    import run_critical_window_search as _search

EXPERIMENT = "libero_window_superset_validation"
REPEAT_START = 2_000_000
PLAN_FILE = "superset_plan.json"


@dataclasses.dataclass
class Args:
    source_run: Optional[str] = None
    base_windows: str = "30:40,79:100"  # Percent of each paired clean duration.
    repeats_per_setting: int = 200
    seed: Optional[int] = None  # Preserve original seed; use a new repeat-index range.
    alpha: float = 0.05
    min_failure_difference: float = 0.0  # Fraction, base minus superset.
    host: str = "localhost"
    port: int = 8000
    save_video: bool = False
    output_dir: str = "rollout/libero_window_superset_validation"
    resume: Optional[str] = None
    dry_run: bool = False
    analyze_only: Optional[str] = None


def window_mask(windows, size):
    mask = np.zeros(size, dtype=bool)
    for a, b in windows:
        if not 0 <= a < b <= size:
            raise ValueError("Window outside time axis")
        mask[a:b] = True
    return mask


def parse_base_windows(value, core_args):
    size = _search.grid_size(core_args)
    windows = []
    for item in value.split(","):
        endpoints = item.strip().split(":")
        if len(endpoints) != 2:
            raise ValueError("Use percentage windows such as 30:40,79:100")
        a, b = (float(x) for x in endpoints)
        scaled = np.asarray([a, b]) * size / 100
        if (
            not np.all(np.isfinite(scaled))
            or not 0 <= a < b <= 100
            or not np.allclose(scaled, np.round(scaled), atol=1e-8, rtol=0)
        ):
            raise ValueError("Base endpoints must align to the source search grid and lie in [0,100]")
        windows.append([round(scaled[0]), round(scaled[1])])
    base = _search.make_candidate(core_args, windows, "locked_followup_base")
    if base is None:
        raise ValueError("Base windows are empty or exceed the source coverage limit")
    return base


def build_plan(args):
    if not args.source_run:
        raise ValueError("--source-run is required for a new experiment")
    if not 1 <= args.repeats_per_setting < _search.VALIDATION_OFFSET:
        raise ValueError("repeats_per_setting must be between 1 and 999999")
    if not 0 < args.alpha < 0.5 or not 0 <= args.min_failure_difference < 1:
        raise ValueError("Invalid alpha or minimum failure difference")
    source = pathlib.Path(args.source_run)
    if source.is_file():
        source = source.parent
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != _search.EXPERIMENT or manifest.get("status") != "complete":
        raise ValueError("source_run must be a completed critical-window search")
    source_args = _search.Args(**manifest["config"])
    core_args = dataclasses.replace(
        source_args,
        validation_repeats=args.repeats_per_setting,
        seed=source_args.seed if args.seed is None else args.seed,
        host=args.host,
        port=args.port,
        save_video=args.save_video,
        output_dir=args.output_dir,
        resume=None,
        dry_run=False,
        analyze_only=None,
    )
    _search.validate_args(core_args)
    if _search.VALIDATION_OFFSET + source_args.validation_repeats > REPEAT_START:
        raise ValueError("Follow-up repeat range overlaps source validation")
    base = parse_base_windows(args.base_windows, core_args)
    size = _search.grid_size(core_args)
    base_mask = window_mask(base["windows"], size)
    registry = json.loads((source / "candidates.json").read_text(encoding="utf-8"))
    with (source / "discovery_ranking.csv").open(encoding="utf-8", newline="") as stream:
        measured_ids = {row["candidate_id"] for row in csv.DictReader(stream)}
    supersets = []
    for key in sorted(measured_ids):
        spec = registry[key]
        if spec["kind"] != "windows":
            continue
        candidate = _search.make_candidate(core_args, spec["windows"], "source_discovery_superset")
        if candidate is None:
            continue
        if candidate["id"] != key:
            raise ValueError("Source candidate identity disagrees with its windows")
        mask = window_mask(candidate["windows"], size)
        if np.all(mask[base_mask]) and np.any(mask & ~base_mask):
            supersets.append(candidate)
    supersets.sort(key=lambda x: (x["coverage"], x["windows"]))
    if not supersets:
        raise ValueError("No measured strict supersets found within the coverage budget")
    for index, candidate in enumerate(supersets, 1):
        candidate["label"] = f"S{index:02d}"
    clean = {"id": "clean", "kind": "clean", "windows": [], "coverage": 0.0, "origins": ["control"]}
    return {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_run": str(source.resolve()),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "candidates.json", "discovery_ranking.csv")
        },
        "source_server_metadata": manifest["server_metadata"],
        "source_initial_state_hashes": manifest["initial_state_hashes"],
        "core_config": dataclasses.asdict(core_args),
        "base": base,
        "supersets": supersets,
        "conditions": [clean, base, *supersets],
        "repeat_start": REPEAT_START,
        "repeats": args.repeats_per_setting,
        "alpha": args.alpha,
        "min_failure_difference": args.min_failure_difference,
        "selection": "All strict supersets among completed source discovery candidates; no filtering by outcomes.",
        "mapping": "Protect the mapped base; trim added query cells only if needed. Planned inclusion is audited per pair.",
        "hypothesis": "Base final failure probability is greater than each superset's probability.",
        "statistics": "One-sided exact paired McNemar; Holm over ALL locked superset comparisons. "
        "Positive difference means base fails more. Conservative paired intervals are pointwise.",
        "scope": "Only these measured source candidates, task, initial states and delay; not all possible supersets.",
        "episodes": (len(supersets) + 2) * args.repeats_per_setting,
    }


def nested_windows(core_args, candidate, base, reference_steps):
    """Keep every discretized base cell, even when original budget trimming differs."""
    mapped_base = _search.mapped_windows(core_args, base, reference_steps)
    original = _search.mapped_windows(core_args, candidate, reference_steps)
    q = core_args.replan_steps
    protected = window_mask(mapped_base, reference_steps)
    mask = window_mask(original, reference_steps) | protected
    budget = int(np.floor(min(candidate["coverage"], core_args.max_coverage) * reference_steps / q + 1e-9)) * q
    for start in reversed(range(0, reference_steps // q * q, q)):
        if np.count_nonzero(mask) <= budget:
            break
        if mask[start] and not np.any(protected[start : start + q]):
            mask[start : start + q] = False
    if np.count_nonzero(mask) > budget or not np.all(mask[protected]):
        raise ValueError("Cannot map a superset without violating inclusion/coverage")
    return _search.mask_runs(mask)


class SupersetSession(_search.SearchSession):
    experiment = EXPERIMENT

    def __init__(self, plan, directory, client, task, initial_states, metadata, episode_runner=None):
        self.plan = plan
        if json.loads(json.dumps(metadata)) != plan["source_server_metadata"]:
            raise ValueError("Server capabilities differ from source run; use the same policy configuration")
        core_args = _search.Args(**plan["core_config"])
        super().__init__(core_args, directory, client, task, initial_states, metadata, episode_runner)
        if self.manifest["initial_state_hashes"] != plan["source_initial_state_hashes"]:
            raise ValueError("Initial state contents differ from the source experiment")
        if self.manifest["protocol"] != self.experiment_protocol():
            raise ValueError("Locked superset plan differs from the saved manifest")
        self.manifest["estimate"] = {
            "supersets": len(plan["supersets"]),
            "conditions": len(plan["conditions"]),
            "repeats_per_setting": plan["repeats"],
            "episodes": plan["episodes"],
        }
        for name in ("run_window_superset_validation.py", "analyze_window_superset_validation.py"):
            path = pathlib.Path(__file__).with_name(name)
            self.manifest["implementation_sha256"].setdefault(name, hashlib.sha256(path.read_bytes()).hexdigest())
        for spec in plan["conditions"]:
            self.register(spec)
        self.checkpoint(self.manifest.get("phase", "locked"), self.manifest["status"])

    def experiment_protocol(self):
        result = _search.protocol(self.args)
        result.update(
            splits=f"New locked follow-up only; repeat_index begins at {self.plan['repeat_start']}.",
            random_control="No random placement control: this follow-up tests fixed set inclusion.",
            statistics=self.plan["statistics"],
            scope=self.plan["scope"],
            mapping=self.plan["mapping"],
            minimum_failure_increase=None,
            plan_sha256=hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest(),
        )
        return result

    def repeat_index_for(self, split, pair_index):
        return self.plan["repeat_start"] + pair_index

    def map_trial_windows(self, spec, reference_steps, repeat_index):
        if spec["kind"] == "clean":
            return []
        return nested_windows(self.args, spec, self.plan["base"], reference_steps)

    def execute_trial(self, split, candidate_id, pair_index):
        record = super().execute_trial(split, candidate_id, pair_index)
        if candidate_id != "clean":
            spec = self.registry[candidate_id]
            mapped_base = _search.mapped_windows(self.args, self.plan["base"], record["reference_steps"])
            base_mask = window_mask(mapped_base, record["reference_steps"])
            actual_mask = window_mask(record["windows_steps"], record["reference_steps"])
            if not np.all(actual_mask[base_mask]):
                raise RuntimeError("Planned window inclusion audit failed")
            record.update(
                planned_contains_base=True,
                mapped_base_windows=mapped_base,
                effective_equal_to_base=bool(np.array_equal(base_mask, actual_mask)),
                mapped_extra_steps=int(np.count_nonzero(actual_mask & ~base_mask)),
                mapping_repaired=record["windows_steps"]
                != _search.mapped_windows(self.args, spec, record["reference_steps"]),
            )
            path = self.directory / "episodes" / split / candidate_id / f"pair_{pair_index:04d}" / "record.json"
            _search.atomic_json(path, record)
        return record

    def execute(self):
        self.checkpoint("paired_validation")
        try:
            for pair in range(self.plan["repeats"]):
                # Each clean duration is available before any intervention of its pair.
                self.execute_trial("validation", "clean", pair)
                conditions = self.plan["conditions"][1:]
                rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, pair, 719331]))
                for index in rng.permutation(len(conditions)):
                    spec = conditions[int(index)]
                    record = self.execute_trial("validation", spec["id"], pair)
                    logging.info(
                        "pair %d/%d %s %s steps=%d actual_coverage=%.1f%%",
                        pair + 1,
                        self.plan["repeats"],
                        spec.get("label", "BASE"),
                        record["outcome"],
                        record["steps"],
                        record["actual_coverage"] * 100,
                    )
            self.checkpoint("complete", "complete")
        except KeyboardInterrupt:
            self.checkpoint("paired_validation", "interrupted")
            raise
        except Exception:
            self.checkpoint("paired_validation", "error")
            raise


def analyze_results(directory):
    try:
        from examples.libero import analyze_window_superset_validation as analysis
    except ImportError:
        import analyze_window_superset_validation as analysis
    return analysis.analyze(directory)


def run(args):
    if args.analyze_only:
        return analyze_results(pathlib.Path(args.analyze_only))
    if args.resume:
        directory = pathlib.Path(args.resume)
        plan = json.loads((directory / PLAN_FILE).read_text(encoding="utf-8"))
        if plan.get("experiment") != EXPERIMENT:
            raise ValueError("Not a superset validation directory")
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete":
            return analyze_results(directory)
    else:
        plan = build_plan(args)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = pathlib.Path(args.output_dir) / stamp
        directory.mkdir(parents=True, exist_ok=False)
        _search.atomic_json(directory / PLAN_FILE, plan)
    logging.info(
        "Locked %d supersets + BASE + clean; %d repeats = %d episodes; output: %s",
        len(plan["supersets"]),
        plan["repeats"],
        plan["episodes"],
        directory,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "supersets": [
                        {"label": s["label"], "windows": s["windows"], "coverage": s["coverage"]}
                        for s in plan["supersets"]
                    ],
                    "episodes": plan["episodes"],
                    "grid_size": _search.grid_size(_search.Args(**plan["core_config"])),
                },
                indent=2,
            )
        )
        return directory / PLAN_FILE
    # Validate analysis dependency before connecting or collecting expensive episodes.
    import matplotlib as mpl

    mpl.use("Agg")
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    config = plan["core_config"]
    suite = benchmark.get_benchmark_dict()[config["task_suite_name"]]()
    task = suite.get_task(config["task_id"])
    states = suite.get_task_init_states(config["task_id"])
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        session = SupersetSession(plan, directory, client, task, states, client.get_server_metadata())
        try:
            session.execute()
        finally:
            analyze_results(directory)  # Also save clearly labelled partial reports on interruption/error.
    finally:
        client._ws.close()  # noqa: SLF001 - existing websocket client exposes no public close().
    return directory / "analysis" / "report.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Args):
        default = getattr(Args(), field.name)
        flags = list(dict.fromkeys(("--" + field.name.replace("_", "-"), "--" + field.name)))
        if isinstance(default, bool):
            parser.add_argument(*flags, dest=field.name, action="store_true", default=argparse.SUPPRESS)
        else:
            kind = int if field.name == "seed" else type(default) if default is not None else str
            parser.add_argument(*flags, dest=field.name, type=kind, default=argparse.SUPPRESS)
    values = vars(parser.parse_args(argv))
    if "resume" in values and set(values) - {"resume", "host", "port"}:
        parser.error("--resume restores the locked plan; only host/port may be overridden")
    return Args(**values)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print(run(parse_args()))
    except KeyboardInterrupt:
        logging.warning("Interrupted. Continue the locked experiment with --resume <run-directory>")
