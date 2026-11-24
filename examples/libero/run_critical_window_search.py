"""Automatically search and validate time windows sensitive to camera delay.

One command runs clean references, exhaustive coarse combinations, global fine
single-window/random scans, local refinement, and locked held-out validation.
Only final LIBERO outcomes rank/confirm windows. No manually labelled stages.
Requires the existing explicit-noise policy client/server and LIBERO renderer.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import hashlib
import itertools
import json
import logging
import math
import pathlib
import sys
from typing import Optional

import numpy as np

try:
    from examples.libero import critical_window_workers as _workers
    from examples.libero import run_trajectory_delay as _trajectory
except ImportError:
    import critical_window_workers as _workers
    import run_trajectory_delay as _trajectory

_shift = _trajectory._shift
EXPERIMENT = "libero_critical_window_search"
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
    delay_ms: float = 600.0
    max_coverage: float = 0.5
    coverage_budgets: str = "0.1,0.2,0.3,0.4,0.5"
    coarse_bins: int = 10
    fine_bins: int = 20
    refinement_factor: int = 5
    random_candidates: int = 64
    refinement_rounds: int = 1
    refine_top: int = 3
    discovery_repeats: int = 10
    validation_repeats: int = 50
    matched_controls: int = 1
    min_failure_increase: float = 0.2
    alpha: float = 0.05
    seed: int = 7
    control_hz: float = 20.0
    replan_steps: int = 1
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    save_video: bool = False
    video_fps: int = 20
    object_names: str = ""
    keep_full_traces: bool = False
    output_dir: str = "rollout/libero_critical_windows"
    resume: Optional[str] = None
    dry_run: bool = False
    analyze_only: Optional[str] = None


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def grid_size(args):
    return args.fine_bins * args.refinement_factor


def budgets_for(args):
    values = [float(value.strip()) for value in args.coverage_budgets.split(",") if value.strip()]
    if not values or any(not np.isfinite(value) or not 0 < value <= 0.5 for value in values):
        raise ValueError("coverage_budgets must contain fractions in (0, 0.5]")
    return sorted({value for value in values if value <= args.max_coverage} | {args.max_coverage})


def state_ids(args):
    ids = [int(item.strip()) for item in args.initial_state_ids.split(",") if item.strip()]
    ids = ids or [args.initial_state_id]
    if min(ids) < 0 or len(ids) != len(set(ids)):
        raise ValueError("initial state IDs must be distinct nonnegative integers")
    return ids


def validate_args(args):
    if args.task_suite_name not in _shift.MAX_STEPS_BY_SUITE or args.delay_target not in _shift.CAMERA_KEYS:
        raise ValueError("Choose a supported task suite and exactly one delay camera")
    positive = (args.coarse_bins, args.fine_bins, args.refinement_factor, args.discovery_repeats,
                args.validation_repeats, args.matched_controls, args.replan_steps,
                args.resize_size, args.video_fps, args.workers)
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("Grid sizes, repeats, controls, replanning, workers and image/video sizes must be positive integers")
    if args.workers < len(_workers.parse_servers(args.servers, args.host, args.port)):
        raise ValueError("--workers must be at least the number of --servers addresses")
    if args.fine_bins % args.coarse_bins or args.coarse_bins > 16:
        raise ValueError("fine_bins must be a multiple of coarse_bins; coarse_bins must be <=16")
    if min(args.random_candidates, args.refinement_rounds, args.refine_top, args.seed, args.num_steps_wait) < 0:
        raise ValueError("Counts, seed and warmup must be nonnegative")
    if args.discovery_repeats >= VALIDATION_OFFSET or args.validation_repeats >= VALIDATION_OFFSET:
        raise ValueError("Repeat counts would overlap discovery and validation seed ranges")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    exact = args.delay_ms * args.control_hz / 1000.0
    if not np.isfinite(exact) or exact <= 0 or not np.isclose(exact, round(exact), atol=1e-8, rtol=0):
        raise ValueError("delay_ms must be positive and align to the simulator frame period")
    if not np.isfinite(args.max_coverage) or not 0 < args.max_coverage <= 0.5:
        raise ValueError("max_coverage must be in (0, 0.5]")
    if math.floor(args.fine_bins * args.max_coverage + 1e-9) < 1:
        raise ValueError("fine_bins is too small to represent a nonempty window within max_coverage")
    if not 0 < args.min_failure_increase <= 1 or not 0 < args.alpha < 0.5:
        raise ValueError("Invalid minimum failure increase or alpha")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    budgets_for(args)
    state_ids(args)
    return int(round(exact)), args.max_steps or _shift.MAX_STEPS_BY_SUITE[args.task_suite_name]


def merge_windows(windows):
    merged = []
    for start, end in sorted((int(a), int(b)) for a, b in windows if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def window_length(windows):
    return sum(end - start for start, end in merge_windows(windows))


def make_candidate(args, windows, origin):
    windows = merge_windows(windows)
    size = grid_size(args)
    if (not windows or windows[0][0] < 0 or windows[-1][1] > size
            or window_length(windows) > math.floor(args.max_coverage * size + 1e-9)):
        return None
    digest = hashlib.sha256(json.dumps(windows, separators=(",", ":")).encode()).hexdigest()[:16]
    return {"id": "w_" + digest, "kind": "windows", "windows": windows,
            "coverage": window_length(windows) / size, "origins": [origin]}


def mask_candidate(args, indices, bins, origin):
    width = grid_size(args) // bins
    return make_candidate(args, [(index * width, (index + 1) * width) for index in indices], origin)


def unique_candidates(candidates):
    found = {}
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate["id"] in found:
            found[candidate["id"]]["origins"] = sorted(set(found[candidate["id"]]["origins"] + candidate["origins"]))
        else:
            found[candidate["id"]] = candidate
    return list(found.values())


def coarse_candidates(args):
    return unique_candidates(
        mask_candidate(args, indices, args.coarse_bins, "coarse_exhaustive")
        for count in range(1, math.floor(args.coarse_bins * args.max_coverage + 1e-9) + 1)
        for indices in itertools.combinations(range(args.coarse_bins), count)
    )


def fine_candidates(args):
    candidates = [mask_candidate(args, range(start, start + length), args.fine_bins, "fine_single")
                  for length in range(1, math.floor(args.fine_bins * args.max_coverage + 1e-9) + 1)
                  for start in range(args.fine_bins - length + 1)]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 71013]))
    limit = math.floor(args.fine_bins * args.max_coverage + 1e-9)
    random = {}
    for _ in range(max(100, args.random_candidates * 100)):
        if len(random) >= args.random_candidates or limit < 1:
            break
        count = int(rng.integers(1, limit + 1))
        candidate = mask_candidate(args, rng.choice(args.fine_bins, count, replace=False),
                                   args.fine_bins, "fine_random")
        if candidate:
            random[candidate["id"]] = candidate
    return unique_candidates(candidates + list(random.values()))


def refine_candidates(args, parents, round_index):
    result = []
    size = grid_size(args)
    fine_width = size // args.fine_bins
    for parent in parents:
        windows = parent["windows"]
        origin = f"refinement_{round_index}:{parent['id']}"
        for index, (start, end) in enumerate(windows):
            for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
                a, b = start + da, end + db
                if 0 <= a < b <= size:
                    result.append(make_candidate(args, windows[:index] + [[a, b]] + windows[index + 1:], origin))
            result.append(make_candidate(args, windows[:index] + windows[index + 1:], origin))
        # Add or remove each fine cell across the whole task, not just near a peak.
        active = np.zeros(size, dtype=bool)
        for start, end in windows:
            active[start:end] = True
        for start in range(0, size, fine_width):
            for value in (False, True):
                updated = active.copy()
                updated[start:start + fine_width] = value
                result.append(make_candidate(args, mask_runs(updated), origin))
    return unique_candidates(result)


def mask_runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def mapped_windows(args, candidate, reference_steps):
    """Round to common query boundaries and trim any rounding-induced overbudget."""
    if reference_steps < 1:
        raise ValueError("Reference episode must contain actions")
    q, size = args.replan_steps, grid_size(args)
    units = reference_steps // q
    windows = merge_windows([(a * units // size * q, b * units // size * q)
                             for a, b in candidate["windows"]])
    fraction = min(args.max_coverage, candidate["coverage"])
    budget = math.floor(fraction * reference_steps / q + 1e-9) * q
    while window_length(windows) > budget:
        index = max(range(len(windows)), key=lambda i: (windows[i][1] - windows[i][0], i))
        windows[index][1] -= q
        windows = merge_windows(windows)
    return windows


def random_shape_windows(windows, total_units, rng):
    """Uniform gap compositions preserve each window length and window count."""
    lengths = [b - a for a, b in windows]
    count = len(lengths)
    if not count:
        return []
    free = total_units - sum(lengths) - (count - 1)
    if free < 0:
        raise ValueError("Not enough room for separated shape-matched windows")
    lengths = [lengths[i] for i in rng.permutation(count)]
    bars = sorted(int(value) for value in rng.choice(free + count, count, replace=False))
    gaps = np.diff([-1, *bars, free + count]) - 1
    result, cursor = [], int(gaps[0])
    for index, length in enumerate(lengths):
        result.append([cursor, cursor + length])
        cursor += length
        if index + 1 < count:
            cursor += 1 + int(gaps[index + 1])
    return result


def trial_windows(args, spec, registry, reference_steps, repeat_index):
    if spec["kind"] == "clean":
        return []
    source = registry[spec["source_id"]] if spec["kind"] == "matched_random" else spec
    windows = mapped_windows(args, source, reference_steps)
    if spec["kind"] != "matched_random":
        return windows
    q = args.replan_steps
    salt = int(hashlib.sha256(spec["id"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, repeat_index, salt, 90017]))
    unit_windows = [[a // q, b // q] for a, b in windows]
    # Do not reject overlap with the candidate: that would bias the random control.
    return [[a * q, b * q] for a, b in random_shape_windows(unit_windows, reference_steps // q, rng)]


def window_delay(step, windows, delay_steps):
    return delay_steps if any(start <= step < end for start, end in windows) else 0


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None, None
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return max(0.0, centre - radius), min(1.0, centre + radius)


def exact_paired_p(harmed, rescued):
    """One-sided exact McNemar test: intervention increases failure probability."""
    n = harmed + rescued
    if not n:
        return 1.0
    return sum(math.comb(n, k) for k in range(harmed, n + 1)) / (1 << n)


def paired_statistics(reference, intervention, expected):
    reference = {row["pair_index"]: row for row in reference if row["outcome"] in ("success", "failure")}
    intervention = {row["pair_index"]: row for row in intervention if row["outcome"] in ("success", "failure")}
    pairs = [(reference[key], intervention[key]) for key in sorted(reference.keys() & intervention.keys())]
    n = len(pairs)
    harmed = sum(a["success"] and not b["success"] for a, b in pairs)
    rescued = sum(not a["success"] and b["success"] for a, b in pairs)
    base_failures = sum(not a["success"] for a, _ in pairs)
    failures = sum(not b["success"] for _, b in pairs)
    low, high = wilson(failures, n)
    # Simultaneous marginal 97.5% Wilson intervals for discordant event rates;
    # their Bonferroni difference is conservative and non-degenerate at 0/1.
    bl, bu = wilson(harmed, n, z=2.241402727604947)
    cl, cu = wilson(rescued, n, z=2.241402727604947)
    ratios = [b.get("actual_coverage", 0.0) for _, b in pairs]
    return {
        "completed_pairs": n, "expected_pairs": expected, "complete": n == expected,
        "reference_failures": base_failures, "failures": failures,
        "reference_failure_rate": base_failures / n if n else None,
        "failure_rate": failures / n if n else None,
        "failure_ci_low": low, "failure_ci_high": high,
        "failure_increase": (harmed - rescued) / n if n else None,
        "increase_ci_low": max(-1.0, bl - cu) if n else None,
        "increase_ci_high": min(1.0, bu - cl) if n else None,
        "clean_success_to_failure": harmed, "clean_failure_to_success": rescued,
        "p_one_sided": exact_paired_p(harmed, rescued) if n else None,
        "actual_coverage_max": max(ratios) if ratios else None,
        "actual_budget_violations": sum(not b.get("actual_budget_valid", True) for _, b in pairs),
        "episodes_without_delayed_queries": sum(not b.get("n_delayed_queries", 0) for _, b in pairs),
    }


def holm_adjust(pvalues):
    result, floor = {}, 0.0
    ordered = sorted((p, key) for key, p in pvalues.items() if p is not None)
    for index, (p, key) in enumerate(ordered):
        floor = max(floor, min(1.0, (len(ordered) - index) * p))
        result[key] = floor
    return result


def discovery_ranking(args, registry, records):
    grouped = collections.defaultdict(list)
    for record in records:
        if record["split"] == "discovery":
            grouped[record["candidate_id"]].append(record)
    rows = []
    for key, spec in registry.items():
        if spec["kind"] != "windows" or key not in grouped:
            continue
        stats = paired_statistics(grouped["clean"], grouped[key], args.discovery_repeats)
        if stats["complete"]:
            rows.append({"candidate_id": key, "coverage": spec["coverage"], "windows": spec["windows"], **stats})
    return sorted(rows, key=lambda row: (-row["failure_increase"], row["coverage"],
                                        len(row["windows"]), row["candidate_id"]))


def budget_leaders(args, ranking):
    leaders = []
    for budget in budgets_for(args):
        eligible = [row for row in ranking if row["coverage"] <= budget + 1e-10]
        if eligible:
            leaders.append({"budget": budget, "candidate_id": eligible[0]["candidate_id"]})
    return leaders


def search_estimate(args):
    validate_args(args)
    coarse, fine = coarse_candidates(args), fine_candidates(args)
    fixed = len({row["id"] for row in coarse + fine})
    parent_limit = len(budgets_for(args)) + args.refine_top
    window_limit = min(math.floor(grid_size(args) / 2),
                       math.floor(args.fine_bins * args.max_coverage) + args.refinement_rounds)
    refinement_upper = args.refinement_rounds * parent_limit * (7 * window_limit + 2 * args.fine_bins)
    finalist_limit = len(budgets_for(args))
    validation_upper = args.validation_repeats * (
        1 + finalist_limit * (1 + args.matched_controls + window_limit)
    )
    return {
        "coarse_candidates": len(coarse), "global_fine_and_random_candidates": len(fine),
        "unique_candidates_before_refinement": fixed,
        "episodes_before_refinement": args.discovery_repeats * (1 + fixed),
        "refinement_candidate_upper_bound": refinement_upper,
        "validation_episode_upper_bound": validation_upper,
        "note": "Refinement and validation counts depend on discovery leaders; every stage is planned automatically.",
    }


def noise_hashes(trace):
    if "query_noise_hashes" in trace:
        return trace["query_noise_hashes"]
    return np.asarray([hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()
                       for value in trace["noise"]], dtype="U64")


def read_trace(directory, record):
    with np.load(directory / record["trace_path"], allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify_trace(args, record, trace, windows, reference=None):
    """Verify actual image sources, fixed query timing, actions and shared noise."""
    delay, _ = validate_args(args)
    steps = record["steps"]
    queries = np.arange(0, steps, args.replan_steps, dtype=np.int32)
    if not np.array_equal(trace["query_steps"], queries):
        raise ValueError("Query times changed between intervention conditions")
    expected = np.asarray([window_delay(int(t), windows, delay) for t in queries], dtype=np.int32)
    if not np.array_equal(trace["query_delay_steps"], expected):
        raise ValueError("Actual queried image ages disagree with the intervention windows")
    sources = np.repeat(queries[:, None], 3, axis=1)
    sources[:, _shift.CAMERA_KEYS.index(args.delay_target)] -= expected
    if not np.array_equal(trace["input_source_steps"], sources):
        raise ValueError("Selected-camera or current unselected-camera/state provenance is incorrect")
    indices = np.searchsorted(queries, np.arange(steps), side="right") - 1
    if (not np.array_equal(trace["action_query_steps"], queries[indices])
            or not np.array_equal(trace["action_delay_steps"], expected[indices])):
        raise ValueError("Executed actions crossed a window boundary with an old plan")
    if reference is not None:
        if (trace["start_sim_state"].shape != reference["start_sim_state"].shape
                or not np.allclose(trace["start_sim_state"], reference["start_sim_state"], rtol=0, atol=1e-9)):
            raise ValueError("Paired post-warmup simulator states differ")
        _, current, baseline = np.intersect1d(queries, reference["query_steps"], return_indices=True)
        if not np.array_equal(noise_hashes(trace)[current], noise_hashes(reference)[baseline]):
            raise ValueError("Paired queries did not use identical explicit initial noise")


def compact_trace(path, trace):
    keys = ("steps", "time_s", "eef_pos", "eef_quat_xyzw", "gripper_qpos", "executed_actions",
            "query_steps", "query_delay_steps", "input_source_steps", "action_query_steps",
            "action_delay_steps", "start_sim_state")
    arrays = {key: trace[key] for key in keys if key in trace}
    arrays["query_noise_hashes"] = noise_hashes(trace)
    temporary = path.with_suffix(".compact.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def protocol(args):
    return {
        "outcome": "Final LIBERO success/failure only; no trajectory or activation metric ranks or confirms candidates.",
        "time_origin": "First policy action after common warmup; simulation time, not wall clock.",
        "reference": "Each intervention uses its paired clean episode duration, including unsuccessful clean runs; no post-intervention duration is used to enlarge the budget.",
        "coverage": "Union of delayed action intervals <=max_coverage of paired clean duration; endpoints share the fixed global query grid. Any rounding excess is trimmed before execution.",
        "actual_coverage": "Also report delayed executed actions / all executed actions. Any actual-ratio violation prevents the unconditional confirmed label; never remove such episodes from failure statistics.",
        "history": "Selected camera reads this same closed-loop episode's t-delay frame inside windows; current image elsewhere. Other camera and self state always current.",
        "replanning": "Every condition queries at step % replan_steps == 0; no candidate-dependent extra replans. Window boundaries align to this grid.",
        "pairing": "Same initial state, common warmup and explicit PCG64 SeedSequence([seed,repeat_index,step]) noise at matched steps.",
        "splits": f"Discovery repeat_index starts at 0; validation starts at {VALIDATION_OFFSET}. Final windows/control rules lock before any validation outcome.",
        "random_control": "Each validation pair independently draws separated windows with exactly the mapped candidate's lengths/count. Overlap with the candidate is allowed to keep the control unbiased.",
        "statistics": "One-sided exact McNemar; Holm across all locked finalist-versus-clean/random primary comparisons. Paired increase interval is a conservative Wilson-Bonferroni interval, not a bootstrap.",
        "scope": "Coarse masks exhaust the stated grid; finer global scans/random combinations/refinement are not exhaustive over continuous time or all fine masks.",
        "temporal_resolution": "Normalized endpoints are rounded to whole replan_steps; sub-query windows may collapse. With replan_steps>1, a final partial action chunk cannot be separately targeted. Default replan_steps=1 avoids that extra coarsening.",
        "generalization": "Results apply to the requested task, state IDs and delay. Held-out noise seeds alone do not establish generalization to unseen tasks or initial states.",
        "minimum_failure_increase": args.min_failure_increase,
    }


class SearchSession:
    experiment = EXPERIMENT

    def experiment_protocol(self):
        return protocol(self.args)

    def repeat_index_for(self, split, pair_index):
        return pair_index + (VALIDATION_OFFSET if split == "validation" else 0)

    def map_trial_windows(self, spec, reference_steps, repeat_index):
        return trial_windows(self.args, spec, self.registry, reference_steps, repeat_index)

    def __init__(self, args, directory, client, task, initial_states, metadata, episode_runner=None,
                 worker_client_factory=None):
        self.args, self.directory, self.client, self.task = args, pathlib.Path(directory), client, task
        self.initial_states, self.metadata = initial_states, metadata
        self.episode_runner = episode_runner or _trajectory.run_episode
        self.worker_client_factory = worker_client_factory or _workers.create_client
        self._pool = None
        self.delay_steps, self.max_steps = validate_args(args)
        self.server_endpoints = _workers.parse_servers(args.servers, args.host, args.port)
        self.noise_shape = _trajectory.noise_shape_from_metadata(metadata)
        if args.replan_steps > self.noise_shape[0] or not callable(getattr(client, "infer_with_noise", None)):
            raise ValueError("Explicit-noise client required; replan_steps cannot exceed action horizon")
        if max(state_ids(args)) >= len(initial_states):
            raise ValueError("initial state ID out of range")
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        config = dataclasses.asdict(args)
        for key in ("resume", "dry_run", "analyze_only", "output_dir", "host", "port", "workers", "servers"):
            config.pop(key, None)
        state_hashes = {str(index): hashlib.sha256(np.asarray(initial_states[index], dtype=np.float64).tobytes()).hexdigest()
                        for index in state_ids(args)}
        implementation = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (pathlib.Path(__file__), pathlib.Path(_trajectory.__file__),
                         pathlib.Path(_shift.__file__), pathlib.Path(_workers.__file__))
        }
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (self.manifest.get("experiment") != self.experiment
                    or self.manifest.get("schema_version") != SCHEMA_VERSION
                    or self.manifest["scientific_config"] != config
                    or self.manifest["server_metadata"] != json.loads(json.dumps(metadata))
                    or self.manifest["initial_state_hashes"] != state_hashes):
                raise ValueError("Resume configuration, server capabilities, or initial states differ from the saved run")
        else:
            self.manifest = {
                "experiment": self.experiment, "schema_version": SCHEMA_VERSION,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "config": dataclasses.asdict(args), "scientific_config": config,
                "server_metadata": metadata, "initial_state_hashes": state_hashes,
                "protocol": self.experiment_protocol(), "status": "created", "estimate": search_estimate(args),
                "client_environment": {"python": sys.version, "numpy": np.__version__},
                "implementation_sha256": implementation,
            }
            atomic_json(manifest_path, self.manifest)
        # Parallelism is an execution setting, not part of the locked scientific protocol.
        for key in ("workers", "host", "port", "servers"):
            self.manifest["config"][key] = getattr(args, key)
        self.manifest.setdefault("execution_history", []).append({
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "workers": args.workers, "host": args.host, "port": args.port,
            "servers": [_workers.server_address(*endpoint) for endpoint in self.server_endpoints],
            "implementation_sha256": implementation,
        })
        candidate_path = self.directory / "candidates.json"
        self.registry = json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else {}
        self.register({"id": "clean", "kind": "clean", "windows": [], "coverage": 0.0, "origins": ["control"]})
        self.records = {}
        for path in sorted(self.directory.glob("episodes/*/*/pair_*/record.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            key = (row["split"], row["candidate_id"], row["pair_index"])
            if key in self.records:
                raise ValueError(f"Duplicate trial {key}")
            self.records[key] = row

    def register(self, spec):
        key = spec["id"]
        if key in self.registry:
            old = self.registry[key]
            if old["kind"] != spec["kind"] or old.get("windows") != spec.get("windows"):
                raise ValueError("Candidate identity collision")
            old["origins"] = sorted(set(old["origins"] + spec["origins"]))
        else:
            self.registry[key] = spec

    def checkpoint(self, phase, status="running"):
        self.manifest.update(status=status, phase=phase)
        atomic_json(self.directory / "manifest.json", self.manifest)
        atomic_json(self.directory / "candidates.json", self.registry)

    def freeze_stage(self, name, builder):
        path = self.directory / "stage_plans" / f"{name}.json"
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
        else:
            plan = {"name": name, "candidates": unique_candidates(builder())}
            atomic_json(path, plan)
        for spec in plan["candidates"]:
            self.register(spec)
        self.checkpoint(name)
        return [spec["id"] for spec in plan["candidates"]]

    def completed_trial(self, split, candidate_id, pair_index):
        key = (split, candidate_id, pair_index)
        old = self.records.get(key)
        if old and old["outcome"] in ("success", "failure"):
            if not (self.directory / old["trace_path"]).exists():
                raise ValueError(f"Completed trial lost its audit trace: {old['trace_path']}")
            return old
        return None

    def prepare_trial(self, split, candidate_id, pair_index):
        """Coordinator freezes the exact episode inputs before dispatch."""
        key = (split, candidate_id, pair_index)
        old = self.records.get(key)
        args, spec = self.args, self.registry[candidate_id]
        repeat_index = self.repeat_index_for(split, pair_index)
        initial_id = state_ids(args)[pair_index % len(state_ids(args))]
        episode_args = dataclasses.replace(args, initial_state_id=initial_id)
        reference = None if candidate_id == "clean" else self.records[(split, "clean", pair_index)]
        if reference is not None and reference["outcome"] not in ("success", "failure"):
            raise ValueError("Intervention requires a completed paired clean reference")
        reference_steps = reference["steps"] if reference else None
        windows = self.map_trial_windows(spec, reference_steps, repeat_index) if reference else []
        directory = self.directory / "episodes" / split / candidate_id / f"pair_{pair_index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        if old:
            with (directory / "previous_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(old, allow_nan=False) + "\n")
        prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else str(self.task.language)
        target = _shift.BASELINE_TARGET if candidate_id == "clean" else args.delay_target
        episode_kwargs = dict(
            args=episode_args, task=self.task,
            initial_state=self.initial_states[initial_id], prompt=prompt,
            setting=_trajectory.Setting(candidate_id, target, 0, 0), repeat_index=repeat_index,
            noise_shape=self.noise_shape, warmup_steps=max(args.num_steps_wait, self.delay_steps),
            history_steps=self.delay_steps, max_steps=self.max_steps, output_dir=directory,
        )
        return {
            "key": key, "directory": directory, "reference": reference,
            "reference_steps": reference_steps, "windows": windows, "spec": spec,
            "job": {"episode_kwargs": episode_kwargs, "windows": windows, "delay_steps": self.delay_steps},
        }

    def finish_trial(self, prepared, record):
        """Only the coordinator audits results and commits shared metadata."""
        args = self.args
        key = prepared["key"]
        split, candidate_id, pair_index = key
        directory, reference, spec = prepared["directory"], prepared["reference"], prepared["spec"]
        windows, reference_steps = prepared["windows"], prepared["reference_steps"]
        record["trace_path"] = (directory / record["trace_path"]).relative_to(self.directory).as_posix()
        if record.get("video_path"):
            record["video_path"] = (directory / record["video_path"]).relative_to(self.directory).as_posix()
        record.update(split=split, candidate_id=candidate_id, pair_index=pair_index,
                      reference_steps=reference_steps if reference else record["steps"],
                      windows_steps=windows,
                      windows_seconds=[[a / args.control_hz, b / args.control_hz] for a, b in windows],
                      nominal_coverage=spec.get("coverage"),
                      requested_window_count=len(spec.get("windows", windows)))
        if record["outcome"] in ("success", "failure"):
            try:
                trace = read_trace(self.directory, record)
                ref_trace = read_trace(self.directory, reference) if reference else None
                verify_trace(args, record, trace, windows, ref_trace)
                delayed = int(np.count_nonzero(trace["action_delay_steps"]))
                record.update(
                    planned_delay_steps=window_length(windows), actual_delay_steps=delayed,
                    planned_coverage=window_length(windows) / record["reference_steps"],
                    actual_coverage=delayed / record["steps"] if record["steps"] else 0.0,
                    actual_budget_valid=delayed <= args.max_coverage * record["steps"] + 1e-9,
                    window_reached=bool(np.count_nonzero(trace["query_delay_steps"])),
                    effective_window_count=len(windows),
                )
                if record["planned_coverage"] > args.max_coverage + 1e-9:
                    raise ValueError("Planned intervention exceeds the paired reference coverage budget")
                if ref_trace is not None:
                    n = min(len(trace["eef_pos"]), len(ref_trace["eef_pos"]))
                    distance = np.linalg.norm(trace["eef_pos"][:n] - ref_trace["eef_pos"][:n], axis=1)
                    record.update(eef_mean_separation_m=float(distance.mean()),
                                  eef_max_separation_m=float(distance.max()))
                if not args.keep_full_traces:
                    compact_trace(self.directory / record["trace_path"], trace)
                record["trace_format"] = "full" if args.keep_full_traces else "critical_compact_v1"
            except Exception as error:
                record.update(outcome="error", success=False, error=f"Audit verification failed: {error!r}")
        atomic_json(directory / "record.json", record)
        self.records[key] = record
        atomic_json(self.directory / "progress.json", {
            "phase": self.manifest["phase"], "last_trial": list(key), "last_outcome": record["outcome"],
            "completed_trials": sum(row["outcome"] in ("success", "failure") for row in self.records.values()),
        })
        if record["outcome"] == "interrupted":
            raise KeyboardInterrupt()
        if record["outcome"] == "error":
            raise RuntimeError(f"Trial stopped on infrastructure/audit error, not a task failure: {record['error']}. "
                               f"Fix the cause then use --resume {self.directory}")
        return record

    def execute_trial(self, split, candidate_id, pair_index):
        completed = self.completed_trial(split, candidate_id, pair_index)
        if completed is not None:
            return completed
        prepared = self.prepare_trial(split, candidate_id, pair_index)
        record = _workers.run_rollout_job(prepared["job"], self.client, self.episode_runner)
        record["policy_server"] = _workers.server_address(*self.server_endpoints[0])
        return self.finish_trial(prepared, record)

    def close_workers(self, *, terminate=False):
        if self._pool is not None:
            self._pool.close(terminate=terminate)
            self._pool = None

    def run_stage(self, name, candidates, split):
        self.checkpoint(name)
        repeats = self.args.discovery_repeats if split == "discovery" else self.args.validation_repeats
        phase_salt = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        total, done = len(candidates) * repeats, 0
        logging.info("Stage %s: %d candidates x %d pairs = %d trials (completed trials are reused)",
                     name, len(candidates), repeats, total)
        # Duplicates must never be dispatched simultaneously to the same directory.
        candidates = list(dict.fromkeys(candidates))
        total = len(candidates) * repeats

        def trial_keys():
            for pair in range(repeats):
                rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, phase_salt, pair]))
                for index in rng.permutation(len(candidates)):
                    yield split, candidates[int(index)], pair

        def log_result(record):
            nonlocal done
            done += 1
            logging.info("[%s %d/%d workers=%d] %s pair=%d %s actions=%d coverage=%.1f%%",
                         name, done, total, self.args.workers, record["candidate_id"], record["pair_index"],
                         record["outcome"], record["steps"], 100 * record.get("actual_coverage", 0.0))

        if self.args.workers == 1:
            for key in trial_keys():
                log_result(self.execute_trial(*key))
        else:
            prepared_trials = {}

            def pending_jobs():
                for key in trial_keys():
                    completed = self.completed_trial(*key)
                    if completed is not None:
                        log_result(completed)
                        continue
                    prepared = self.prepare_trial(*key)
                    prepared_trials[key] = prepared
                    yield key, prepared["job"]

            # Do not start processes just to revisit a stage already completed on resume.
            jobs = iter(pending_jobs())
            first = next(jobs, None)
            if first is not None:
                if self._pool is None:
                    self._pool = _workers.RolloutPool(
                        self.args.workers, self.args.host, self.args.port, self.metadata,
                        self.episode_runner, self.worker_client_factory,
                        server_endpoints=self.server_endpoints,
                    )
                    self.manifest["execution_history"][-1]["worker_pids"] = self._pool.pids
                    self.manifest["execution_history"][-1]["worker_assignments"] = self._pool.assignments
                    logging.info("Started simulation workers: %s", json.dumps(self._pool.assignments))
                    self.checkpoint(name)
                try:
                    for key, record in self._pool.iter_results(itertools.chain([first], jobs)):
                        log_result(self.finish_trial(prepared_trials.pop(key), record))
                except BaseException as error:
                    with (self.directory / "parallel_events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({
                            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "phase": name, "error": repr(error),
                            "unfinished_trials": [list(key) for key in prepared_trials],
                        }) + "\n")
                    self.close_workers(terminate=True)
                    raise
        rows = discovery_ranking(self.args, self.registry, list(self.records.values()))
        write_csv(self.directory / "discovery_ranking.csv", rows)

    def validation_plan(self):
        path = self.directory / "validation_plan.json"
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
        else:
            ranking = discovery_ranking(self.args, self.registry, list(self.records.values()))
            leaders = budget_leaders(self.args, ranking)
            if not leaders:
                raise RuntimeError("No complete discovery candidates; cannot create validation plan")
            finalists = list(dict.fromkeys(row["candidate_id"] for row in leaders))
            specs, controls, ablations, comparisons = [], [], [], []
            for key in finalists:
                spec = self.registry[key]
                specs.append(spec)
                comparisons.append({"id": key + ":clean", "candidate_id": key,
                                    "reference_id": "clean", "purpose": "vs_clean"})
                for index in range(self.args.matched_controls):
                    control = {"id": f"random_{key}_{index:02d}", "kind": "matched_random",
                               "source_id": key, "control_index": index, "windows": [],
                               "coverage": spec["coverage"], "origins": ["held_out_random_control"]}
                    specs.append(control)
                    controls.append({"candidate_id": key, "control_id": control["id"]})
                    comparisons.append({"id": key + ":" + control["id"], "candidate_id": key,
                                        "reference_id": control["id"], "purpose": "vs_random"})
                for index in range(len(spec["windows"])):
                    child = make_candidate(self.args, spec["windows"][:index] + spec["windows"][index + 1:],
                                           "held_out_ablation:" + key)
                    if child:
                        specs.append(child)
                    ablations.append({"candidate_id": key, "removed_window_index": index,
                                      "ablation_id": child["id"] if child else "clean"})
            plan = {
                "locked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "selection": "Discovery failure-rate increase only; ties prefer less coverage/fewer windows.",
                "budget_leaders": leaders, "finalists": finalists,
                "controls": controls, "ablations": ablations, "primary_comparisons": comparisons,
                "candidates": unique_candidates(specs),
                "validation_repeat_start": VALIDATION_OFFSET,
                "rule": "Do not change candidates after observing validation outcomes; actual-budget violations remain in outcomes and prevent confirmation.",
            }
            atomic_json(path, plan)
        for spec in plan["candidates"]:
            self.register(spec)
        self.checkpoint("validation_locked")
        return plan

    def execute(self):
        try:
            self.run_stage("00_discovery_clean", ["clean"], "discovery")
            coarse = self.freeze_stage("01_coarse", lambda: coarse_candidates(self.args))
            self.run_stage("01_coarse", coarse, "discovery")
            fine = self.freeze_stage("02_global_fine", lambda: fine_candidates(self.args))
            self.run_stage("02_global_fine", fine, "discovery")
            for index in range(self.args.refinement_rounds):
                name = f"03_refinement_{index:02d}"

                def build_refinement():
                    ranking = discovery_ranking(self.args, self.registry, list(self.records.values()))
                    ids = [row["candidate_id"] for row in budget_leaders(self.args, ranking)]
                    ids.extend(row["candidate_id"] for row in ranking[:self.args.refine_top])
                    return refine_candidates(self.args, [self.registry[key] for key in dict.fromkeys(ids)], index)

                candidates = self.freeze_stage(name, build_refinement)
                self.run_stage(name, candidates, "discovery")
            plan = self.validation_plan()
            # These clean outcomes are collected only AFTER the finalists/control rule lock.
            self.run_stage("04_validation_clean", ["clean"], "validation")
            self.run_stage("05_validation", [spec["id"] for spec in plan["candidates"]], "validation")
            self.checkpoint("complete", "complete")
        except KeyboardInterrupt:
            self.checkpoint(self.manifest.get("phase", "unknown"), "interrupted")
            raise
        except Exception:
            self.checkpoint(self.manifest.get("phase", "unknown"), "error")
            raise
        finally:
            self.close_workers(terminate=self.manifest["status"] != "complete")
        return self.directory


def restore_execution_args(args, saved_config, directory):
    """Restore scientific args, allowing only explicit execution overrides."""
    defaults = Args()
    explicit = getattr(args, "_connection_overrides", set())
    updates = {key: getattr(args, key) for key in ("host", "port", "workers", "servers")
               if key in explicit or getattr(args, key) != getattr(defaults, key)}
    if {"host", "port"} & updates.keys() and "servers" not in updates:
        # Explicit legacy endpoint switches a resumed multi-server run back to one server.
        updates["servers"] = ""
    return dataclasses.replace(Args(**saved_config), resume=str(directory), dry_run=False, analyze_only=None, **updates)


def run(args):
    if args.analyze_only:
        return analyze_results(pathlib.Path(args.analyze_only))
    if args.resume:
        directory = pathlib.Path(args.resume)
        if directory.is_file():
            directory = directory.parent
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        args = restore_execution_args(args, saved["config"], directory)
        if saved["status"] == "complete":
            return analyze_results(directory)
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        directory = pathlib.Path(args.output_dir) / stamp
    validate_args(args)
    estimate = search_estimate(args)
    logging.info("Search estimate: %s", json.dumps(estimate))
    logging.info("Simulation workers: %d (each worker owns one websocket client)", args.workers)
    endpoints = _workers.parse_servers(args.servers, args.host, args.port)
    addresses = [_workers.server_address(*endpoint) for endpoint in endpoints]
    logging.info("Policy servers: %s", ", ".join(addresses))
    if len(endpoints) > 1:
        logging.info("Workers are pinned round-robin to %d independent policy servers; use the same policy/checkpoint "
                     "on every server. Metadata equality cannot verify identical weights.", len(endpoints))
    elif args.workers > 1:
        logging.info("The current policy server serializes model inference; parallelism overlaps simulator/rendering "
                     "with inference but does not batch requests or replicate the model")
    if args.dry_run:
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "search_plan.json"
        atomic_json(path, {"experiment": EXPERIMENT, "config": dataclasses.asdict(args),
                           "worker_servers": [addresses[index % len(addresses)] for index in range(args.workers)],
                           "estimate": estimate, "protocol": protocol(args)})
        print(json.dumps(estimate, indent=2))
        return path
    # Fail before expensive rollouts if plotting dependencies are unavailable.
    import matplotlib

    matplotlib.use("Agg")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError("task_id out of range")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    client = _workers.create_client(*endpoints[0])
    try:
        session = SearchSession(args, directory, client, task, states, client.get_server_metadata())
        session.execute()
    finally:
        client._ws.close()
    return analyze_results(directory)


def analyze_results(directory):
    try:
        from examples.libero import analyze_critical_window_search as analysis
    except ImportError:
        import analyze_critical_window_search as analysis
    return analysis.analyze(directory)


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
        parser.error("Use either --servers or --host/--port, not both")
    if "resume" in values:
        conflicting = set(values) - {"resume", "host", "port", "workers", "servers"}
        if conflicting:
            parser.error("--resume restores the saved scientific configuration; only --host/--port/--servers/--workers may be overridden")
    args = Args(**values)
    args._connection_overrides = set(values) & {"host", "port", "workers", "servers"}
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print(run(parse_args()))
    except KeyboardInterrupt:
        logging.warning("Interrupted; use --resume <saved-run-directory> to continue the same frozen search")
