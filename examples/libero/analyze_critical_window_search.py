"""Rebuild critical-window reports and figures without running new episodes."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

try:
    from examples.libero import run_critical_window_search as _run
except ImportError:
    import run_critical_window_search as _run


def load_experiment(path):
    directory = pathlib.Path(path)
    if directory.is_file():
        directory = directory.parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != _run.EXPERIMENT or manifest.get("schema_version") != _run.SCHEMA_VERSION:
        raise ValueError("Requires a critical-window-search manifest")
    args = _run.Args(**manifest["config"])
    _run.validate_args(args)
    registry = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    records, seen = [], set()
    for path in sorted(directory.glob("episodes/*/*/pair_*/record.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["split"], row["candidate_id"], row["pair_index"])
        if key in seen:
            raise ValueError(f"Duplicate episode identity {key}")
        if row["candidate_id"] not in registry:
            raise ValueError("Episode refers to an unregistered candidate")
        expected_repeat = row["pair_index"] + (_run.VALIDATION_OFFSET if row["split"] == "validation" else 0)
        if row["repeat_index"] != expected_repeat:
            raise ValueError("Discovery/validation repeat indices are not independent")
        if row["outcome"] in ("success", "failure") and bool(row["success"]) != (row["outcome"] == "success"):
            raise ValueError("Episode success flag disagrees with final outcome")
        if row.get("planned_coverage", 0) > args.max_coverage + 1e-9:
            raise ValueError("A candidate exceeds the reference-duration coverage budget")
        seen.add(key)
        records.append(row)
    plan_path = directory / "validation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    return directory, manifest, args, registry, records, plan


def validation_tables(args, registry, records, plan, run_complete=True):
    grouped = collections.defaultdict(list)
    for row in records:
        if row["split"] == "validation":
            grouped[row["candidate_id"]].append(row)
    comparisons = []
    for spec in plan["primary_comparisons"]:
        stats = _run.paired_statistics(grouped[spec["reference_id"]], grouped[spec["candidate_id"]],
                                       args.validation_repeats)
        comparisons.append({**spec, **stats})
    adjusted = _run.holm_adjust({row["id"]: row["p_one_sided"] if row["p_one_sided"] is not None else 1.0
                                for row in comparisons})
    for row in comparisons:
        row["p_holm"] = adjusted.get(row["id"])
        row["significant"] = bool(row["complete"] and row["p_holm"] is not None
                                  and row["p_holm"] <= args.alpha and row["increase_ci_low"] > 0)
    lookup = {row["id"]: row for row in comparisons}
    finalists = []
    for key in plan["finalists"]:
        primary = lookup[key + ":clean"]
        random_comparisons = [row for row in comparisons if row["candidate_id"] == key and row["purpose"] == "vs_random"]
        effect = bool(primary["significant"] and primary["failure_increase"] >= args.min_failure_increase)
        specific = bool(random_comparisons and all(row["significant"] for row in random_comparisons))
        budget_valid = primary["actual_budget_violations"] == 0 and primary["complete"]
        confirmed = effect and specific and budget_valid and run_complete
        reasons = []
        if not run_complete:
            reasons.append("search/validation run is not complete")
        if not primary["complete"]:
            reasons.append("incomplete held-out pairs")
        if not effect:
            reasons.append("failure increase does not meet the locked effect/statistical criteria")
        if not specific:
            reasons.append("not established as more harmful than all matched randomized controls")
        if not budget_valid:
            reasons.append("actual episode-duration coverage budget not established for all pairs")
        finalists.append({
            "candidate_id": key, "windows": registry[key]["windows"], "coverage": registry[key]["coverage"],
            **primary, "confirmed": confirmed, "failure_sensitive": effect, "time_specific": specific,
            "actual_budget_valid": budget_valid, "reason": "; ".join(reasons) if reasons else "all locked criteria met",
            "median_reference_duration_s": float(np.median([row["reference_steps"] / args.control_hz
                                                             for row in grouped[key] if "reference_steps" in row]))
            if any("reference_steps" in row for row in grouped[key]) else None,
            "example_windows_seconds": next((row["windows_seconds"] for row in sorted(grouped[key], key=lambda r: r["pair_index"])
                                              if "windows_seconds" in row), []),
        })
    ablations = []
    for spec in plan["ablations"]:
        # Descriptive only: these comparisons are not additional primary claims.
        stats = _run.paired_statistics(grouped[spec["ablation_id"]], grouped[spec["candidate_id"]],
                                       args.validation_repeats)
        ablations.append({**spec, **stats, "inference": "descriptive leave-one-window-out comparison"})
    condition_rows = []
    for key in ["clean", *[spec["id"] for spec in plan["candidates"]]]:
        counts = collections.Counter(row["outcome"] for row in grouped[key])
        stats = _run.paired_statistics(grouped["clean"], grouped[key], args.validation_repeats)
        condition_rows.append({"candidate_id": key, "kind": registry[key]["kind"],
                               "coverage": registry[key]["coverage"], "errors": counts["error"],
                               "interrupted": counts["interrupted"], **stats})
    return finalists, comparisons, ablations, condition_rows


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def sensitivity_plot(args, ranking, directory):
    rows = [row for row in ranking if len(row["windows"]) == 1]
    if not rows:
        return
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(11, 6))
    size = _run.grid_size(args)
    starts = [row["windows"][0][0] / size * 100 for row in rows]
    lengths = [row["coverage"] * 100 for row in rows]
    effects = [row["failure_increase"] * 100 for row in rows]
    image = axis.scatter(starts, lengths, c=effects, cmap="RdBu_r", vmin=-100, vmax=100,
                         marker="s", s=70, edgecolors="0.8", linewidths=0.25)
    fig.colorbar(image, ax=axis, label="Final task failure-rate increase (percentage points)")
    axis.set(xlabel="Window start (% of paired clean duration)", ylabel="Window length (% of paired clean duration)",
             xlim=(-2, 102), ylim=(0, args.max_coverage * 100 + 3),
             title="Discovery single-window sensitivity | each square is a measured condition")
    axis.set_facecolor("#eeeeee")
    axis.grid(alpha=0.15)
    fig.text(0.5, 0.02, "Exploratory data used for search; unmeasured combinations are blank. "
             "Final claims use independent validation outcomes.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(directory / "single_window_sensitivity.png", dpi=160)
    plt.close(fig)


def window_comparison_plot(args, registry, plan, rows, finalists, directory):
    if not rows:
        return
    plt = _pyplot()
    finalist_lookup = {row["candidate_id"]: row for row in finalists}
    final_ids = set(plan["finalists"])
    random_ids = {row["control_id"] for row in plan["controls"]}
    # Keep primary comparisons readable; all ablation results remain in CSV/JSON.
    displayed = [row for row in rows if row["candidate_id"] in final_ids | random_ids | {"clean"}]
    height = max(4, 0.55 * len(displayed) + 1.7)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, height), sharey=True,
                                    gridspec_kw={"width_ratios": [2.1, 1]})
    size, labels = _run.grid_size(args), []
    for y, row in enumerate(displayed):
        key, spec = row["candidate_id"], registry[row["candidate_id"]]
        color = "#c85845" if key in final_ids else "#427d9e" if key in random_ids else "0.4"
        if key == "clean":
            label = "Clean baseline"
            left.text(50, y, "Always current", ha="center", va="center", fontsize=9, color="0.4")
        elif key in random_ids:
            source = plan["finalists"].index(spec["source_id"]) + 1
            label = f"F{source} random {spec['control_index'] + 1}"
            left.text(50, y, "Shape matched; placement randomized per pair", ha="center", va="center",
                      fontsize=8, color=color)
        else:
            index = plan["finalists"].index(key) + 1
            status = "confirmed" if finalist_lookup[key]["confirmed"] else "not confirmed"
            label = f"F{index} ({status})"
            for a, b in spec["windows"]:
                left.broken_barh([(a / size * 100, (b - a) / size * 100)], (y - 0.3, 0.6),
                                 facecolors=color)
        labels.append(label)
        if row["failure_rate"] is not None:
            rate = row["failure_rate"] * 100
            right.errorbar(rate, y,
                           xerr=[[max(0, rate - row["failure_ci_low"] * 100)],
                                 [max(0, row["failure_ci_high"] * 100 - rate)]],
                           fmt="o", color=color, capsize=3)
            right.text(103, y, f"{row['failures']}/{row['completed_pairs']}", va="center", fontsize=8)
        else:
            right.text(10, y, "No completed pairs", va="center", fontsize=8)
    left.set_yticks(range(len(displayed)))
    left.set_yticklabels(labels, fontsize=9)
    left.set(xlim=(0, 100), xlabel="Time (% of paired clean reference duration)",
             ylim=(len(displayed) - 0.4, -0.7))
    right.set(xlim=(-3, 122), xlabel="Final task failure rate (%)")
    right.set_xticks([0, 25, 50, 75, 100])
    for axis in (left, right):
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Locked held-out validation | window locations and final failure rates", fontsize=13)
    fig.text(0.5, 0.02, "Intervals: 95% Wilson. Finalist labels require failure evidence, matched random comparisons "
             f"and <={args.max_coverage:.0%} reference/actual coverage.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(directory / "validated_windows.png", dpi=160)
    plt.close(fig)


def coverage_plot(plan, rows, directory):
    lookup = {row["candidate_id"]: row for row in rows}
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(10, 6))
    for leader in plan["budget_leaders"]:
        row = lookup[leader["candidate_id"]]
        if row["failure_rate"] is None:
            continue
        x, rate = leader["budget"] * 100, row["failure_rate"] * 100
        axis.errorbar(x, rate, yerr=[[max(0, rate - row["failure_ci_low"] * 100)],
                                     [max(0, row["failure_ci_high"] * 100 - rate)]],
                      fmt="o", color="#c85845", capsize=4)
        controls = [lookup[value["control_id"]]["failure_rate"] for value in plan["controls"]
                    if value["candidate_id"] == leader["candidate_id"]]
        controls = [value for value in controls if value is not None]
        if controls:
            axis.scatter([x], [100 * np.mean(controls)], marker="s", color="#427d9e")
        axis.annotate(f"{row['coverage']:.0%} selected", (x, rate), xytext=(0, 8),
                      ha="center", textcoords="offset points", fontsize=8)
    baseline = lookup["clean"]["failure_rate"]
    if baseline is not None:
        axis.axhline(baseline * 100, color="0.4", linestyle="--", label="Clean baseline")
    axis.plot([], [], "o", color="#c85845", label="Discovery-selected windows (held-out rate)")
    axis.plot([], [], "s", color="#427d9e", label="Shape-matched random control mean")
    axis.set(xlabel="Allowed intervention coverage (% of paired clean duration)", ylabel="Final task failure rate (%)",
             ylim=(-4, 111), title="Coverage budget versus final failure | candidates fixed before validation")
    budgets = [row["budget"] * 100 for row in plan["budget_leaders"]]
    if budgets:
        axis.set_xlim(min(budgets) - 5, max(budgets) + 5)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "coverage_vs_failure.png", dpi=160)
    plt.close(fig)


def analyze(path):
    directory, manifest, args, registry, records, plan = load_experiment(path)
    target = directory / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    ranking = _run.discovery_ranking(args, registry, records)
    _run.write_csv(target / "discovery_ranking.csv", ranking)
    _run.write_csv(target / "episodes.csv", records)
    sensitivity_plot(args, ranking, target)
    finalists, comparisons, ablations, conditions = [], [], [], []
    if plan:
        finalists, comparisons, ablations, conditions = validation_tables(
            args, registry, records, plan, run_complete=manifest["status"] == "complete",
        )
        _run.write_csv(target / "validation_finalists.csv", finalists)
        _run.write_csv(target / "validation_comparisons.csv", comparisons)
        _run.write_csv(target / "validation_ablations.csv", ablations)
        _run.write_csv(target / "validation_failure_rates.csv", conditions)
        window_comparison_plot(args, registry, plan, conditions, finalists, target)
        coverage_plot(plan, conditions, target)
    previous_attempts = []
    for path in directory.glob("episodes/*/*/pair_*/previous_attempts.jsonl"):
        previous_attempts.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    report = {
        "experiment": _run.EXPERIMENT, "schema_version": _run.SCHEMA_VERSION,
        "status": manifest["status"], "task": args.task_suite_name, "task_id": args.task_id,
        "delay_target": args.delay_target, "delay_ms": args.delay_ms,
        "grid_size": _run.grid_size(args), "initial_state_ids": _run.state_ids(args),
        "protocol": manifest["protocol"], "discovery_candidates_evaluated": len(ranking),
        "validation_plan": plan, "validation_finalists": finalists,
        "primary_comparisons": comparisons, "ablations": ablations, "validation_conditions": conditions,
        "confirmed_candidates": [row["candidate_id"] for row in finalists if row["confirmed"]],
        "runtime_errors": sum(row["outcome"] == "error" for row in records),
        "runtime_error_attempts": sum(row["outcome"] == "error" for row in records + previous_attempts),
        "previous_attempts": len(previous_attempts),
        "interruptions": sum(row["outcome"] == "interrupted" for row in records),
        "scope": [
            "Confirmation uses held-out final task outcomes only. Trajectory separation does not define importance.",
            "Intervals are fractions of each paired clean duration, not manually assigned task stages.",
            "All complete pairs are included, including early success, clean failure and actual-coverage violations.",
            "No confirmation if the actual executed intervention exceeds the coverage cap in any validation pair.",
            "Null results mean not established under this delay, sample size and search resolution; they do not prove no critical windows exist.",
        ],
    }
    _run.atomic_json(target / "report.json", report)
    lines = ["关键时间窗口搜索结果", f"任务: {args.task_suite_name}/{args.task_id}; 相机: {args.delay_target}; 延迟: {args.delay_ms:g} ms",
             f"状态: {manifest['status']}; 已评估发现候选: {len(ranking)}; 验证回合/条件: {args.validation_repeats}",
             "时间窗口以每个配对 clean 回合时长的百分比表示。运行错误不计为任务失败。", ""]
    for index, row in enumerate(finalists, 1):
        intervals = ", ".join(f"[{100 * a / _run.grid_size(args):.1f}%, {100 * b / _run.grid_size(args):.1f}%)"
                              for a, b in row["windows"])
        rate, increase = row["failure_rate"], row["failure_increase"]
        value = f"{rate:.1%}" if rate is not None else "N/A"
        delta = f"{100 * increase:.1f} 个百分点" if increase is not None else "N/A"
        status = "已通过独立验证" if row["confirmed"] else "尚未满足全部验证标准"
        lines.extend([
            f"F{index} {row['candidate_id']}: {intervals}; 名义覆盖 {row['coverage']:.1%}",
            f"  验证失败率 {value} ({row['failures']}/{row['completed_pairs']}); 相对 clean 增加 {delta}",
            f"  {status}; actual {args.max_coverage:.0%} 约束超限回合: {row['actual_budget_violations']}",
            f"  验证配对首个回合实际时间窗（秒）: {row['example_windows_seconds']}",
            f"  判定说明: {row['reason']}",
        ])
    if not finalists:
        lines.append("尚未完成候选锁定/独立验证；这些发现数据不能作为最终关键窗口结论。")
    if finalists and not any(row["confirmed"] for row in finalists):
        lines.append("未找到满足全部预设标准的窗口，不能将发现阶段的最高失败率直接当作已验证结论。")
    text = "\n".join(lines) + "\n"
    (directory / "results.txt").write_text(text, encoding="utf-8")
    print(text)
    return target / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, help="Search directory or manifest.json")
    print(analyze(**vars(parser.parse_args())))
