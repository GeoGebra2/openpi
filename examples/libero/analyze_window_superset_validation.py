"""Rebuild fixed-superset validation tables and figures without LIBERO or a server."""
# ruff: noqa: PLC0415, RUF001

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from statistics import NormalDist

import numpy as np

try:
    from examples.libero import run_critical_window_search as _search
    from examples.libero import run_window_superset_validation as _run
except ImportError:
    import run_critical_window_search as _search
    import run_window_superset_validation as _run


def comparison_tables(plan, records, run_complete):
    groups = collections.defaultdict(list)
    for row in records:
        groups[row["candidate_id"]].append(row)
    core = _search.Args(**plan["core_config"])
    base_id, expected = plan["base"]["id"], plan["repeats"]
    conditions = []
    for spec in plan["conditions"]:
        rows = groups[spec["id"]]
        stats = _search.paired_statistics(groups["clean"], rows, expected)
        conditions.append(
            {
                "candidate_id": spec["id"],
                "label": spec.get("label", "CLEAN" if spec["id"] == "clean" else "BASE"),
                "windows": spec["windows"],
                "nominal_coverage": spec["coverage"],
                **stats,
                "errors": sum(r["outcome"] == "error" for r in rows),
                "interrupted": sum(r["outcome"] == "interrupted" for r in rows),
                "mapping_repairs": sum(r.get("mapping_repaired", False) for r in rows),
                "mapped_equal_to_base": sum(r.get("effective_equal_to_base", False) for r in rows),
            }
        )
    comparisons = []
    for spec in plan["supersets"]:
        # reference=SUPERSET, intervention=BASE: one-sided test of base > superset.
        stats = _search.paired_statistics(groups[spec["id"]], groups[base_id], expected)
        sup = {r["pair_index"]: r for r in groups[spec["id"]] if r["outcome"] in ("success", "failure")}
        base = {r["pair_index"]: r for r in groups[base_id] if r["outcome"] in ("success", "failure")}
        inclusion_errors, budget_violations, mapping_changes, equal = 0, 0, 0, 0
        for pair in sorted(base.keys() & sup.keys()):
            b, s = base[pair], sup[pair]
            if b["reference_steps"] != s["reference_steps"]:
                inclusion_errors += 1
                continue
            length = b["reference_steps"]
            protected = _run.window_mask(b["windows_steps"], length)
            mask = _run.window_mask(s["windows_steps"], length)
            expected_base = _search.mapped_windows(core, plan["base"], length)
            expected_super = _run.nested_windows(core, spec, plan["base"], length)
            if (
                not np.all(mask[protected])
                or b["windows_steps"] != expected_base
                or s["windows_steps"] != expected_super
            ):
                inclusion_errors += 1
            budget_violations += int(not b.get("actual_budget_valid", False) or not s.get("actual_budget_valid", False))
            equal += int(np.array_equal(mask, protected))
            mapping_changes += int(s["windows_steps"] != _search.mapped_windows(core, spec, length))
        n = stats["completed_pairs"]
        a, b = stats["clean_success_to_failure"], stats["clean_failure_to_success"]
        # Bonferroni bounds for the two discordant probabilities give a conservative
        # paired-difference interval. These are pointwise, not simultaneous across S01..Sn.
        z = NormalDist().inv_cdf(1 - plan["alpha"] / 4)
        al, ah = _search.wilson(a, n, z)
        bl, bh = _search.wilson(b, n, z)
        comparisons.append(
            {
                "candidate_id": spec["id"],
                "label": spec["label"],
                "windows": spec["windows"],
                "nominal_coverage": spec["coverage"],
                "completed_pairs": n,
                "complete": stats["complete"],
                "base_failures": stats["failures"],
                "superset_failures": stats["reference_failures"],
                "base_failure_rate": stats["failure_rate"],
                "superset_failure_rate": stats["reference_failure_rate"],
                "base_minus_superset": stats["failure_increase"],
                "difference_ci_low": max(-1.0, al - bh) if n else None,
                "difference_ci_high": min(1.0, ah - bl) if n else None,
                "base_fails_superset_succeeds": a,
                "base_succeeds_superset_fails": b,
                "p_one_sided": stats["p_one_sided"],
                "inclusion_errors": inclusion_errors,
                "budget_violation_pairs": budget_violations,
                "mapping_repair_pairs": mapping_changes,
                "mapped_equal_pairs": equal,
            }
        )
    adjusted = _search.holm_adjust(
        {r["candidate_id"]: r["p_one_sided"] if r["p_one_sided"] is not None else 1.0 for r in comparisons}
    )
    for row in comparisons:
        row["p_holm"] = adjusted[row["candidate_id"]]
        row["numerically_lower"] = bool(row["completed_pairs"] and row["base_minus_superset"] > 0)
        row["statistically_lower"] = bool(
            row["complete"]
            and row["numerically_lower"]
            and row["p_holm"] <= plan["alpha"]
            and row["difference_ci_low"] > 0
            and row["base_minus_superset"] >= plan["min_failure_difference"]
        )
        row["confirmed_lower"] = bool(
            run_complete
            and row["statistically_lower"]
            and row["inclusion_errors"] == 0
            and row["budget_violation_pairs"] == 0
        )
    return conditions, comparisons


def _pyplot():
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def plot_results(plan, conditions, comparisons, directory, status):
    plt = _pyplot()
    size = _search.grid_size(_search.Args(**plan["core_config"]))
    height = max(5, 0.48 * len(conditions) + 1.7)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, height), sharey=True, gridspec_kw={"width_ratios": [2, 1.2]})
    labels = []
    for y, row in enumerate(conditions):
        color = "#c65c43" if row["label"] == "BASE" else "#427d9e" if row["label"] != "CLEAN" else "0.5"
        labels.append(f"{row['label']}  ({row['nominal_coverage']:.0%})")
        for a, b in row["windows"]:
            left.broken_barh([(a / size * 100, (b - a) / size * 100)], (y - 0.3, 0.6), facecolors=color)
        if row["label"] == "CLEAN":
            left.text(50, y, "Always current", ha="center", va="center", color="0.5")
        if row["failure_rate"] is not None:
            rate = row["failure_rate"] * 100
            right.barh(y, rate, height=0.6, color=color, alpha=0.9)
            right.errorbar(
                rate,
                y,
                xerr=[[max(0, rate - row["failure_ci_low"] * 100)], [max(0, row["failure_ci_high"] * 100 - rate)]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
            )
            right.text(102, y, f"{row['failures']}/{row['completed_pairs']}", va="center", fontsize=9)
    left.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlim=(0, 100),
        ylim=(len(labels) - 0.4, -0.7),
        xlabel="Time (% of each paired clean duration)",
    )
    right.set(xlim=(0, 120), xticks=[0, 25, 50, 75, 100], xlabel="Final task failure rate (%)")
    for axis in (left, right):
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Fixed superset validation | {status} | new paired seeds", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Bars use completed clean pairs; intervals: 95% Wilson. "
        "Colors identify conditions; confirmation uses paired comparisons below.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(directory / "windows_failure_rates.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, height))
    labels = []
    for y, row in enumerate(comparisons):
        labels.append(row["label"])
        if row["base_minus_superset"] is None:
            continue
        delta = row["base_minus_superset"] * 100
        color = "#387c59" if row["confirmed_lower"] else "#427d9e"
        axis.errorbar(
            delta,
            y,
            xerr=[[max(0, delta - row["difference_ci_low"] * 100)], [max(0, row["difference_ci_high"] * 100 - delta)]],
            fmt="o",
            color=color,
            capsize=4,
        )
        axis.text(
            1.02,
            y,
            f"{delta:+.1f} pp   Holm p={row['p_holm']:.3g}   "
            f"{'confirmed' if row['confirmed_lower'] else 'not confirmed'}",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
    axis.axvline(0, color="0.5", linewidth=1)
    axis.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        ylim=(len(labels) - 0.4, -0.7),
        xlabel="BASE failure rate minus SUPERSET failure rate (percentage points)",
        title=f"Positive means superset fails less | {status}",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.text(
        0.5,
        0.02,
        f"Pointwise {1 - plan['alpha']:.0%} conservative paired intervals; Holm tests all "
        f"{len(comparisons)} comparisons. Green requires completed run, valid inclusion and coverage.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 0.73, 0.98))
    fig.savefig(directory / "paired_failure_differences.png", dpi=170)
    plt.close(fig)


def analyze(directory):
    directory = pathlib.Path(directory)
    plan = json.loads((directory / _run.PLAN_FILE).read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = []
    seen = set()
    for path in sorted(directory.glob("episodes/validation/*/pair_*/record.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["candidate_id"], row["pair_index"])
        if key in seen:
            raise ValueError(f"Duplicate trial {key}")
        seen.add(key)
        records.append(row)
    status = manifest.get("status", "plan_only")
    conditions, comparisons = comparison_tables(plan, records, status == "complete")
    all_lower = bool(comparisons and all(r["confirmed_lower"] for r in comparisons))
    report = {
        "experiment": _run.EXPERIMENT,
        "status": status,
        "base": plan["base"],
        "conditions": conditions,
        "comparisons": comparisons,
        "all_tested_supersets_confirmed_lower": all_lower,
        "n_confirmed_lower": sum(r["confirmed_lower"] for r in comparisons),
        "runtime_errors": sum(r["outcome"] == "error" for r in records),
        "protocol": {key: plan[key] for key in ("selection", "mapping", "hypothesis", "statistics", "scope")},
    }
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    _search.atomic_json(out / "report.json", report)
    _search.write_csv(out / "failure_rates.csv", conditions)
    _search.write_csv(out / "paired_comparisons.csv", comparisons)
    _search.write_csv(out / "pair_outcomes.csv", records)
    plot_results(plan, conditions, comparisons, out, status)
    core = _search.Args(**plan["core_config"])
    size = _search.grid_size(core)

    def windows_text(windows):
        return " + ".join(f"[{a / size:.1%}, {b / size:.1%})" for a, b in windows)

    lines = [
        "窗口包含关系独立验证",
        f"状态: {status}; 每条件计划 {plan['repeats']} 回合",
        f"任务: {core.task_suite_name}/{core.task_id}; 相机: {core.delay_target}; 延迟: {core.delay_ms:g} ms",
        f"基础双窗: {windows_text(plan['base']['windows'])}",
        "差值 = 基础双窗失败率 - 超集失败率；正值表示超集更少失败。运行错误不计为任务失败。",
        "旧实验的 32% 不用于检验；基础双窗与所有超集均使用本批新配对种子重跑。",
        "",
    ]
    base_row = next(r for r in conditions if r["label"] == "BASE")
    lines.append(f"基础双窗新批次失败: {base_row['failures']}/{base_row['completed_pairs']}")
    for row in comparisons:
        lines.extend(
            [
                f"{row['label']} {windows_text(row['windows'])}; 覆盖 {row['nominal_coverage']:.1%}",
                (
                    f"  配对失败: 基础 {row['base_failures']}/{row['completed_pairs']}, "
                    f"超集 {row['superset_failures']}/{row['completed_pairs']}"
                ),
            ]
        )
        if row["base_minus_superset"] is not None:
            lines.append(
                f"  差值 {row['base_minus_superset'] * 100:+.1f} 个百分点; "
                f"Holm p={row['p_holm']:.5g}; "
                f"{'验证通过：超集失败率更低' if row['confirmed_lower'] else '尚未验证通过'}"
            )
        lines.append(
            f"  包含检查异常 {row['inclusion_errors']}; actual 50% 约束超限配对 "
            f"{row['budget_violation_pairs']}; 取整后等同基础双窗的配对 {row['mapped_equal_pairs']}"
        )
    lines.extend(
        [
            "",
            f"验证通过 {report['n_confirmed_lower']}/{len(comparisons)} 个超集。",
            "本批所有已测试超集均验证为更少失败。"
            if all_lower
            else "尚不能断言所有已测试超集的失败率都更低；数值差异或检验未通过不等于证明相同。",
            plan["scope"],
        ]
    )
    (directory / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    print(analyze(parser.parse_args().directory))
