"""Offline paired success analysis for persistent wrist KV repair experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_effect(reference, treatment):
    keys = sorted(reference.keys() & treatment.keys())
    if not keys:
        return {"n_pairs": 0, "delta_success_rate": None, "bootstrap_ci95": None, "rescued": 0, "harmed": 0}
    a = np.array([reference[key] for key in keys], dtype=int)
    b = np.array([treatment[key] for key in keys], dtype=int)
    differences = b - a
    rescued = int(np.sum(differences == 1))
    harmed = int(np.sum(differences == -1))
    # Resample whole paired episodes, never individual inference steps or tokens.
    rng = np.random.default_rng(0)
    if len(keys) > 1 and not np.all(differences == differences[0]):
        bootstrap = np.array([rng.choice(differences, len(keys), replace=True).mean() for _ in range(4000)])
        interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    else:
        interval = None
    return {
        "n_pairs": len(keys), "delta_success_rate": float(differences.mean()), "bootstrap_ci95": interval,
        "matched_delay_success_rate": float(a.mean()), "matched_condition_success_rate": float(b.mean()),
        "rescued": rescued, "harmed": harmed, "both_success": int(np.sum((a == 1) & (b == 1))),
        "both_failure": int(np.sum((a == 0) & (b == 0))),
        "degenerate_bootstrap": bool(np.all(differences == differences[0])),
    }


def _pair_key(record):
    return tuple(record[key] for key in (
        "repeat_index", "seed", "task_suite_name", "task_id", "initial_state_id", "prompt",
    ))


def _trace_path(directory, record):
    value = record.get("trace_path")
    if not value:
        return None
    path = directory / pathlib.PurePosixPath(value.replace("\\", "/")).name
    return path if path.is_file() else None


def _sham_trace_checks(directory, completed):
    delayed = completed.get("delay", {})
    sham = completed.get("self", {})
    reports = []
    for key in sorted(delayed.keys() & sham.keys()):
        a_path, b_path = _trace_path(directory, delayed[key]), _trace_path(directory, sham[key])
        row = {"repeat_index": key[0], "available": bool(a_path and b_path)}
        if a_path and b_path:
            with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
                x, y = np.asarray(a["executed_actions"]), np.asarray(b["executed_actions"])
                row["same_shape"] = x.shape == y.shape
                row["exact_equal"] = bool(np.array_equal(x, y))
                if x.shape == y.shape and x.size:
                    delta = x.astype(np.float64) - y.astype(np.float64)
                    row["max_abs"] = float(np.abs(delta).max())
                    row["rmse"] = float(np.sqrt(np.square(delta).mean()))
        reports.append(row)
    return reports


def _write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory: pathlib.Path):
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != "libero_wrist_kv_patch" or report.get("schema_version") != 1:
        raise ValueError("Expected wrist KV patch experiment schema v1")
    conditions = [setting["name"] for setting in report["conditions"]]
    if len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate condition names")
    completed = {name: {} for name in conditions}
    sites = sorted({record.get("patch_site", report["config"].get("site", "wrist_kv")) for record in report["records"]})
    if len(sites) > 1:
        raise ValueError(f"Cannot combine multiple patch sites: {sites}")
    seen = set()
    for record in report["records"]:
        name = record["condition"]
        key = _pair_key(record)
        if name not in completed or (name, key) in seen:
            raise ValueError("Unknown condition or duplicate paired episode")
        seen.add((name, key))
        if record["outcome"] not in ("success", "failure", "error", "interrupted"):
            raise ValueError("Unknown episode outcome")
        if record["outcome"] in ("success", "failure"):
            completed[name][key] = record
    binary = {
        name: {key: record["outcome"] == "success" for key, record in group.items()}
        for name, group in completed.items()
    }
    summary, paired_rows = [], []
    for name in conditions:
        group = [row for row in report["records"] if row["condition"] == name]
        total = len(binary[name])
        successes = sum(binary[name].values())
        interval = wilson(successes, total)
        effect = paired_effect(binary.get("delay", {}), binary[name])
        summary.append({
            "condition": name, "planned": report["config"]["repeats_per_setting"], "attempted": len(group),
            "completed": total, "successes": successes, "failures": total - successes,
            "errors": sum(row["outcome"] == "error" for row in group),
            "interrupted": sum(row["outcome"] == "interrupted" for row in group),
            "success_rate": successes / total if total else None,
            "wilson_low": interval[0] if interval else None, "wilson_high": interval[1] if interval else None,
            "n_pairs_vs_delay": effect["n_pairs"], "delta_vs_delay": effect["delta_success_rate"],
            "rescued": effect["rescued"], "harmed": effect["harmed"],
        })
        if name != "delay":
            for key in sorted(binary.get("delay", {}).keys() & binary[name].keys()):
                paired_rows.append({
                    "condition": name, "repeat_index": key[0], "seed": key[1],
                    "delay_success": int(binary["delay"][key]), "condition_success": int(binary[name][key]),
                    "delta": int(binary[name][key]) - int(binary["delay"][key]),
                })
    comparisons = {name: paired_effect(binary.get("delay", {}), binary[name]) for name in conditions if name != "delay"}
    result = {
        "experiment": "libero_wrist_kv_patch_analysis", "schema_version": 1,
        "config": report["config"], "patch_site": sites[0] if sites else report["config"].get("site"),
        "conditions": summary, "paired_vs_delay": comparisons,
        "sham_trace_checks": _sham_trace_checks(directory, completed),
        "notes": [
            "Success intervals: Wilson 95%; paired deltas: episode-pair percentile bootstrap 95% (4000 resamples).",
            "Bootstrap intervals are omitted for one pair or identical pair differences; consult Wilson intervals.",
            "Failures include max_steps. Errors/interruption are excluded from completed rates and listed separately.",
            "Compare only within matched repeat, seed, task, initial state and prompt in this experiment directory.",
            "Layer scan is exploratory with multiple comparisons; validate selected layers using new seeds.",
            "Repair identifies intervention efficacy, not a necessary or unique failure mechanism.",
            "All-layer wrist patch retains delayed information at non-wrist slots; it is not full clean inference.",
        ],
    }
    output = directory / "kv_patch_analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(directory / "kv_patch_success.csv", summary)
    _write_csv(directory / "kv_patch_pairs.csv", paired_rows,
               ["condition", "repeat_index", "seed", "delay_success", "condition_success", "delta"])
    _plot(directory, result)
    return output


def _plot(directory, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["conditions"]
    for filename, selected in (
        ("kv_patch_pilot.png", [row for row in rows if not row["condition"].startswith("layer_")]),
        ("kv_patch_layers.png", [row for row in rows if row["condition"].startswith("layer_")]),
    ):
        if not selected:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(max(9, len(selected) * 0.55), 8), constrained_layout=True)
        valid = [(i, row) for i, row in enumerate(selected) if row["completed"]]
        for index, row in valid:
            rate = row["success_rate"]
            axes[0].errorbar(index, rate, yerr=[[max(0, rate - row["wilson_low"])],
                                              [max(0, row["wilson_high"] - rate)]],
                             fmt="o", color="#267e99", capsize=4)
            axes[0].annotate(f"{row['successes']}/{row['completed']}", (index, rate), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=8)
        for row in selected:
            if not row["completed"]:
                axes[0].text(selected.index(row), 0.5, "No completed\nruns", ha="center", fontsize=8)
        for control, color in (("clean", "#168163"), ("delay", "#c4463d"), ("patch_all", "#8a67a9")):
            row = next((row for row in rows if row["condition"] == control and row["completed"]), None)
            if row:
                axes[0].axhline(row["success_rate"], color=color, linestyle="--", alpha=0.6, label=control)
        for index, row in enumerate(selected):
            effect = report["paired_vs_delay"].get(row["condition"])
            if effect and effect["n_pairs"]:
                delta = effect["delta_success_rate"] * 100
                interval = effect["bootstrap_ci95"]
                error = [[max(0, delta - interval[0] * 100)], [max(0, interval[1] * 100 - delta)]] if interval else None
                axes[1].errorbar(index, delta, yerr=error, fmt="o", color="#168163", capsize=4)
                axes[1].text(index, 128, f"n={effect['n_pairs']}\n+{effect['rescued']}/-{effect['harmed']}",
                             ha="center", va="top", fontsize=8)
        axes[0].set(ylim=(-0.05, 1.13), ylabel="success rate", title="Success rate with Wilson 95% intervals")
        axes[0].legend(fontsize=8)
        axes[1].axhline(0, color="#999999", linewidth=1)
        axes[1].set(ylabel="paired improvement (percentage points)", ylim=(-115, 150),
                    title="Relative to delay | paired bootstrap 95% | +rescued / -harmed")
        for ax in axes:
            ax.set_xticks(range(len(selected)))
            ax.set_xticklabels([row["condition"] for row in selected], rotation=45, ha="right")
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"Persistent wrist delay {report['config']['delay_ms']} ms | "
            f"site={report.get('patch_site', report['config'].get('site', 'wrist_kv'))} | "
            f"{report['config']['phase']}"
        )
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=pathlib.Path)
    args = parser.parse_args()
    print(analyze(args.input_dir))
