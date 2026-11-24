"""Empirical failure rates in independently binned full-task mismatch measurements."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np


MEASURES = {
    "m_max": ("Peak mismatch M_max", "m_max_failure_rate.png"),
    "a_m": ("Integrated mismatch A_M (s)", "a_m_failure_rate.png"),
    "m_d": ("Mean mismatch M_d = A_M / executed task duration", "m_d_failure_rate.png"),
}


def write_csv(path, rows):
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(failures, count):
    if not count:
        return None, None
    p, z = failures / count, 1.959963984540054
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def bin_failures(rows, measure, bins):
    """Equal-width bins from metric values only; [left,right), last right-inclusive."""
    if bins < 1 or not rows:
        raise ValueError("Need positive bins and at least one completed sweep episode")
    values = np.asarray([row[measure] for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"Invalid {measure} values")
    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        # Repeated identical measurements cannot form several distinct intervals.
        padding = max(abs(lower) * 0.01, 1e-6)
        edges = np.array([max(0.0, lower - padding), upper + padding])
    else:
        edges = np.unique(np.linspace(lower, upper, bins + 1))
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    result = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        group = [r for r, slot in zip(rows, indices) if slot == index]
        count, failures = len(group), sum(r["failure"] for r in group)
        lo, hi = wilson(failures, count)
        delays = dict(sorted(collections.Counter(r["delay_ms"] for r in group).items()))
        result.append({"metric": measure, "bin_index": index, "left": float(left), "right": float(right),
                       "right_inclusive": index == len(edges) - 2, "n": count, "failures": failures,
                       "successes": count - failures, "failure_rate": failures / count if count else None,
                       "failure_ci_low": lo, "failure_ci_high": hi,
                       "delay_counts": json.dumps(delays, sort_keys=True)})
    if sum(r["n"] for r in result) != len(rows):
        raise ValueError("Binning lost episodes")
    return result


def plot_bins(groups, measure, output, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(max(9, len(groups) * 0.8), 5.8))
    labels = []
    for i, row in enumerate(groups):
        close = "]" if row["right_inclusive"] else ")"
        labels.append(f"[{row['left']:.4g}, {row['right']:.4g}{close}")
        if row["n"]:
            value = row["failure_rate"] * 100
            axis.bar(i, value, color="#287d8e", width=0.78)
            low, high = row["failure_ci_low"] * 100, row["failure_ci_high"] * 100
            axis.errorbar(i, value, yerr=[[max(0, value - low)], [max(0, high - value)]],
                          fmt="none", color="#23353d", capsize=3)
            axis.text(i, high + 2, f"{row['failures']}/{row['n']}", ha="center", fontsize=9)
        else:
            axis.bar(i, 3, color="white", edgecolor="0.65", hatch="///", width=0.78)
            axis.text(i, 5, "n=0", ha="center", color="0.45", fontsize=9)
    axis.set(xticks=range(len(groups)), xticklabels=labels, ylim=(0, 115),
             yticks=[0, 20, 40, 60, 80, 100], ylabel="Task failure rate (%)", xlabel=MEASURES[measure][0])
    axis.tick_params(axis="x", rotation=35, labelsize=9)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_title(f"Empirical failure rate vs {measure.upper()}\n{subtitle}", fontsize=11)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    fig.text(0.5, 0.015, "Labels: failed/total | whiskers: descriptive 95% Wilson intervals | hatched: no observations",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(pathlib.Path(output) / MEASURES[measure][1], dpi=160)
    plt.close(fig)


def load_rows(directory, manifest):
    config = manifest["config"]
    rows, errors, seen = [], [], set()
    for path in sorted(directory.glob("episodes/*/*/pair_*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["phase"], record["condition"], record["pair_index"])
        if key in seen:
            raise ValueError(f"Duplicate episode key {key}")
        seen.add(key)
        if record["outcome"] not in ("success", "failure"):
            errors.append({"key": list(key), "outcome": record["outcome"], "error": record.get("error")})
            continue
        if record["success"] != (record["outcome"] == "success"):
            raise ValueError(f"Inconsistent outcome: {key}")
        duration = record["steps"] / config["control_hz"]
        if duration <= 0:
            raise ValueError(f"No executed actions: {key}")
        sweep = record["phase"] == "sweep"
        peak, area = (record["m_peak"], record["m_area_s"]) if sweep else (0.0, 0.0)
        mean = area / duration
        if not np.all(np.isfinite([peak, area, mean])) or min(peak, area, mean) < 0:
            raise ValueError(f"Invalid measurements: {key}")
        if sweep and (not np.isclose(record["m_mean"], mean) or not np.isclose(record["task_s"], duration)
                      or not np.isclose(record["exposure_s"], duration)
                      or record["n_delayed_queries"] != record["n_queries"]):
            raise ValueError(f"Not full-task exposure or inconsistent mean: {key}")
        rows.append({"phase": record["phase"], "condition": record["condition"],
                     "pair_index": record["pair_index"], "seed": record["seed"],
                     "delay_target": config["delay_target"], "delay_ms": record["effective_delay_ms"],
                     "success": int(record["success"]), "failure": int(not record["success"]),
                     "termination_reason": record["termination_reason"], "steps": record["steps"],
                     "task_s": duration, "m_max": peak, "a_m": area, "m_d": mean,
                     "trace_path": record["trace_path"], "video_path": record.get("video_path") or "",
                     "video_timing_path": record.get("video_timing_path") or "",
                     "video_error": record.get("video_error") or ""})
    return rows, errors


def analyze(directory, bins=None):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != "libero_full_task_mismatch" or manifest.get("schema_version") != 1:
        raise ValueError("Requires a full-task-mismatch experiment")
    config = manifest["config"]
    bins = config["bins"] if bins is None else bins
    if bins < 1:
        raise ValueError("bins must be positive")
    rows, errors = load_rows(directory, manifest)
    levels = sorted({int(d.strip()) for d in config["delay_levels_ms"].split(",") if d.strip()})
    expected = [("calibration", "clean", 0, config["calibration_repeats"])]
    expected.extend(("sweep", f"d{d:03d}", d, config["repeats"]) for d in levels)
    expected_keys = {(p, c, i) for p, c, d, n in expected for i in range(n)}
    actual_keys = {(r["phase"], r["condition"], r["pair_index"]) for r in rows}
    if not actual_keys <= expected_keys:
        raise ValueError("Unexpected episode condition/pair index")
    complete = not errors and actual_keys == expected_keys
    if manifest["status"] == "complete" and not complete:
        raise ValueError("Completed manifest has missing or errored episodes")
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "episodes.csv", rows)
    rates = []
    for phase, condition, delay, count in expected:
        group = [r for r in rows if r["phase"] == phase and r["condition"] == condition]
        n, failed = len(group), sum(r["failure"] for r in group)
        lo, hi = wilson(failed, n)
        rates.append({"phase": phase, "condition": condition, "delay_ms": delay, "expected_n": count,
                      "n": n, "failures": failed, "successes": n - failed,
                      "failure_rate": failed / n if n else None, "success_rate": (n - failed) / n if n else None,
                      "failure_ci_low": lo, "failure_ci_high": hi,
                      "median_task_s": float(np.median([r["task_s"] for r in group])) if n else None,
                      **{f"median_{m}": float(np.median([r[m] for r in group])) if n else None for m in MEASURES}})
    write_csv(output / "delay_rates.csv", rates)
    sweep = [r for r in rows if r["phase"] == "sweep"]
    binned = []
    if sweep:
        subtitle = f"{config['delay_target']} | delays {','.join(map(str, levels))} ms | n={len(sweep)}"
        if not complete:
            subtitle = "PARTIAL RUN | " + subtitle
        for measure in MEASURES:
            groups = bin_failures(sweep, measure, bins)
            binned.extend(groups)
            write_csv(output / f"{measure}_bins.csv", groups)
            plot_bins(groups, measure, output, subtitle)
    report = {"experiment": manifest["experiment"], "run_status": manifest["status"], "complete": complete,
              "config": config, "bins_requested": bins, "calibration": manifest.get("mismatch_calibration"),
              "sweep_n": len(sweep), "sweep_failures": sum(r["failure"] for r in sweep),
              "non_task_outcomes": errors, "delay_rates": rates, "bins": binned,
              "interpretation": ["Empirical pooled association, without fitting a risk model.",
                                 "Three independent equal-width ranges use all completed sweep measurements, not outcomes.",
                                 "Empty bins have no rate; labels are failures/count, final bin includes its right edge.",
                                 "Mean uses each episode's executed duration, including failed timeouts, excluding warmup.",
                                 "Shared seeds correlate observations across delay conditions; Wilson bars are descriptive.",
                                 "Exposure and task duration follow closed-loop execution; curves do not establish causality."]}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["全任务固定图像时延：错配分箱失败率", f"状态: {manifest['status']}; 数据齐全: {complete}",
             f"相机: {config['delay_target']}; 每个时延计划 {config['repeats']} 回合；分箱数 {bins}",
             "T=该回合实际执行步数/控制频率，包含失败超时，排除 warmup；M_d=A_M/T。",
             "clean 仅用于冻结距离尺度，单独报告成败，不进入三个直方统计。"]
    for row in rates:
        rate = f"{100 * row['failure_rate']:.1f}%" if row["n"] else "无数据"
        lines.append(f"{row['condition']}: {row['n']}/{row['expected_n']} 回合，失败 {row['failures']}，失败率 {rate}")
    lines.extend([f"运行错误/中断（不作为任务失败）: {len(errors)}",
                  "三个图合并所有时延条件，各按本指标实测最小值至最大值等宽分箱；标注失败数/总数。",
                  "空箱不代表零风险；图中置信区间为描述性 Wilson 区间，跨时延复用 seed 的相关性尚未校正。",
                  "失败可能延长任务时间并增加 A_M；这些图表示执行后的统计关联，不能据此断言因果阈值。"])
    (directory / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--bins", type=int, default=None)
    options = parser.parse_args()
    print(analyze(options.directory, options.bins))
