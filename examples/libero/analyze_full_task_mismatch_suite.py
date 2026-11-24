"""Aggregate completed full-task mismatch runs from several LIBERO tasks."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

try:
    from examples.libero import analyze_full_task_mismatch as single_analysis
except ImportError:
    import analyze_full_task_mismatch as single_analysis


METRICS = single_analysis.MEASURES


def write_csv(path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with pathlib.Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    integer = {"pair_index", "seed", "delay_ms", "success", "failure", "steps"}
    numeric = {"task_s", "m_max", "a_m", "m_d"}
    for row in rows:
        for key in ("video_path", "video_timing_path", "video_error"):
            row.setdefault(key, "")
        for key in integer:
            row[key] = int(float(row[key]))
        for key in numeric:
            row[key] = float(row[key])
    return rows


def wilson(failures, count):
    return single_analysis.wilson(failures, count)


def failure_rate_rows(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[key] for key in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        count, failures = len(group), sum(row["failure"] for row in group)
        low, high = wilson(failures, count)
        item = dict(zip(group_keys, key))
        item.update(n=count, failures=failures, successes=count - failures,
                    failure_rate=failures / count, failure_ci_low=low, failure_ci_high=high)
        output.append(item)
    return output


def metric_bins(rows, metric, bin_count):
    return single_analysis.bin_failures(rows, metric, bin_count)


def analyze(directory, bins=None):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / "suite_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Suite must complete all selected tasks before aggregate analysis")
    bins = manifest.get("config", {}).get("bins", 10) if bins is None else bins
    if bins < 1:
        raise ValueError("bins must be positive")
    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    all_rows, task_rows = [], []
    for task in manifest["tasks"]:
        child = directory / task["directory"]
        child_manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        if child_manifest.get("status") != "complete":
            raise ValueError(f"Task {task['task_id']} is not complete")
        rows = read_csv(child / "analysis" / "episodes.csv")
        for row in rows:
            row["task_id"] = int(task["task_id"])
            row["task_name"] = task["task_name"]
            row["source_directory"] = task["directory"]
            for key in ("video_path", "video_timing_path"):
                if row[key]:
                    row[key] = (pathlib.Path(task["directory"]) / row[key]).as_posix()
        all_rows.extend(rows)
        task_rows.extend(row for row in rows if row["phase"] == "sweep")
    if not task_rows:
        raise ValueError("No sweep rows found")
    sweep = task_rows
    write_csv(output / "episodes.csv", all_rows)
    delay_rates = failure_rate_rows(sweep, ["delay_ms"])
    task_delay_rates = failure_rate_rows(sweep, ["task_id", "task_name", "delay_ms"])
    write_csv(output / "delay_rates.csv", delay_rates)
    write_csv(output / "task_delay_rates.csv", task_delay_rates)
    bins_rows = []
    subtitle = (f"{manifest['task_suite_name']} | {len(manifest['tasks'])} tasks | "
                f"all sweep delays pooled | n={len(sweep)}")
    for metric in METRICS:
        groups = metric_bins(sweep, metric, bins)
        bins_rows.extend(groups)
        write_csv(output / f"{metric}_bins.csv", groups)
        single_analysis.plot_bins(groups, metric, output, subtitle)
    report = {
        "experiment": manifest["experiment"], "task_suite_name": manifest["task_suite_name"],
        "task_count": len(manifest["tasks"]), "task_ids": manifest["task_ids"],
        "sweep_n": len(sweep), "sweep_failures": sum(row["failure"] for row in sweep),
        "tasks": manifest["tasks"], "delay_rates": delay_rates, "task_delay_rates": task_delay_rates,
        "bins_requested": bins, "bins": bins_rows,
        "protocol": "Rows are combined across tasks for aggregate rates; task_delay_rates preserves stratification. Calibration rows are excluded from sweep metrics.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {manifest['task_suite_name']} 多任务全任务时延错配汇总", "",
             f"任务数：{len(manifest['tasks'])}；任务 ID：{','.join(map(str, manifest['task_ids']))}",
             f"Sweep 回合数：{len(sweep)}；失败数：{sum(row['failure'] for row in sweep)}",
             "每个任务先独立完成校准和时延实验，再合并 sweep 记录。下面的延迟统计是所有选定任务合并后的描述性统计；任务差异请查看 task_delay_rates.csv。", "",
             "## 按延迟的合并失败率", "", "| 延迟(ms) | 回合数 | 失败数 | 失败率 | 95% Wilson 区间 |", "|---:|---:|---:|---:|---:|"]
    for row in delay_rates:
        lines.append(f"| {row['delay_ms']} | {row['n']} | {row['failures']} | {100*row['failure_rate']:.1f}% | {100*row['failure_ci_low']:.1f}%–{100*row['failure_ci_high']:.1f}% |")
    lines += ["", "合并统计适合描述套件总体表现，但不同任务的成功率、轨迹长度和偏差分布可能不同；不要把合并曲线解释成每个任务都具有相同风险函数。", "", "## 测量量分箱与任务失败率", "",
              f"合并所有选定任务、所有延迟条件的 sweep 回合，分别按每个指标的实测范围等宽划分 {bins} 箱。横轴为测量量区间，纵轴为该箱失败数/总回合数；clean 校准回合不进入分箱。",
              "柱上标注失败数/总数，误差线为描述性 95% Wilson 区间；空箱标为 n=0，不表示零失败率。", ""]
    for metric in METRICS:
        lines.append(f"![{metric} failure rate]({(output / METRICS[metric][1]).resolve().as_posix()})")
        lines.append("")
    lines += ["## 输出文件", "", "`episodes.csv` 是带 task_id/task_name 的合并记录；`delay_rates.csv` 是合并失败率；`task_delay_rates.csv` 保留每个任务的分层结果；三个 `*_failure_rate.png` 是测量量分箱—任务失败率条形图；`*_bins.csv` 是对应的分箱底表。"]
    (output / "report.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--bins", type=int, default=None)
    args = parser.parse_args()
    print(analyze(args.directory, args.bins))
