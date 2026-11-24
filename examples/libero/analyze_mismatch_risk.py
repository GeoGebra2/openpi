"""Analyze state/image mismatch exposure and frozen held-out failure risk."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

import numpy as np

try:
    from examples.libero import mismatch_risk_metrics as metrics
    from examples.libero import run_critical_window_search as critical
except ImportError:
    import mismatch_risk_metrics as metrics
    import run_critical_window_search as critical


def load(directory):
    try:
        from examples.libero import run_mismatch_risk as run
    except ImportError:
        import run_mismatch_risk as run
    directory = pathlib.Path(directory)
    if directory.is_file():
        directory = directory.parent
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != run.EXPERIMENT or manifest.get("schema_version") != run.SCHEMA_VERSION:
        raise ValueError("Requires a mismatch-risk experiment directory")
    args = run.Args(**manifest["config"])
    registry = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(directory.glob("episodes/*/*/pair_*/record.json"))]
    identities = [(row["split"], row["candidate_id"], row["pair_index"]) for row in records]
    if len(set(identities)) != len(records):
        raise ValueError("Duplicate episode identity")
    for row in records:
        expected = row["pair_index"] + (run.VALIDATION_OFFSET if row["split"] == "validation" else 0)
        if row["repeat_index"] != expected:
            raise ValueError("Discovery and validation seeds overlap or do not match the protocol")
        if row["outcome"] in ("success", "failure") and row["success"] != (row["outcome"] == "success"):
            raise ValueError("Inconsistent task outcome")
    return directory, manifest, args, registry, records


def build_rows(directory, manifest, args, registry, records):
    try:
        from examples.libero import run_mismatch_risk as run
    except ImportError:
        import run_mismatch_risk as run
    if "mismatch_calibration" not in manifest:
        return [], []
    rows, reference_audit = [], []
    clean = {(row["split"], row["pair_index"]): row for row in records if row["candidate_id"] == "clean"}
    for record in records:
        if record["outcome"] not in ("success", "failure"):
            continue
        reference = clean[(record["split"], record["pair_index"])]
        reference_steps, reason = run.calibration_reference(reference, manifest["timing_calibration"])
        if record["candidate_id"] == "clean":
            reference_audit.append({"split": record["split"], "pair_index": record["pair_index"],
                                    "initial_state_id": record["initial_state_id"], "outcome": record["outcome"],
                                    "raw_steps": record["steps"], "normalization_steps": reference_steps,
                                    "reference_duration_source": reason})
        trace = critical.read_trace(directory, record)
        values, _, _ = metrics.exposure(trace, manifest["mismatch_calibration"], args.control_hz)
        windows = record["windows_steps"]
        start = windows[0][0] / reference_steps if windows else 0.0
        end = windows[-1][1] / reference_steps if windows else 0.0
        gaps = [(b[0] - a[1]) / reference_steps for a, b in zip(windows[:-1], windows[1:])]
        rows.append({"split": record["split"], "candidate_id": record["candidate_id"],
                     "pair_index": record["pair_index"], "initial_state_id": record["initial_state_id"],
                     "failure": int(not record["success"]), "effective_delay_ms": record["effective_delay_ms"],
                     "start_frac": start, "end_frac": end,
                     "duration_frac": sum(b - a for a, b in windows) / reference_steps,
                     "max_gap_frac": max(gaps) if gaps else 0.0, "window_count": len(windows),
                     "reference_s": reference_steps / args.control_hz,
                     "reference_duration_source": reason, "raw_clean_steps": reference["steps"],
                     "n_delayed_queries": record["n_delayed_queries"], **values})
    return rows, reference_audit


def training_fingerprint(rows):
    data = sorted((row for row in rows if row["split"] == "discovery"),
                  key=lambda row: (row["candidate_id"], row["pair_index"]))
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze_models(directory):
    directory, manifest, args, registry, records = load(directory)
    rows, _ = build_rows(directory, manifest, args, registry, records)
    training = [row for row in rows if row["split"] == "discovery"]
    fingerprint = training_fingerprint(training)
    path = directory / "risk_models.json"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen["training_sha256"] != fingerprint:
            raise ValueError("Discovery data changed after risk models were frozen")
        return frozen
    if not (directory / "validation_plan.json").exists():
        raise ValueError("Lock the validation plan before freezing risk models")
    if any(row["split"] == "validation" for row in records):
        raise ValueError("Risk models must be frozen before validation begins")
    y = [row["failure"] for row in training]
    models = {name: metrics.fit_logistic(training, features) for name, features in metrics.MODEL_FEATURES.items()}
    result = {"training_sha256": fingerprint, "training_n": len(training), "training_failures": sum(y),
              "baseline_failure_probability": (sum(y) + 0.5) / (len(y) + 1), "models": models,
              "status": "fitted" if models["mismatch_time"] else "one_class_training_data",
              "fit_protocol": "Fixed ridge=2; discovery-only mean/std, coefficients and prevalence; no held-out tuning.",
              "prediction_scope": "Post-exposure association with final outcome; not a prospective early-warning or causal model."}
    critical.atomic_json(path, result)
    return result


def paired_tables(records, plan, expected):
    grouped = collections.defaultdict(list)
    for row in records:
        if row["split"] == "validation":
            grouped[row["candidate_id"]].append(row)
    rates = []
    for key in ["clean", *[spec["id"] for spec in plan["candidates"]]]:
        stats = critical.paired_statistics(grouped["clean"], grouped[key], expected)
        rates.append({"candidate_id": key, **stats})
    comparisons = []
    for spec in plan["primary_comparisons"]:
        comparisons.append({**spec, **critical.paired_statistics(grouped[spec["reference_id"]],
                                                                 grouped[spec["candidate_id"]], expected)})
    adjusted = critical.holm_adjust({row["id"]: row["p_one_sided"] for row in comparisons})
    for row in comparisons:
        row["p_holm"] = adjusted.get(row["id"])
    return rates, comparisons


def recovery_table(records, plan, args):
    grouped = collections.defaultdict(dict)
    for row in records:
        if row["split"] == "validation" and row["outcome"] in ("success", "failure"):
            grouped[row["candidate_id"]][row["pair_index"]] = row
    result = []
    for spec in plan.get("recovery", {}).get("groups", []):
        keys = ["clean", spec["first"], spec["second"], spec["both"]]
        pairs = sorted(set.intersection(*(set(grouped[key]) for key in keys)))
        if not pairs:
            continue
        interactions, gaps, mapping_errors = [], [], 0
        for pair in pairs:
            clean, first, second, both = [grouped[key][pair] for key in keys]
            interactions.append(int(not both["success"]) - int(not first["success"])
                                - int(not second["success"]) + int(not clean["success"]))
            expected_windows = critical.merge_windows(first["windows_steps"] + second["windows_steps"])
            mapping_errors += int(expected_windows != both["windows_steps"])
            windows = both["windows_steps"]
            gaps.append((windows[1][0] - windows[0][1]) / args.control_hz if len(windows) == 2 else 0)
        rng = np.random.default_rng(args.seed + int(100 * spec["gap_frac"]))
        bootstrap = np.mean(np.asarray(interactions)[rng.integers(0, len(pairs), (500, len(pairs)))], axis=1)
        result.append({**spec, "pairs": len(pairs), "mean_actual_gap_s": float(np.mean(gaps)),
                       "interaction": float(np.mean(interactions)), "ci_low": float(np.quantile(bootstrap, 0.025)),
                       "ci_high": float(np.quantile(bootstrap, 0.975)), "mapping_errors": mapping_errors,
                       "inference": "Exploratory paired bootstrap; not corrected evidence of an internal memory mechanism."})
    return result


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def plot_timing(reference_audit, directory, hz):
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(10, 4))
    for index, row in enumerate(reference_audit):
        abnormal = row["reference_duration_source"] != "paired_clean_duration"
        axis.plot([index, index], [row["raw_steps"] / hz, row["normalization_steps"] / hz],
                  color="#b65041" if abnormal else "0.7", alpha=0.6)
        axis.scatter(index, row["raw_steps"] / hz, marker="x", color="#b65041" if abnormal else "0.4")
        axis.scatter(index, row["normalization_steps"] / hz, marker="o", color="#367c9c", s=24)
    axis.plot([], [], "x", color="0.4", label="Raw paired clean duration")
    axis.plot([], [], "o", color="#367c9c", label="Window-normalization duration")
    axis.set(xlabel="Clean episode (discovery then validation)", ylabel="Duration (s)",
             title="Clean timing audit | outcomes retained when duration is replaced")
    axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "clean_duration_audit.png", dpi=150)
    plt.close(fig)


def plot_search(rows, directory):
    plt = pyplot()
    training = [row for row in rows if row["split"] == "discovery" and row["window_count"] == 1]
    delays = sorted({row["effective_delay_ms"] for row in training})
    if not delays:
        return
    fig, axes = plt.subplots(1, len(delays), figsize=(5 * len(delays), 4), squeeze=False, sharey=True)
    for axis, delay in zip(axes[0], delays):
        groups = collections.defaultdict(list)
        for row in training:
            if row["effective_delay_ms"] == delay:
                groups[(row["candidate_id"], row["start_frac"], row["duration_frac"])].append(row["failure"])
        for (_, start, duration), failures in groups.items():
            axis.scatter(start * 100, duration * 100, c=[np.mean(failures)], vmin=0, vmax=1,
                         cmap="YlOrRd", s=100, marker="s", edgecolor="0.5", linewidth=0.3)
        axis.set(xlabel="Window start (% of clean reference)", title=f"{delay:g} ms | discovery", xlim=(-3, 100))
    axes[0, 0].set_ylabel("Window duration (% of clean reference)")
    fig.suptitle("Measured final failure rates | unmeasured locations left blank")
    fig.tight_layout()
    fig.savefig(directory / "window_search.png", dpi=150)
    plt.close(fig)


def plot_risk(rows, directory):
    plt = pyplot()
    validation = [row for row in rows if row["split"] == "validation" and row["candidate_id"] != "clean"]
    if not validation:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for failure, marker, label in ((0, "o", "Success"), (1, "x", "Failure")):
        group = [row for row in validation if row["failure"] == failure]
        if group:
            probability = [row["p_mismatch_time"] for row in group] if "p_mismatch_time" in group[0] else "0.5"
            scatter = axes[0].scatter([row["m_peak"] for row in group], [row["m_area_s"] for row in group],
                                      c=probability, vmin=0, vmax=1, cmap="viridis", marker=marker, s=35, label=label)
    axes[0].set(xlabel="Peak state mismatch M_max", ylabel="Exposure integral A_M (s)",
                title="Held-out rollouts | color = frozen predicted risk")
    axes[0].legend()
    if any("p_mismatch_time" in row for row in validation):
        fig.colorbar(scatter, ax=axes[0], label="P(final failure)")
        for name, color in (("timing_delay", "#ba6b39"), ("mismatch_time", "#347da0")):
            predicted = np.array([row["p_" + name] for row in validation])
            outcome = np.array([row["failure"] for row in validation])
            order = np.argsort(predicted)
            for index, group in enumerate(np.array_split(order, min(5, max(1, len(order) // 10)))):
                if not len(group):
                    continue
                observed, predicted_mean = outcome[group].mean(), predicted[group].mean()
                lo, hi = critical.wilson(int(outcome[group].sum()), len(group))
                axes[1].errorbar(predicted_mean, observed, yerr=[[max(0, observed - lo)], [max(0, hi - observed)]],
                                 fmt="o", color=color, capsize=3, label=name if index == 0 else None)
        axes[1].plot([0, 1], [0, 1], "--", color="0.6")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Insufficient training outcome variation\nObserved data still saved", ha="center")
    axes[1].set(xlabel="Mean frozen prediction", ylabel="Observed failure fraction", xlim=(-0.03, 1.03),
                ylim=(-0.03, 1.03), title="Held-out calibration | descriptive Wilson intervals")
    fig.tight_layout()
    fig.savefig(directory / "risk_prediction.png", dpi=150)
    plt.close(fig)


def plot_surface(rows, directory):
    """Show only measured dose cells, avoiding unsupported continuous extrapolation."""
    plt = pyplot()
    training = [row for row in rows if row["split"] == "discovery" and row["candidate_id"] != "clean"]
    validation = [row for row in rows if row["split"] == "validation" and row["candidate_id"] != "clean"]
    if not training or not validation:
        return
    def edges(values):
        values = np.asarray(values)
        # Floating-point noise is not a distinct geometric dose bin.
        tolerance = 1e-3 * max(1, float(np.max(np.abs(values))))
        if np.ptp(values) < tolerance:
            margin = max(0.01, abs(float(np.median(values))) * 0.05)
            return np.array([float(np.min(values)) - margin, float(np.max(values)) + margin])
        raw = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
        distinct = [raw[0]]
        for value in raw[1:-1]:
            if value - distinct[-1] > tolerance:
                distinct.append(value)
        if raw[-1] - distinct[-1] > tolerance:
            distinct.append(raw[-1])
        else:
            distinct[-1] = raw[-1]
        return np.array(distinct)

    x = edges([row["m_peak"] for row in training])
    y = edges([row["m_area_s"] for row in training])
    observed = np.full((len(y) - 1, len(x) - 1), np.nan)
    predicted = observed.copy()
    counts = np.zeros_like(observed, dtype=int)
    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            group = [row for row in validation if x[j] <= row["m_peak"] <= x[j + 1]
                     and y[i] <= row["m_area_s"] <= y[i + 1]
                     and (j == len(x) - 2 or row["m_peak"] < x[j + 1])
                     and (i == len(y) - 2 or row["m_area_s"] < y[i + 1])]
            counts[i, j] = len(group)
            if len(group) >= 3:
                observed[i, j] = np.mean([row["failure"] for row in group])
                if "p_mismatch_time" in group[0]:
                    predicted[i, j] = np.mean([row["p_mismatch_time"] for row in group])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, values, title in zip(axes, (observed, predicted), ("Observed held-out failure", "Mean frozen predicted risk")):
        image = axis.imshow(values, origin="lower", vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        for i, j in zip(*np.where(counts > 0)):
            axis.text(j, i, f"n={counts[i,j]}" + (f"\n{values[i,j]:.2f}" if np.isfinite(values[i, j]) else ""),
                      ha="center", va="center", fontsize=8,
                      color="white" if np.isfinite(values[i, j]) and values[i, j] > 0.6 else "black")
        axis.set_xticks(range(len(x) - 1))
        axis.set_xticklabels([f"{a:.3g}–{b:.3g}" for a, b in zip(x[:-1], x[1:])], rotation=25, fontsize=8)
        axis.set_yticks(range(len(y) - 1))
        axis.set_yticklabels([f"{a:.3g}–{b:.3g}" for a, b in zip(y[:-1], y[1:])], fontsize=8)
        axis.set(xlabel="Peak mismatch M_max", title=title)
    axes[0].set_ylabel("Exposure integral A_M (s)")
    fig.suptitle("Initial risk map | discovery-fixed bins; cells with n<3 left blank")
    fig.subplots_adjust(left=0.12, bottom=0.23, wspace=0.28, right=0.85, top=0.84)
    fig.colorbar(image, cax=fig.add_axes([0.89, 0.25, 0.018, 0.55]), label="P(final failure)")
    fig.savefig(directory / "risk_map.png", dpi=150)
    plt.close(fig)


def plot_recovery(rows, directory):
    if not rows:
        return
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(7, 4))
    for row in rows:
        value = row["interaction"]
        axis.errorbar(row["mean_actual_gap_s"], value,
                      yerr=[[max(0, value - row["ci_low"])], [max(0, row["ci_high"] - value)]],
                      fmt="o", capsize=4, color="#347da0")
    axis.axhline(0, linestyle="--", color="0.6")
    axis.set(xlabel="Normal-input gap (s)", ylabel="P11 - P10 - P01 + P00",
             title="Recovery quartet interaction | exploratory paired bootstrap")
    fig.tight_layout()
    fig.savefig(directory / "recovery_interaction.png", dpi=150)
    plt.close(fig)


def analyze(directory):
    directory, manifest, args, registry, records = load(directory)
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    rows, reference_audit = build_rows(directory, manifest, args, registry, records)
    model_path = directory / "risk_models.json"
    frozen = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else None
    model_scores, baseline_scores = {}, {}
    if frozen:
        if frozen["training_sha256"] != training_fingerprint(rows):
            raise ValueError("Frozen risk model training data no longer matches discovery traces")
        validation = [row for row in rows if row["split"] == "validation"]
        for name, model in frozen["models"].items():
            if model is None:
                continue
            probabilities = metrics.predict(model, rows)
            for row, probability in zip(rows, probabilities):
                row["p_" + name] = float(probability)
            model_scores[name] = {}
            for subset, sample in (("all", validation),
                                   ("interventions", [row for row in validation if row["candidate_id"] != "clean"])):
                p = np.array([row["p_" + name] for row in sample])
                baseline = np.repeat(frozen["baseline_failure_probability"], len(sample))
                model_scores[name][subset] = metrics.scores(sample, p)
                baseline_scores[subset] = metrics.scores(sample, baseline)
                if sample:
                    model_scores[name][subset]["brier_improvement_vs_prevalence"] = metrics.grouped_brier_improvement(
                        sample, p, baseline, args.seed)
                    if name != "timing_delay" and "p_timing_delay" in sample[0]:
                        timing_prediction = np.array([row["p_timing_delay"] for row in sample])
                        model_scores[name][subset]["brier_improvement_vs_timing_delay"] = metrics.grouped_brier_improvement(
                            sample, p, timing_prediction, args.seed)
    plan_path = directory / "validation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    rates, comparisons = paired_tables(records, plan, args.validation_repeats) if plan else ([], [])
    recovery = recovery_table(records, plan, args) if plan else []
    train = [row for row in rows if row["split"] == "discovery"]
    valid = [row for row in rows if row["split"] == "validation"]
    failures = sum(row["failure"] for row in train)
    vfailures = sum(row["failure"] for row in valid)
    enough = min(failures, len(train) - failures) >= 20 and min(vfailures, len(valid) - vfailures) >= 10
    score = model_scores.get("mismatch_time", {}).get("interventions")
    improvement = score.get("brier_improvement_vs_prevalence") if score else None
    primary = frozen.get("models", {}).get("mismatch_time") if frozen else None
    supported = bool(manifest["status"] == "complete" and enough and improvement and improvement["ci_low"] > 0
                     and primary and primary["converged"])
    report = {"experiment": manifest["experiment"], "status": manifest["status"], "config": manifest["config"],
              "timing_calibration": manifest.get("timing_calibration"),
              "mismatch_calibration": manifest.get("mismatch_calibration"),
              "training_n": len(train), "training_failures": failures, "validation_n": len(valid),
              "validation_failures": vfailures, "model_scores": model_scores, "prevalence_baseline": baseline_scores,
              "initial_predictive_evidence": supported, "outcome_variation_adequate": enough,
              "model_status": frozen["status"] if frozen else "not_yet_frozen",
              "reference_durations_replaced": sum(row["reference_duration_source"] != "paired_clean_duration" for row in reference_audit),
              "runtime_errors": sum(row["outcome"] == "error" for row in records),
              "validation_rates": rates, "paired_comparisons": comparisons, "recovery": recovery,
              "limits": ["Measured state-source distance covers robot self state, not environmental appearance changes.",
                         "Exposure is measured along completed closed-loop execution and may be shortened by early success; "
                         "this is a post-exposure association model, not prospective early-warning validation.",
                         "Time and dose are not independently manipulated. Recovery interactions are exploratory.",
                         "Held-out seeds test within-task prediction; unseen-task generalization is not established.",
                         "No monotonic risk or fixed universal threshold is assumed; insufficient events never count as success."]}
    critical.atomic_json(output / "report.json", report)
    for filename, values in (("episodes.csv", rows), ("clean_duration_audit.csv", reference_audit),
                             ("validation_rates.csv", rates), ("paired_comparisons.csv", comparisons),
                             ("recovery_interactions.csv", recovery)):
        critical.write_csv(output / filename, values)
    plot_timing(reference_audit, output, args.control_hz)
    plot_search(rows, output)
    plot_risk(rows, output)
    plot_surface(rows, output)
    plot_recovery(recovery, output)
    lines = ["图像与实时 self state 错配风险实验", f"任务: {args.task_suite_name}/{args.task_id}; 相机: {args.delay_target}",
             f"状态: {manifest['status']}; 发现集 {len(train)} 回合/{failures} 失败；独立验证 {len(valid)} 回合/{vfailures} 失败。",
             f"clean 归一化时长替换: {report['reference_durations_replaced']} 回合；原始成败全部保留。",
             f"模型状态: {report['model_status']}",
             "已获得初步独立预测证据。" if supported else "尚未建立充分的独立预测证据；请查看事件数量、验证 Brier/AUROC 与区间。",
             "模型在验证开始前冻结；预测的是已观测延迟暴露与最终失败的关系，尚不是在线预警器。"]
    if score:
        lines.append(f"干预验证集 Brier={score['brier']:.4f}; AUROC={score['auroc']}。")
        lines.append(f"相对常数预测的 Brier 改善及区间: {score.get('brier_improvement_vs_prevalence')}。")
        lines.append(f"相对时间/毫秒模型的 Brier 改善及区间: {score.get('brier_improvement_vs_timing_delay')}。")
    (directory / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(analyze(parser.parse_args().directory))
