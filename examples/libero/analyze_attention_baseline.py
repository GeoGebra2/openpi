"""Offline summaries and plots for ``run_attention_baseline.py`` outputs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Optional

import numpy as np


def analyze(
    input_dir: pathlib.Path,
    *,
    focus_layers: tuple[int, ...] = (0, 8, 11, 12, 14, 17),
    review_steps: tuple[int, int] = (35, 45),
) -> pathlib.Path:
    if review_steps[0] < 0 or review_steps[1] < review_steps[0]:
        raise ValueError("review_steps must be a nonnegative increasing interval")
    if not focus_layers or any(layer < 0 for layer in focus_layers):
        raise ValueError("focus_layers must contain nonnegative layer indices")
    files = sorted(input_dir.glob("*_attention.npz"))
    if not files:
        raise FileNotFoundError(f"No *_attention.npz files found in {input_dir}")
    summary_path = input_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    run_outcomes = {
        pathlib.PurePosixPath(record["attention_path"].replace("\\", "/")).name: record["outcome"]
        for record in summary.get("records", []) if record.get("attention_path")
    }
    skipped = []
    rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    source_groups: Optional[list[str]] = None
    for path in files:
        outcome = run_outcomes.get(path.name, "unknown")
        if outcome in ("error", "interrupted"):
            skipped.append({"file": path.name, "outcome": outcome})
            continue
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        groups = [str(value) for value in data["source_groups"].tolist()]
        if source_groups is None:
            source_groups = groups
        elif groups != source_groups:
            raise ValueError(f"Source groups differ in {path.name}: {groups} vs {source_groups}")
        mass = np.asarray(data["source_mass"], dtype=np.float64)
        first_mass = np.asarray(data["first_action_source_mass"], dtype=np.float64)
        steps = np.asarray(data["env_steps"], dtype=np.int32)
        counts = np.asarray(data["source_token_counts"], dtype=np.float64)
        if (
            mass.ndim != 6
            or first_mass.shape != mass.shape
            or steps.shape != (mass.shape[0],)
            or counts.shape != (mass.shape[0], len(groups))
            or mass.shape[-1] != len(groups)
            or not mass.size
        ):
            raise ValueError(f"Inconsistent attention dimensions in {path.name}: {mass.shape}")
        if any(not np.all(np.isfinite(value)) for value in (mass, first_mass, counts)):
            raise ValueError(f"Non-finite attention values in {path.name}")
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f"Environment steps must increase strictly in {path.name}")
        # Average over KV heads and query heads; keep denoise step and layer.
        mean_mass = np.mean(mass, axis=(3, 4))
        mean_first = np.mean(first_mass, axis=(3, 4))
        agent_indices = [i for i, name in enumerate(groups) if "base" in name or "agentview" in name]
        wrist_indices = [i for i, name in enumerate(groups) if "wrist" in name]
        if not agent_indices or not wrist_indices:
            raise ValueError(f"Could not identify agentview/wrist groups in {groups}")
        agent = np.sum(mean_mass[..., agent_indices], axis=-1)
        wrist = np.sum(mean_mass[..., wrist_indices], axis=-1)
        share = wrist / np.maximum(agent + wrist, 1e-12)
        # First-action query gives a complementary view less diluted by later actions.
        first_agent = np.sum(mean_first[..., agent_indices], axis=-1)
        first_wrist = np.sum(mean_first[..., wrist_indices], axis=-1)
        first_share = first_wrist / np.maximum(first_agent + first_wrist, 1e-12)
        run_name = path.stem[:-len("_attention")] if path.stem.endswith("_attention") else path.stem
        for index, step in enumerate(steps):
            rows.append({
                "run": run_name,
                "outcome": outcome,
                "env_step": int(step),
                "wrist_share_mean": float(np.mean(share[index])),
                "wrist_share_first_action": float(np.mean(first_share[index])),
                "wrist_mass_mean": float(np.mean(wrist[index])),
                "agentview_mass_mean": float(np.mean(agent[index])),
                "valid_wrist_tokens": float(np.sum(counts[index, wrist_indices])),
                "valid_agentview_tokens": float(np.sum(counts[index, agent_indices])),
            })
        all_records.append({
            "path": str(path), "steps": steps, "share": share, "first_share": first_share,
            "mean_mass": mean_mass, "run": run_name, "outcome": outcome,
            "agent": agent, "wrist": wrist,
        })

    if not rows:
        raise ValueError("No usable attention records; inspect summary.json for errors or interruptions")
    rows.sort(key=lambda row: (row["env_step"], row["run"]))
    csv_path = input_dir / "attention_by_env_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["env_step"], []).append(row)
    step_summary = [
        {
            "env_step": step,
            "records": len(group),
            "wrist_share_mean": float(np.mean([row["wrist_share_mean"] for row in group])),
            "wrist_share_std": float(np.std([row["wrist_share_mean"] for row in group])),
            "agentview_mass_mean": float(np.mean([row["agentview_mass_mean"] for row in group])),
            "wrist_mass_mean": float(np.mean([row["wrist_mass_mean"] for row in group])),
        }
        for step, group in sorted(grouped.items())
    ]
    success = _read_success_summary(input_dir)
    detail = _write_details(input_dir, all_records, summary, focus_layers, review_steps)
    report = {
        "experiment": "libero_attention_baseline_analysis",
        "source_groups": source_groups,
        "runs_with_attention": len(all_records),
        "skipped_incomplete_runs": skipped,
        "step_summary": step_summary,
        "success_summary": success,
        "detailed_analysis": detail,
        "interpretation": {
            "wrist_share": "attention mass on wrist image groups divided by mass on agentview plus wrist groups",
            "causal_warning": "This is model attention allocation, not a causal importance estimate.",
            "aggregation": "mean over KV heads and query heads; denoise and layer axes are retained in NPZ files",
            "time_alignment": "runs aligned by environment step, not by manipulation stage",
            "late_steps": "later steps include only runs still active; see the number of contributing runs",
            "first_action": "the first action token of each sampled chunk, not the first observation of an episode",
        },
    }
    report_path = input_dir / "attention_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot(input_dir, all_records, source_groups)
    _plot_details(input_dir, all_records, detail, review_steps)
    return report_path


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_details(input_dir, records, summary, focus_layers, review_steps):
    layer_count = records[0]["share"].shape[2]
    if any(record["share"].shape[2] != layer_count for record in records):
        raise ValueError("Cannot combine runs with different layer counts")
    layer_rows = []
    for record in records:
        # Reduce denoising within a run first; runs, not heads/tokens, are replicates.
        values = {
            "wrist_share": record["share"].mean(axis=1),
            "first_action_wrist_share": record["first_share"].mean(axis=1),
            "wrist_mass": record["wrist"].mean(axis=1),
            "agentview_mass": record["agent"].mean(axis=1),
        }
        for index, step in enumerate(record["steps"]):
            for layer in range(layer_count):
                layer_rows.append({
                    "run": record["run"], "outcome": record["outcome"],
                    "env_step": int(step), "layer": layer,
                    **{key: float(value[index, layer]) for key, value in values.items()},
                })
    _write_csv(input_dir / "attention_by_layer.csv", layer_rows)

    grouped = {}
    for row in layer_rows:
        for outcome in ("all", row["outcome"]):
            grouped.setdefault((outcome, row["env_step"], row["layer"]), []).append(row)
    stats = []
    for (outcome, step, layer), rows in sorted(grouped.items()):
        stat = {"outcome": outcome, "env_step": step, "layer": layer, "n_runs": len(rows)}
        for key in ("wrist_share", "first_action_wrist_share", "wrist_mass", "agentview_mass"):
            values = [row[key] for row in rows]
            stat[key + "_mean"] = float(np.mean(values))
            stat[key + "_std"] = float(np.std(values))
        stats.append(stat)
    _write_csv(input_dir / "attention_layer_summary.csv", stats)

    metadata = {
        pathlib.PurePosixPath(row["attention_path"].replace("\\", "/")).name: row
        for row in summary.get("records", []) if row.get("attention_path")
    }
    fps = summary.get("config", {}).get("video_fps")
    valid_fps = isinstance(fps, (float, int)) and np.isfinite(fps) and fps > 0
    video_rows = []
    for record in records:
        meta = metadata.get(pathlib.Path(record["path"]).name, {})
        original_video = meta.get("video_path")
        video_name = pathlib.PurePosixPath(original_video.replace("\\", "/")).name if original_video else ""
        # One frame is recorded before each env.step, after warmup; frame index == env_step.
        for index, step in enumerate(record["steps"]):
            if review_steps[0] <= step <= review_steps[1]:
                video_rows.append({
                    "run": record["run"], "outcome": record["outcome"], "env_step": int(step),
                    "video_file": video_name, "video_frame_zero_based": int(step),
                    "video_time_seconds": float(step / fps) if valid_fps else None,
                    "video_fps": fps if valid_fps else None,
                    "wrist_share": float(record["share"][index].mean()),
                    "video_present_locally": bool(video_name and (input_dir / video_name).is_file()),
                })
    # Keep the file header even if the requested interval has no samples.
    video_path = input_dir / "attention_video_review.csv"
    if video_rows:
        _write_csv(video_path, video_rows)
    else:
        video_path.write_text(
            "run,outcome,env_step,video_file,video_frame_zero_based,video_time_seconds,"
            "video_fps,wrist_share,video_present_locally\n", encoding="utf-8",
        )
    outcome_counts = {
        outcome: sum(record["outcome"] == outcome for record in records)
        for outcome in sorted({"success", "failure", "unknown", *[record["outcome"] for record in records]})
    }
    return {
        "layer_count": layer_count,
        "focus_layers": [layer for layer in dict.fromkeys(focus_layers) if layer < layer_count],
        "unavailable_focus_layers": [layer for layer in focus_layers if layer >= layer_count],
        "outcome_run_counts": outcome_counts,
        "success_failure_comparison_available": outcome_counts["success"] > 0 and outcome_counts["failure"] > 0,
        "review_steps": list(review_steps), "video_review_rows": len(video_rows),
        "video_time_note": "Playback time = env_step / video_fps; not simulation or wall-clock time.",
        "statistics": "Equal weight per active run at each step; shading is population SD, not a confidence interval.",
        "outcome_note": "Unknown outcomes are kept separate; missing groups are not treated as zero attention.",
        "focus_note": "Layers and time window selected after inspecting the baseline; exploratory analysis only.",
    }


def _plot_details(input_dir, records, detail, review_steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = {"all": "#333333", "success": "#188264", "failure": "#c74939", "unknown": "#797979"}
    cohorts = {"all": records}
    for outcome in detail["outcome_run_counts"]:
        subset = [record for record in records if record["outcome"] == outcome]
        if subset:
            cohorts[outcome] = subset

    def curve(ax, subset, key, label, color, layer=None):
        steps, aligned = _aligned_values(subset, key)
        values = [item.mean(axis=1)[:, layer] if layer is not None else item.mean(axis=(1, 2)) for item in aligned]
        means = np.array([item.mean() for item in values])
        stds = np.array([item.std() for item in values])
        ax.plot(steps, means, label=label, color=color, linewidth=1.5)
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.12)

    def decorate(ax, title):
        ax.set(title=title, xlabel="environment step", ylabel="wrist / (agentview + wrist)", ylim=(0, 1))
        ax.axhline(0.5, color="#aaaaaa", linewidth=0.6, linestyle=":")
        # Highlight only the intersection, so a short rollout isn't stretched to step 45.
        low = max(review_steps[0], min(int(record["steps"][0]) for record in records))
        high = min(review_steps[1], max(int(record["steps"][-1]) for record in records))
        if low < high:
            ax.axvspan(low, high, color="#edb94f", alpha=0.12)
        ax.grid(alpha=0.2)

    for name, layers in (
        ("attention_layers_all.png", list(range(detail["layer_count"]))),
        ("attention_layers_focus.png", detail["focus_layers"]),
    ):
        columns = min(3, max(1, len(layers)))
        nrows = max(1, (len(layers) + columns - 1) // columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 3.2 * nrows), squeeze=False,
                                 constrained_layout=True)
        for ax, layer in zip(axes.flat, layers):
            for outcome, subset in cohorts.items():
                curve(ax, subset, "share", f"{outcome} (n={len(subset)})", colors.get(outcome), layer)
            decorate(ax, f"Layer {layer} (zero based)")
        for ax in list(axes.flat)[len(layers):]:
            ax.set_visible(False)
        if layers:
            axes.flat[0].legend(fontsize=8)
        fig.suptitle("Layer attention over time | mean +/- SD across active runs")
        fig.savefig(input_dir / name, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for outcome, subset in cohorts.items():
        if outcome == "all":
            continue
        label = f"{outcome} (n={len(subset)})"
        color = colors.get(outcome)
        curve(axes[0, 0], subset, "share", label, color)
        curve(axes[0, 1], subset, "first_share", label, color)
        curve(axes[1, 0], subset, "wrist", label + " wrist", color)
        steps, aligned = _aligned_values(subset, "agent")
        axes[1, 0].plot(steps, [item.mean() for item in aligned], color=color, linestyle="--",
                        label=label + " agentview")
        axes[1, 1].step(steps, [len(item) for item in aligned], where="mid", label=label, color=color)
    decorate(axes[0, 0], "All action tokens")
    decorate(axes[0, 1], "First action token")
    axes[1, 0].set(title="Absolute image attention mass", xlabel="environment step", ylabel="mass", ylim=(0, 1))
    axes[1, 1].set(title="Active runs at each step", xlabel="environment step", ylabel="runs", ylim=(0, None))
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    missing = [name for name in ("success", "failure") if not detail["outcome_run_counts"][name]]
    subtitle = " | No " + "/".join(missing) + " samples; comparison unavailable" if missing else ""
    fig.suptitle("Attention by outcome" + subtitle)
    fig.savefig(input_dir / "attention_by_outcome.png", dpi=160)
    plt.close(fig)


def _read_success_summary(input_dir: pathlib.Path) -> dict[str, Any]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        return {"available": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"available": True, **summary.get("aggregate", {})}


def _aligned_values(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Average each run once at each environment step; never concatenate episodes in time."""
    grouped: dict[int, list[np.ndarray]] = {}
    for record in records:
        for step, value in zip(record["steps"], record[key]):
            grouped.setdefault(int(step), []).append(value)
    steps = np.asarray(sorted(grouped), dtype=np.int32)
    return steps, [np.stack(grouped[int(step)]) for step in steps]


def _plot(input_dir: pathlib.Path, records: list[dict[str, Any]], groups: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    steps, aligned = _aligned_values(records, "share")
    means = np.asarray([np.mean(values) for values in aligned])
    stds = np.asarray([np.std(values.mean(axis=(1, 2))) for values in aligned])
    # [environment step, denoise step, layer], equal weight per run at each step.
    share = np.stack([values.mean(axis=0) for values in aligned])

    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for record in records:
        axes[0, 0].plot(record["steps"], record["share"].mean(axis=(1, 2)), alpha=0.25, linewidth=1)
    axes[0, 0].plot(steps, means, marker="o", label="wrist / (agentview + wrist)")
    axes[0, 0].fill_between(steps, means - stds, means + stds, alpha=0.2)
    axes[0, 0].set(xlabel="environment step", ylabel="attention share", ylim=(0, 1), title="Wrist share over rollout")
    axes[0, 0].grid(alpha=0.25)

    # Each record retains [denoise_step, layer]. Average one axis at a time so the
    # two heatmaps answer different questions instead of duplicating each other.
    denoise_values = share.mean(axis=2)
    denoise_image = axes[0, 1].pcolormesh(
        steps, np.arange(denoise_values.shape[1]), denoise_values.T,
        shading="nearest", vmin=0, vmax=1,
    )
    axes[0, 1].set(xlabel="environment step", ylabel="denoise step", title="Wrist share by denoise step")
    figure.colorbar(denoise_image, ax=axes[0, 1], label="wrist share")

    layer_values = share.mean(axis=1)
    layer_image = axes[1, 0].pcolormesh(
        steps, np.arange(layer_values.shape[1]), layer_values.T,
        shading="nearest", vmin=0, vmax=1,
    )
    axes[1, 0].set(xlabel="environment step", ylabel="layer (zero based)", title="Wrist share by layer")
    figure.colorbar(layer_image, ax=axes[1, 0], label="wrist share")

    # First action token in every sampled chunk, averaged within runs then across runs.
    first_mean = np.mean([record["first_share"].mean(axis=(0, 2)) for record in records], axis=0)
    axes[1, 1].plot(np.arange(first_mean.shape[0]), first_mean, marker="o")
    axes[1, 1].set(xlabel="denoise step", ylabel="wrist share", ylim=(0, 1), title="First action token")
    axes[1, 1].grid(alpha=0.25)
    _, aligned_mass = _aligned_values(records, "mean_mass")
    mean_mass = np.stack([values.mean(axis=(0, 1, 2)) for values in aligned_mass])
    labels = {"base_0_rgb": "agentview", "left_wrist_0_rgb": "wrist", "right_wrist_0_rgb": "right wrist (masked)"}
    for index, group in enumerate(groups):
        axes[0, 2].plot(steps, mean_mass[:, index], label=labels.get(group, group))
    axes[0, 2].set(xlabel="environment step", ylabel="mass over all keys", ylim=(0, 1), title="Absolute source mass")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.25)
    axes[1, 2].step(steps, [len(values) for values in aligned], where="mid")
    axes[1, 2].set(xlabel="environment step", ylabel="runs", title="Contributing runs (may decrease)")
    axes[1, 2].grid(alpha=0.25)
    figure.suptitle("LIBERO normal attention baseline\n" + ", ".join(groups))
    figure.savefig(input_dir / "attention_analysis.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=pathlib.Path)
    parser.add_argument("--focus-layers", nargs="+", type=int, default=[0, 8, 11, 12, 14, 17])
    parser.add_argument("--review-steps", nargs=2, type=int, default=[35, 45], metavar=("START", "END"))
    args = parser.parse_args()
    print(analyze(args.input_dir, focus_layers=tuple(args.focus_layers), review_steps=tuple(args.review_steps)))
