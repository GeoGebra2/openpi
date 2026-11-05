"""Plot measured LIBERO trajectories and time-aligned clean/delay separation."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib

import numpy as np

try:
    from examples.libero.run_trajectory_delay import EXPERIMENT, SCHEMA_VERSION
except ImportError:
    from run_trajectory_delay import EXPERIMENT, SCHEMA_VERSION


STYLES = {"success": "-", "failure": "--"}
PAIR_FIELDS = ("seed", "task_suite_name", "task_id", "initial_state_id", "initial_state_sha256",
               "prompt", "control_hz", "warmup_steps")


def quaternion_distance_deg(a, b):
    """Shortest rotation angle; quaternion q and -q describe the same pose."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a, axis=-1, keepdims=True), np.linalg.norm(b, axis=-1, keepdims=True)
    if a.shape != b.shape or a.shape[-1] != 4 or np.any(na == 0) or np.any(nb == 0):
        raise ValueError("Invalid paired quaternions")
    dot = np.abs(np.sum((a / na) * (b / nb), axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))


def first_sustained_step(steps, distances, threshold_m, consecutive):
    if threshold_m <= 0 or consecutive < 1:
        raise ValueError("Positive separation threshold and consecutive sample count required")
    above = np.asarray(distances) >= threshold_m
    for index in range(len(above) - consecutive + 1):
        if above[index:index + consecutive].all() and np.all(np.diff(steps[index:index + consecutive]) == 1):
            return int(steps[index])
    return None


def load_trace(directory, record):
    path = directory / record["trace_path"]
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    n = len(trace["steps"])
    if n != record["steps"] + 1 or n < 1 or not np.array_equal(trace["steps"], np.arange(n)):
        raise ValueError(f"Invalid initial/terminal sample alignment: {path}")
    shapes = {"eef_pos": (n, 3), "eef_quat_xyzw": (n, 4), "gripper_qpos": (n, 2),
              "executed_actions": (n - 1, 7), "time_s": (n,),
              "object_pos": (n, len(trace["object_names"]), 3)}
    for key, shape in shapes.items():
        if trace[key].shape != shape or not np.all(np.isfinite(trace[key])):
            raise ValueError(f"Invalid {key} in {path}")
    if not np.allclose(trace["time_s"], trace["steps"] / record["control_hz"], rtol=0, atol=1e-9):
        raise ValueError(f"Simulator time does not match control steps: {path}")
    if not np.allclose(np.linalg.norm(trace["eef_quat_xyzw"], axis=1), 1, atol=1e-3):
        raise ValueError(f"Non-unit quaternion: {path}")
    q = trace["query_steps"]
    if q.ndim != 1 or len(q) != len(np.unique(q)) or np.any(q < 0) or np.any(q >= n - 1):
        raise ValueError(f"Invalid policy-query indices: {path}")
    if len(q) != len(trace["noise"]) or not np.all(np.isfinite(trace["noise"])):
        raise ValueError(f"Missing or non-finite shared-noise samples: {path}")
    return trace


def paired_metrics(clean_record, delayed_record, clean, delayed, threshold_m=0.02, consecutive=3):
    for key in PAIR_FIELDS:
        if clean_record.get(key) != delayed_record.get(key):
            raise ValueError(f"Cannot pair different {key}")
    if clean_record["repeat_index"] != delayed_record["repeat_index"]:
        raise ValueError("Cannot pair different repeat_index")
    if clean["start_sim_state"].size == 0 or clean["start_sim_state"].shape != delayed["start_sim_state"].shape:
        raise ValueError("Missing or different initial simulator state")
    if not np.allclose(clean["start_sim_state"], delayed["start_sim_state"], rtol=0, atol=1e-9):
        raise ValueError("Paired post-warmup simulator states differ")
    if not np.allclose(clean["eef_pos"][0], delayed["eef_pos"][0], rtol=0, atol=1e-7):
        raise ValueError("Paired initial end-effector positions differ")
    _, ci, di = np.intersect1d(clean["query_steps"], delayed["query_steps"], return_indices=True)
    if not np.array_equal(clean["noise"][ci], delayed["noise"][di]):
        raise ValueError("Paired queries did not share initial noise")
    steps, ci, di = np.intersect1d(clean["steps"], delayed["steps"], return_indices=True)
    position = np.linalg.norm(clean["eef_pos"][ci] - delayed["eef_pos"][di], axis=1)
    orientation = quaternion_distance_deg(clean["eef_quat_xyzw"][ci], delayed["eef_quat_xyzw"][di])
    fingers = np.linalg.norm(clean["gripper_qpos"][ci] - delayed["gripper_qpos"][di], axis=1)
    first = first_sustained_step(steps, position, threshold_m, consecutive)
    hz = clean_record["control_hz"]
    fields = {
        "condition": delayed_record["condition"], "repeat_index": delayed_record["repeat_index"],
        "clean_outcome": clean_record["outcome"], "delayed_outcome": delayed_record["outcome"],
        "clean_steps": clean_record["steps"], "delayed_steps": delayed_record["steps"],
        "overlap_samples": len(steps), "overlap_last_step": int(steps[-1]),
        "mean_position_distance_m": float(np.mean(position)), "max_position_distance_m": float(np.max(position)),
        "position_distance_at_overlap_end_m": float(position[-1]),
        "mean_orientation_distance_deg": float(np.mean(orientation)),
        "mean_gripper_qpos_distance_m": float(np.mean(fingers)),
        "first_sustained_separation_step": first,
        "first_sustained_separation_time_s": first / hz if first is not None else None,
        "clean_path_length_m": float(np.linalg.norm(np.diff(clean["eef_pos"], axis=0), axis=1).sum()),
        "delayed_path_length_m": float(np.linalg.norm(np.diff(delayed["eef_pos"], axis=0), axis=1).sum()),
        "clean_video": clean_record.get("video_path"), "delayed_video": delayed_record.get("video_path"),
        "clean_video_time_at_separation_s": first / clean_record["video_fps"] if first is not None else None,
        "delayed_video_time_at_separation_s": first / delayed_record["video_fps"] if first is not None else None,
    }
    return fields, {"steps": steps, "time_s": steps / hz, "position_m": position,
                    "orientation_deg": orientation, "gripper_qpos_m": fingers}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def label(condition):
    return condition.replace("observation_wrist_image", "wrist").replace("observation_image", "agentview")


def limits_for(traces):
    positions = np.concatenate([trace["eef_pos"] for trace in traces])
    centre = (positions.min(axis=0) + positions.max(axis=0)) / 2
    radius = max(float(np.ptp(positions, axis=0).max()) * 0.55, 0.02)
    return [(float(value - radius), float(value + radius)) for value in centre]


def style_3d(ax, limits):
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)",
           xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-55)


def path_3d(ax, trace, color, linestyle, *, alpha=1, name=None):
    xyz = trace["eef_pos"]
    ax.plot(*xyz.T, color=color, linestyle=linestyle, linewidth=1.3, alpha=alpha, label=name)
    ax.scatter(*xyz[0], color=color, marker="o", s=15, alpha=alpha)
    ax.scatter(*xyz[-1], color=color, marker="X", s=28, alpha=alpha)


def overview_plots(entries, colors, directory, title):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    traces = [trace for _, trace in entries]
    limits = limits_for(traces)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for record, trace in entries:
        path_3d(ax, trace, colors[record["condition"]], STYLES[record["outcome"]], alpha=0.55)
    style_3d(ax, limits)
    handles = [Line2D([0], [0], color=color, label=label(condition)) for condition, color in colors.items()]
    handles.extend([Line2D([0], [0], color="0.3", linestyle=style, label=outcome)
                    for outcome, style in STYLES.items()])
    handles.extend([Line2D([0], [0], color="0.3", linestyle="", marker=marker, label=text)
                    for marker, text in (("o", "start"), ("X", "terminal sample"))])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title(title + "\nMeasured EEF trajectories; each line is one episode")
    fig.subplots_adjust(right=0.78, top=0.88)
    fig.savefig(directory / "trajectories_3d.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (x, y) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        for record, trace in entries:
            xyz = trace["eef_pos"]
            color = colors[record["condition"]]
            ax.plot(xyz[:, x], xyz[:, y], color=color, linestyle=STYLES[record["outcome"]], alpha=0.55)
            ax.scatter(xyz[-1, x], xyz[-1, y], color=color, marker="X", s=15)
        ax.set(xlabel=f"World {'XYZ'[x]} (m)", ylabel=f"World {'XYZ'[y]} (m)",
               xlim=limits[x], ylim=limits[y])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    fig.suptitle(title + " | XY / XZ / YZ projections")
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(directory / "trajectory_projections.png", dpi=160)
    plt.close(fig)


def pair_plot(clean_record, delayed_record, clean, delayed, fields, series, colors, directory, threshold_m):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for record, trace in ((clean_record, clean), (delayed_record, delayed)):
        path_3d(ax, trace, colors[record["condition"]], STYLES[record["outcome"]],
                name=f"{label(record['condition'])}: {record['outcome']}")
    for index, name in enumerate(clean["object_names"]):
        point = clean["object_pos"][0, index]
        ax.scatter(*point, color="0.45", marker="s", s=18)
        ax.text(*point, str(name), fontsize=6)
    all_positions = [clean["eef_pos"], delayed["eef_pos"], clean["object_pos"][0]]
    style_3d(ax, limits_for([{"eef_pos": np.concatenate(all_positions)}]))
    ax.set_title("EEF paths; o=start, X=end, squares=initial object centres", fontsize=9)
    ax.legend(fontsize=8)
    first = fields["first_sustained_separation_step"]
    if first is not None:
        for record, trace in ((clean_record, clean), (delayed_record, delayed)):
            point = trace["eef_pos"][first]
            ax.scatter(*point, s=95, marker="*", color=colors[record["condition"]], edgecolors="black")
        p, q = clean["eef_pos"][first], delayed["eef_pos"][first]
        ax.plot(*np.stack([p, q]).T, color="black", linestyle=":")
    distance_ax = fig.add_subplot(2, 2, 2)
    distance_ax.plot(series["time_s"], series["position_m"] * 100, color=colors[delayed_record["condition"]])
    distance_ax.axhline(threshold_m * 100, color="0.5", linestyle=":", label="separation threshold")
    distance_ax.set(xlabel="Environment time after warmup (s)", ylabel="Position separation (cm)",
                    title="Same environment step; common time interval only", ylim=(0, None))
    if first is not None:
        distance_ax.axvline(fields["first_sustained_separation_time_s"], color="black", linestyle="--",
                            label=f"first sustained separation: step {first}")
    distance_ax.legend(fontsize=8)
    rotation_ax = fig.add_subplot(2, 2, 3)
    rotation_ax.plot(series["time_s"], series["orientation_deg"], color=colors[delayed_record["condition"]])
    rotation_ax.set(xlabel="Environment time after warmup (s)", ylabel="EEF rotation separation (degrees)",
                    title="Measured orientation; shortest quaternion angle", ylim=(0, None))
    gripper_ax = fig.add_subplot(2, 2, 4)
    for record, trace in ((clean_record, clean), (delayed_record, delayed)):
        # Both finger coordinates are shown; neither is assumed to be a grasp-success signal.
        for index in range(2):
            gripper_ax.plot(trace["time_s"], trace["gripper_qpos"][:, index] * 1000,
                            color=colors[record["condition"]], linestyle=STYLES[record["outcome"]],
                            alpha=1 if index == 0 else 0.45,
                            label=f"{label(record['condition'])} finger {index}")
    gripper_ax.set(xlabel="Environment time after warmup (s)", ylabel="Finger joint position (mm)",
                   title="Measured fingers; faint line = finger 1")
    gripper_ax.legend(fontsize=8)
    for axis in (distance_ax, rotation_ax, gripper_ax):
        axis.grid(alpha=0.25)
    fig.suptitle(f"{clean_record['task_suite_name']} task {clean_record['task_id']} | "
                 f"repeat {delayed_record['repeat_index']:03d} | {label(delayed_record['condition'])} vs clean\n"
                 "Stars mark a positional threshold crossing, not a proven failure cause", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / f"pair_{delayed_record['condition']}_repeat_{delayed_record['repeat_index']:03d}.png", dpi=140)
    plt.close(fig)


def separation_plot(pairs, colors, directory):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    grouped = collections.defaultdict(list)
    for fields, series in pairs:
        grouped[(fields["condition"], fields["delayed_outcome"])].append(series)
    for (condition, outcome), samples in grouped.items():
        longest = max(samples, key=lambda sample: len(sample["time_s"]))
        n = len(longest["time_s"])
        values = np.full((len(samples), n), np.nan)
        for index, sample in enumerate(samples):
            values[index, :len(sample["position_m"])] = sample["position_m"] * 100
        q25, median, q75 = np.nanpercentile(values, (25, 50, 75), axis=0)
        axes[0].plot(longest["time_s"], median, color=colors[condition], linestyle=STYLES[outcome],
                     label=f"{label(condition)} | delayed {outcome} (pairs={len(samples)})")
        axes[0].fill_between(longest["time_s"], q25, q75, color=colors[condition], alpha=0.12)
        axes[1].plot(longest["time_s"], np.isfinite(values).sum(axis=0), color=colors[condition],
                     linestyle=STYLES[outcome])
    axes[0].set(ylabel="Paired EEF separation (cm)", ylim=(0, None),
                title="Median and 25-75% between-pair range (not a confidence interval)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Environment time after common warmup (s)", ylabel="Available pairs", ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Position divergence at matched steps | curves stop when either episode ends")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "separation_vs_time.png", dpi=160)
    plt.close(fig)


def interactive_plot(entries, colors, directory):
    from matplotlib.colors import to_hex
    import plotly.graph_objects as go

    fig = go.Figure()
    for record, trace in entries:
        xyz = trace["eef_pos"]
        name = f"{label(record['condition'])} | {record['outcome']} | repeat {record['repeat_index']:03d}"
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines+markers", name=name,
            line={"color": to_hex(colors[record["condition"]]), "width": 4,
                  "dash": "solid" if record["outcome"] == "success" else "dash"},
            marker={"size": 2}, customdata=np.column_stack([trace["steps"], trace["time_s"]]),
            hovertemplate="step=%{customdata[0]:.0f}<br>time=%{customdata[1]:.2f}s<br>"
                          "X=%{x:.4f}m<br>Y=%{y:.4f}m<br>Z=%{z:.4f}m<extra>%{fullData.name}</extra>",
        ))
    limits = limits_for([trace for _, trace in entries])
    fig.update_layout(title="Measured EEF trajectories | color=condition, solid=success, dashed=failure",
                      scene={"xaxis": {"title": "World X (m)", "range": limits[0]},
                             "yaxis": {"title": "World Y (m)", "range": limits[1]},
                             "zaxis": {"title": "World Z (m)", "range": limits[2]}, "aspectmode": "cube"})
    fig.write_html(str(directory / "trajectories_3d.html"), include_plotlyjs=True, full_html=True, auto_open=False)


def analyze(path, *, threshold_cm=2.0, consecutive=3, pair_limit=30, interactive=False, output_dir=None):
    if not np.isfinite(threshold_cm) or threshold_cm <= 0 or consecutive < 1 or pair_limit < 0:
        raise ValueError("Invalid threshold, consecutive count, or pair limit")
    if interactive:
        try:
            __import__("plotly.graph_objects")
        except ImportError as error:
            raise RuntimeError("Interactive HTML requires plotly; install plotly<6 on Python 3.8, or omit --interactive") from error
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = pathlib.Path(path)
    summary_path = path / "summary.json" if path.is_dir() else path
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != EXPERIMENT or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Requires a run_trajectory_delay schema-v1 summary; old activation/video files lack full trajectories")
    directory = pathlib.Path(output_dir) if output_dir else summary_path.parent / "trajectory_analysis"
    directory.mkdir(parents=True, exist_ok=True)
    conditions = [condition["name"] for condition in summary["conditions"]]
    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(conditions)}
    entries, lookup, seen, excluded = [], {}, set(), []
    for record in summary["records"]:
        key = (record["condition"], record["repeat_index"])
        if key in seen or key[0] not in conditions:
            raise ValueError(f"Duplicate or unknown condition/repeat: {key}")
        seen.add(key)
        if record["outcome"] not in STYLES:
            excluded.append({"condition": key[0], "repeat_index": key[1], "reason": record["outcome"],
                             "error": record.get("error")})
            continue
        trace = load_trace(summary_path.parent, record)
        entries.append((record, trace))
        lookup[key] = (record, trace)
    aggregates = []
    for name in conditions:
        counts = collections.Counter(record["outcome"] for record in summary["records"] if record["condition"] == name)
        completed = counts["success"] + counts["failure"]
        aggregates.append({"condition": name, "successes": counts["success"], "failures": counts["failure"],
                           "errors": counts["error"], "interrupted": counts["interrupted"],
                           "missing_runs": summary["config"]["repeats_per_setting"] - sum(counts.values()),
                           "completed": completed, "success_rate": counts["success"] / completed if completed else None})
    paired, pair_details, unpaired = [], [], []
    plotted = collections.Counter()
    for record, trace in entries:
        if record["condition"] == "clean":
            continue
        reference = lookup.get(("clean", record["repeat_index"]))
        if reference is None:
            unpaired.append({"condition": record["condition"], "repeat_index": record["repeat_index"],
                             "reason": "missing completed clean episode"})
            continue
        clean_record, clean = reference
        fields, series = paired_metrics(clean_record, record, clean, trace, threshold_cm / 100, consecutive)
        paired.append((fields, series))
        pair_details.append(fields)
        if plotted[record["condition"]] < pair_limit:
            pair_plot(clean_record, record, clean, trace, fields, series, colors, directory, threshold_cm / 100)
            plotted[record["condition"]] += 1
    if entries:
        config = summary["config"]
        title = f"{config['task_suite_name']} task {config['task_id']} | initial state {config['initial_state_id']}"
        overview_plots(entries, colors, directory, title)
        if interactive:
            interactive_plot(entries, colors, directory)
    if paired:
        separation_plot(paired, colors, directory)
    write_csv(directory / "condition_outcomes.csv", aggregates)
    write_csv(directory / "paired_trajectory_metrics.csv", pair_details)
    time_rows = []
    for fields, series in paired:
        for index, step in enumerate(series["steps"]):
            time_rows.append({"condition": fields["condition"], "repeat_index": fields["repeat_index"],
                              "clean_outcome": fields["clean_outcome"], "delayed_outcome": fields["delayed_outcome"],
                              "step": int(step), **{key: float(values[index]) for key, values in series.items()
                                                   if key != "steps"}})
    write_csv(directory / "paired_distance_by_step.csv", time_rows)
    report = {
        "experiment": EXPERIMENT, "source_summary": str(summary_path), "conditions": aggregates,
        "pairs": pair_details, "excluded_runs": excluded, "unpaired_runs": unpaired,
        "separation_rule": {"threshold_cm": threshold_cm, "consecutive_samples": consecutive,
                            "meaning": "first sustained positional separation, not onset or cause of failure"},
        "interpretation": [
            "EEF pose is simulator ground truth; action commands are not positions.",
            "Clean and delayed episodes execute independently with matched reset state and per-step noise.",
            "Compare only shared environment time; no resampling, endpoint padding or phase alignment.",
            "Aggregate bands are between-pair IQR, not confidence intervals; contributing pairs change over time.",
            "Success is the environment task predicate; gripper closure and EEF endpoints alone do not establish grasp success.",
            "Object positions are root-body centres, not necessarily grasp targets.",
        ],
    }
    target = directory / "trajectory_analysis.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, help="Run directory or summary.json")
    parser.add_argument("--threshold-cm", type=float, default=2.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--pair-limit", type=int, default=30, help="Maximum pair figures per delay condition; metrics use all")
    parser.add_argument("--interactive", action="store_true", help="Also produce self-contained Plotly HTML")
    parser.add_argument("--output-dir", type=pathlib.Path)
    print(analyze(**vars(parser.parse_args())))
