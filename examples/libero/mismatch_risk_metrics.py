"""Geometric state correspondence and small, dependency-free risk models."""

from __future__ import annotations

import numpy as np


def rotation_distance(first, second):
    """SO(3) angle in radians; xyzw quaternion signs represent the same rotation."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("Expected matching xyzw quaternion arrays")
    norms = [np.linalg.norm(value, axis=-1, keepdims=True) for value in (first, second)]
    if any(np.any(~np.isfinite(value)) or np.any(value < 1e-8) for value in norms):
        raise ValueError("Non-finite or zero quaternion")
    dots = np.sum(first / norms[0] * second / norms[1], axis=-1)
    return 2 * np.arccos(np.clip(np.abs(dots), 0, 1))


def components(trace):
    current = np.asarray(trace["query_current_states"], dtype=float)
    source = np.asarray(trace["query_image_source_states"], dtype=float)
    if current.shape != source.shape or current.shape != (len(trace["query_steps"]), 8):
        raise ValueError("Missing or mismatched query-state provenance")
    values = np.column_stack((
        np.linalg.norm(current[:, :3] - source[:, :3], axis=1),
        rotation_distance(trace["query_current_quats"], trace["query_image_source_quats"]),
        np.linalg.norm(current[:, 6:] - source[:, 6:], axis=1),
    ))
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite state mismatch")
    return values


def calibrate_scales(traces):
    values = []
    for trace in traces:
        p, r, g = trace["eef_pos"], trace["eef_quat_xyzw"], trace["gripper_qpos"]
        if len(p) > 1:
            values.append(np.column_stack((np.linalg.norm(np.diff(p, axis=0), axis=1),
                                            rotation_distance(r[1:], r[:-1]),
                                            np.linalg.norm(np.diff(g, axis=0), axis=1))))
    if not values:
        raise ValueError("Clean trajectories have no state increments")
    # A scale is a unit conversion, not a claimed failure/safety threshold.
    floors = np.array([0.001, 0.005, 0.001])  # metre, radian, metre
    scales = np.maximum(np.quantile(np.concatenate(values), 0.75, axis=0), floors)
    return {"position_m": float(scales[0]), "rotation_rad": float(scales[1]),
            "gripper_m": float(scales[2]), "quantile": 0.75, "floors": floors.tolist(),
            "threshold_tau": 1.0}


def exposure(trace, calibration, hz):
    parts = components(trace)
    scales = np.array([calibration[name] for name in ("position_m", "rotation_rad", "gripper_m")])
    mismatch = np.linalg.norm(parts / scales, axis=1)
    query = np.asarray(trace["query_steps"], dtype=int)
    active = np.asarray(trace["query_delay_steps"]) > 0
    # A query's exposure lasts only while its actions are actually executed.
    durations = np.diff(np.r_[query, len(trace["executed_actions"])]) / hz
    if np.any(durations <= 0) or len(active) != len(query):
        raise ValueError("Invalid executed query intervals")
    if np.any(mismatch[~active] > 1e-5):
        raise ValueError("Current image was paired with a different source state")
    mismatch[~active] = 0
    over = active & (mismatch > calibration["threshold_tau"])
    longest, streak = 0.0, 0.0
    for enabled, duration in zip(over, durations):
        streak = streak + duration if enabled else 0.0
        longest = max(longest, streak)
    result = {
        "m_peak": float(mismatch.max()) if len(mismatch) else 0.0,
        "m_area_s": float(mismatch @ durations),
        "over_tau_s": float(durations[over].sum()), "longest_over_tau_s": float(longest),
        "exposure_s": float(durations[active].sum()),
        "position_peak_m": float(parts[:, 0].max()) if len(parts) else 0.0,
        "rotation_peak_rad": float(parts[:, 1].max()) if len(parts) else 0.0,
        "gripper_peak_m": float(parts[:, 2].max()) if len(parts) else 0.0,
    }
    return result, mismatch, parts


TIMING_FEATURES = ("start_frac", "end_frac", "duration_frac", "max_gap_frac", "window_count", "reference_s")
MODEL_FEATURES = {
    "timing_delay": ("delay_600", *TIMING_FEATURES),
    "mismatch_time": ("log_peak", "log_area", "longest_over_tau_s", *TIMING_FEATURES),
    "mismatch_time_delay": ("log_peak", "log_area", "longest_over_tau_s", "delay_600", *TIMING_FEATURES),
}


def feature_values(row):
    return {**row, "delay_600": row["effective_delay_ms"] / 600,
            "log_peak": float(np.log1p(row["m_peak"])),
            "log_area": float(np.log1p(row["m_area_s"]))}


def design(rows, names):
    return np.asarray([[feature_values(row)[name] for name in names] for row in rows], dtype=float)


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def fit_logistic(rows, names, ridge=2.0):
    """Fixed ridge penalty, training-only scaling; no validation-based tuning."""
    y = np.asarray([row["failure"] for row in rows], dtype=float)
    if len(np.unique(y)) < 2:
        return None
    x = design(rows, names)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1, scale)
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(x.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))
    penalty = np.r_[1e-6, np.repeat(ridge, len(names))]

    def objective(b):
        logits = x @ b
        return float(np.sum(np.logaddexp(0, logits) - y * logits) + 0.5 * np.sum(penalty * b * b))

    converged = False
    for _ in range(100):
        prob = sigmoid(x @ beta)
        weight = np.maximum(prob * (1 - prob), 1e-7)
        gradient = x.T @ (prob - y) + penalty * beta
        hessian = x.T @ (weight[:, None] * x) + np.diag(penalty)
        delta = np.linalg.solve(hessian, gradient)
        step, old = 1.0, objective(beta)
        while step > 1e-6 and objective(beta - step * delta) > old:
            step *= 0.5
        beta -= step * delta
        if np.max(np.abs(step * delta)) < 1e-7:
            converged = True
            break
    return {"features": list(names), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": beta.tolist(), "ridge": ridge, "converged": converged}


def predict(model, rows):
    x = (design(rows, model["features"]) - model["mean"]) / model["scale"]
    return sigmoid(model["coefficients"][0] + x @ np.asarray(model["coefficients"])[1:])


def auc(y, probability):
    """Pairwise AUROC with correct half credit for ties (small pilot samples)."""
    y = np.asarray(y, dtype=bool)
    if not np.any(y) or np.all(y):
        return None
    difference = probability[y, None] - probability[~y][None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def scores(rows, probability):
    if not rows:
        return None
    y = np.asarray([row["failure"] for row in rows], dtype=float)
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return {"n": len(y), "failures": int(y.sum()), "brier": float(np.mean((probability - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "auroc": auc(y, np.asarray(probability))}


def grouped_brier_improvement(rows, probability, baseline, seed=7, repeats=300):
    """Bootstrap whole held-out pairs, preserving all conditions of each seed."""
    if not rows:
        return None
    pair_ids = sorted({row["pair_index"] for row in rows})
    if len(pair_ids) < 3:
        return None
    groups = [np.array([i for i, row in enumerate(rows) if row["pair_index"] == pair]) for pair in pair_ids]
    y = np.array([row["failure"] for row in rows])
    differences = (baseline - y) ** 2 - (probability - y) ** 2
    rng = np.random.default_rng(seed)
    samples = [float(np.mean(differences[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]))
               for _ in range(repeats)]
    return {"estimate": float(differences.mean()), "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)), "bootstrap_unit": "held-out pair (all conditions)",
            "bootstrap_repeats": repeats}
