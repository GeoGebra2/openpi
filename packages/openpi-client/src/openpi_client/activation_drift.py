"""Versioned, NumPy-only paired diagnostics for the LIBERO activation experiment.

Distances are computed between corresponding tokens BEFORE reducing over tokens.
The server computes these summaries to avoid sending full hidden states over the wire.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 2
DIM_GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6), "gripper": slice(6, 7)}


def vector_metrics(a, b):
    """Compare last-axis vectors; b is the reference. Both-zero cosine is defined as 1."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or not a.shape[-1]:
        raise ValueError(f"Mismatched or empty vector shapes: {a.shape}, {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Non-finite activations in paired diagnostics")
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    product = norm_a * norm_b
    cosine = np.divide(np.sum(a * b, axis=-1), product, out=np.zeros_like(product), where=product > 0)
    cosine = np.where((norm_a == 0) & (norm_b == 0), 1.0, np.clip(cosine, -1.0, 1.0))
    difference = np.linalg.norm(a - b, axis=-1)
    return {
        "cosine": cosine,
        "rel_l2": difference / np.maximum(norm_b, 1e-8),
        "l2": difference,
        "rmse": difference / np.sqrt(a.shape[-1]),
        "norm_current": norm_b,
        "norm_delayed": norm_a,
        "nonzero_pair_fraction": ((norm_a > 0) & (norm_b > 0)).astype(np.float64),
    }


def _paired_mask(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError("Paired diagnostics require identical token validity/group masks")
    return a


def token_statistics(a, b, mask):
    """Summarize [..., token, width] pairs, retaining all leading layer/step axes.

    Process one layer at a time to bound float64 temporary memory. P90 is computed
    over token distances within each query, not over episodes or queries.
    """
    a, b = np.asarray(a), np.asarray(b)
    mask = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.ndim < 2 or mask.shape != (a.shape[-2],):
        raise ValueError("Invalid token activation or mask shapes")
    if not mask.any():
        return {}
    result = {}
    for index in np.ndindex(a.shape[:-2]):
        token_a = np.asarray(a[index][mask], dtype=np.float64)
        token_b = np.asarray(b[index][mask], dtype=np.float64)
        metrics = vector_metrics(token_a, token_b)
        reduced = {name: values.mean() for name, values in metrics.items()}
        reduced["cosine_distance_p90"] = np.quantile(1.0 - metrics["cosine"], 0.9)
        reduced["rel_l2_p90"] = np.quantile(metrics["rel_l2"], 0.9)
        reduced["valid_tokens"] = float(mask.sum())
        # Keep pooled diagnostics explicitly separate from the primary token metrics.
        pooled = vector_metrics(token_a.mean(axis=0), token_b.mean(axis=0))
        reduced.update({f"pooled_{name}": pooled[name] for name in ("cosine", "rel_l2")})
        for name, value in reduced.items():
            if name not in result:
                result[name] = np.empty(a.shape[:-2], dtype=np.float64)
            result[name][index] = value
    return result


def _store_tokens(out, name, a, b, mask):
    out.update({f"{name}_{key}": value for key, value in token_statistics(a, b, mask).items()})


def compute_drift(delayed, current):
    """Compute schema-v2 LIBERO metrics from unbatched, token-level model diagnostics."""
    result = {"siglip": {}, "gemma_prefix": {}, "action_expert": {}, "flow": {}}
    if delayed["siglip"].keys() != current["siglip"].keys():
        raise ValueError("Camera sets differ between paired diagnostics")
    for camera, a in delayed["siglip"].items():
        b = current["siglip"][camera]
        mask = _paired_mask(a["mask"], b["mask"])
        if mask.any():
            entry = {}
            _store_tokens(entry, "block_tokens", a["block_tokens"], b["block_tokens"], mask)
            result["siglip"][camera] = entry

    a, b = delayed["gemma_prefix"], current["gemma_prefix"]
    masks = {"all": _paired_mask(a["mask"], b["mask"])}
    if a["group_masks"].keys() != b["group_masks"].keys():
        raise ValueError("Prefix token groups differ")
    masks.update(
        {name: _paired_mask(value, b["group_masks"][name]) & masks["all"] for name, value in a["group_masks"].items()}
    )
    for group, mask in masks.items():
        if mask.any():
            entry = {}
            for name in ("layer_tokens", "final_tokens"):
                _store_tokens(entry, name, a[name], b[name], mask)
            result["gemma_prefix"][group] = entry

    a, b = delayed["action_expert"], current["action_expert"]
    valid = _paired_mask(a["mask"], b["mask"])
    action = _paired_mask(a["action_mask"], b["action_mask"])
    for group, mask in {"action": valid & action, "state": valid & ~action}.items():
        if mask.any():
            entry = {}
            _store_tokens(entry, "layer_tokens", a["layer_tokens"], b["layer_tokens"], mask)
            result["action_expert"][group] = entry

    a, b = np.asarray(delayed["flow"]["v_t"]), np.asarray(current["flow"]["v_t"])
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] < 7:
        raise ValueError("LIBERO flow must have matching [steps, horizon, dim >= 7] shapes")
    groups = {"": slice(0, 7), "_all_dims": slice(None), **{f"_{k}": v for k, v in DIM_GROUPS.items()}}
    if a.shape[-1] > 7:
        groups["_padding"] = slice(7, None)
    for group, dimensions in groups.items():
        metrics = vector_metrics(a[..., dimensions].reshape(a.shape[0], -1), b[..., dimensions].reshape(b.shape[0], -1))
        result["flow"].update({f"v_t{group}_{k}": v for k, v in metrics.items()})
    result["flow"].update({f"v_t_chunk_{k}": v for k, v in vector_metrics(a[..., :7], b[..., :7]).items()})
    return result


def compare_actions(diagnostic, ordinary, *, atol, rtol):
    """Report equivalence without hiding non-finite values or changing the tolerances."""
    a, b = np.asarray(diagnostic, dtype=np.float64), np.asarray(ordinary, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Invalid action arrays in diagnostics consistency check")
    difference = np.abs(a - b)
    return {
        "passed": bool(np.all(difference <= atol + rtol * np.abs(b))),
        "max_abs": float(difference.max()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "atol": float(atol),
        "rtol": float(rtol),
    }
