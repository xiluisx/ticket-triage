"""Fairness gate: per-group metrics on the golden set and a paired-bootstrap gap test.

Light on purpose (numpy/pandas/sklearn only) so the CI quality gate can import it.
Output format: `fairness.json` schema v2 (contracts spec, section 6).

Rule: the challenger fails when the 2.5th percentile of
    delta = gap_challenger - gap_champion
over paired bootstrap resamples of the golden set is > 0 (significantly less fair).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from training.golden import TAG_GROUPS
from training.metrics import bootstrap_ci, macro_f1_present, recall_of

MIN_N = 5
MIN_POSITIVES = 5
N_BOOT = 2000
SEED = 0
MIN_VALID_FRACTION = 0.9
SPECS = {
    "category": ("macro_f1_present", None),
    "priority": ("recall_urgent", "urgent"),
}
METRICS: dict[str, Callable] = {
    "macro_f1_present": macro_f1_present,
    "recall_urgent": recall_of("urgent"),
}


def tag_masks(tags: Sequence[str | None]) -> dict[str, np.ndarray]:
    series = pd.Series(list(tags), dtype=object).fillna("").astype(str)
    masks = {tag: series.str.contains(tag, regex=False).to_numpy() for tag in TAG_GROUPS}
    masks["No tag"] = (series == "").to_numpy()
    return masks


def group_value(
    truth: np.ndarray,
    pred: np.ndarray,
    metric: Callable,
    min_n: int,
    positive: str | None,
    min_positives: int,
) -> tuple[float | None, int, int | None]:
    n = len(truth)
    n_pos = int(np.sum(truth == positive)) if positive is not None else None
    if n < min_n or (n_pos is not None and n_pos < min_positives):
        return None, n, n_pos
    return float(metric(truth, pred)), n, n_pos


def per_group(
    truth,
    pred,
    masks: Mapping[str, Sequence[bool]],
    metric: Callable,
    min_n: int = MIN_N,
    positive: str | None = None,
    min_positives: int = MIN_POSITIVES,
    n_boot: int = 1000,
) -> dict:
    truth, pred = np.asarray(truth), np.asarray(pred)
    out = {}
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        value, n, n_pos = group_value(truth[m], pred[m], metric, min_n, positive, min_positives)
        if value is None:
            out[name] = {"value": None, "lo": None, "hi": None, "n": n, "n_pos": n_pos}
            continue
        lo, hi = bootstrap_ci(truth[m], pred[m], metric, n_boot=n_boot)
        out[name] = {"value": value, "lo": lo, "hi": hi, "n": n, "n_pos": n_pos}
    return out


def gap(groups: Mapping[str, Mapping]) -> float | None:
    """max - min over groups with a value; None when fewer than 2 groups qualify."""
    values = [g["value"] for g in groups.values() if g["value"] is not None]
    return float(max(values) - min(values)) if len(values) >= 2 else None


def _gap_of(truth, pred, masks, metric, positive, min_n, min_positives) -> float | None:
    values = {
        name: {"value": group_value(truth[m], pred[m], metric, min_n, positive, min_positives)[0]}
        for name, m in masks.items()
    }
    return gap(values)


def _skipped(reason: str, n_boot: int, seed: int) -> dict:
    return {
        "status": "skipped",
        "reason": reason,
        "delta_gap": None,
        "ci95": None,
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_delta(
    truth: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    masks: Mapping[str, np.ndarray],
    metric: Callable,
    positive: str | None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    min_n: int = MIN_N,
    min_positives: int = MIN_POSITIVES,
) -> dict:
    masks = {k: np.asarray(v, dtype=bool) for k, v in masks.items()}
    args = (metric, positive, min_n, min_positives)
    g_champ = _gap_of(truth, champion, masks, *args)
    g_chall = _gap_of(truth, challenger, masks, *args)
    if g_champ is None or g_chall is None:
        return _skipped("gap not computable", n_boot, seed)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(truth), size=len(truth))
        sub = {k: v[idx] for k, v in masks.items()}
        gc = _gap_of(truth[idx], champion[idx], sub, *args)
        gx = _gap_of(truth[idx], challenger[idx], sub, *args)
        if gc is not None and gx is not None:
            deltas.append(gx - gc)
    if len(deltas) < MIN_VALID_FRACTION * n_boot:
        return _skipped("too many resamples without a computable gap", n_boot, seed)
    lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    return {
        "status": "failed" if lo > 0 else "passed",
        "delta_gap": float(g_chall - g_champ),
        "ci95": [lo, hi],
        "n_boot": n_boot,
        "seed": seed,
        "n_valid": len(deltas),
    }


def _model_block(version, truth, pred, masks, metric, positive, n_boot) -> dict:
    groups = per_group(truth, pred, masks, metric, positive=positive, n_boot=n_boot)
    return {
        "version": version,
        "gap": gap(groups),
        "per_group": {
            name: {
                "n": g["n"],
                "n_pos": g["n_pos"],
                "value": g["value"],
                "ci95": None if g["value"] is None else [g["lo"], g["hi"]],
            }
            for name, g in groups.items()
        },
    }


def build_report(
    task: str,
    truth: Sequence[str],
    tags: Sequence[str | None],
    challenger: tuple[str, Sequence[str]],
    champion: tuple[str, Sequence[str]] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """fairness.json v2 for one task on the golden set. Models are (version, predicted labels)."""
    metric_name, positive = SPECS[task]
    metric = METRICS[metric_name]
    truth_arr = np.asarray(truth)
    masks = tag_masks(tags)
    chall = np.asarray(challenger[1])
    models = {
        "challenger": _model_block(challenger[0], truth_arr, chall, masks, metric, positive, n_boot)
    }
    if champion is None:
        gate = _skipped("no champion", n_boot, seed)
    else:
        champ = np.asarray(champion[1])
        models["champion"] = _model_block(
            champion[0], truth_arr, champ, masks, metric, positive, n_boot
        )
        gate = paired_delta(truth_arr, champ, chall, masks, metric, positive, n_boot, seed)
    return {
        "schema_version": 2,
        "task": task,
        "eval_set": "golden",
        "dimension": "tag",
        "metric": metric_name,
        "min_positives": MIN_POSITIVES,
        "models": models,
        "gate": gate,
    }
