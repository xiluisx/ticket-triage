"""Evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score

from triage_core.classify import Prediction

Metric = Callable[[Sequence[str], Sequence[str]], float]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def recall_of(label: str) -> Metric:
    def metric(y_true, y_pred) -> float:
        return float(recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0))

    return metric


def ece(
    y_true: Sequence[str], probs: np.ndarray, classes: Sequence[str], n_bins: int = 10
) -> float:
    classes = list(classes)
    y_idx = np.array([classes.index(y) for y in y_true])
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    values = [metric(y_true[i], y_pred[i]) for i in idx]
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def kappa(a: Sequence[str], b: Sequence[str]) -> float:
    return float(cohen_kappa_score(a, b))


def p95(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if len(values) else float("nan")


def probs_matrix(preds: Sequence[Prediction], classes: Sequence[str]) -> np.ndarray:
    return np.array([[p.probs.get(c, 0.0) for c in classes] for p in preds])
