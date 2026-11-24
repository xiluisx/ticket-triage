import math

import numpy as np
import pytest

from training.metrics import (
    accuracy,
    bootstrap_ci,
    ece,
    kappa,
    macro_f1,
    macro_f1_present,
    p95,
    probs_matrix,
    recall_of,
)
from triage_core.classify import Prediction

Y = ["a", "a", "b", "b"]


def test_macro_f1_and_accuracy():
    assert macro_f1(Y, Y, ["a", "b"]) == 1.0
    assert accuracy(Y, ["a", "b", "b", "b"]) == 0.75


def test_recall_of_label():
    assert recall_of("a")(Y, ["a", "b", "b", "b"]) == 0.5


def test_ece_perfect_calibration_is_zero():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert ece(Y, probs, ["a", "b"]) == 0.0


def test_ece_overconfident_wrong():
    probs = np.array([[0.0, 1.0]] * 4)
    assert ece(Y, probs, ["a", "b"]) == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_and_is_deterministic():
    y_true = ["a"] * 50 + ["b"] * 50
    y_pred = ["a"] * 40 + ["b"] * 10 + ["b"] * 50
    lo, hi = bootstrap_ci(y_true, y_pred, accuracy, n_boot=200)
    assert lo <= accuracy(y_true, y_pred) <= hi
    assert (lo, hi) == bootstrap_ci(y_true, y_pred, accuracy, n_boot=200)


def test_kappa_and_p95():
    assert kappa(Y, Y) == 1.0
    assert p95([1.0] * 19 + [100.0]) == pytest.approx(5.95)
    assert np.isnan(p95([]))


def test_probs_matrix_orders_columns():
    preds = [Prediction("b", {"b": 0.7, "a": 0.3}), Prediction("a", {"a": 1.0})]
    assert probs_matrix(preds, ["a", "b"]).tolist() == [[0.3, 0.7], [1.0, 0.0]]


def test_macro_f1_present_ignores_absent_classes():
    assert macro_f1_present(["a", "a", "b"], ["a", "a", "b"]) == 1.0


def test_macro_f1_present_penalizes_predicting_absent_class():
    # truth has only "a"; predicting "z" lowers recall of "a" but "z" is not averaged
    assert macro_f1_present(["a", "a"], ["a", "z"]) == pytest.approx(2 / 3)


def test_recall_is_nan_without_positives():
    assert math.isnan(recall_of("urgent")(["not_urgent"], ["urgent"]))
    assert recall_of("urgent")(["urgent", "urgent"], ["urgent", "not_urgent"]) == 0.5


def test_bootstrap_ci_ignores_nan_resamples():
    lo, hi = bootstrap_ci(
        ["urgent"] + ["not_urgent"] * 9, ["urgent"] * 10, recall_of("urgent"), n_boot=200
    )
    assert lo == 1.0 and hi == 1.0
