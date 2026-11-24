import numpy as np
import pytest

from training.replay import ReplayClassifier, replay_cascade
from training.thresholds import (
    NEVER,
    apply_temperature_matrix,
    choose,
    fit_temperature,
    priority_candidates,
    simulate,
    sweep,
    to_level_thresholds,
)
from triage_core.cascade import LevelThresholds
from triage_core.classify import Prediction
from triage_core.labels import Label, LabelSet

IDS = ["urgent", "not_urgent"]


def test_simulate_accepts_and_escalates():
    level = np.array([[0.95, 0.05], [0.6, 0.4], [0.1, 0.9]])
    pred, lvl = simulate([level], np.array([1, 1, 0]), [np.array([0.9, 0.9])])
    assert pred.tolist() == [0, 1, 1]
    assert lvl.tolist() == [0, 1, 0]


def test_simulate_asymmetric_vector():
    level = np.array([[0.1, 0.9]])
    pred, lvl = simulate([level], np.array([0]), [np.array([0.8, 0.95])])
    assert (pred.tolist(), lvl.tolist()) == ([0], [1])


def test_sweep_picks_cheapest_feasible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    probs = np.where(np.eye(2)[y] == 1, 0.99, 0.01)
    probs[:20] = np.where(np.eye(2)[y[:20]] == 1, 0.1, 0.9)  # 20 mistakes at 0.9 confidence
    final = y.copy()
    candidates = [[np.full(2, t) for t in (0.5, 0.95, NEVER)]]
    rows = sweep(
        [probs],
        final,
        y,
        [0.0, 0.001],
        candidates,
        score=lambda yt, yp: float(np.mean(yt == yp)),
        constraint=lambda yt, yp: float(np.mean(yt == yp)) >= 0.99,
    )
    best = choose(rows)
    # 0.5 accepts the mistakes (fails), NEVER is feasible but costly: 0.95 escalates only 20
    assert best["thresholds"] == [[0.95, 0.95]]
    assert best["usd_per_1k"] == pytest.approx(0.1)
    assert best["coverage"] == [0.9, 0.1]


def test_priority_candidates_never_easier_for_not_urgent():
    for v in priority_candidates(IDS):
        assert v[1] >= v[0]


def test_fit_temperature_softens_overconfident_model():
    y = np.array([0] * 60 + [1] * 40)
    probs = np.tile([0.99, 0.01], (100, 1))
    t = fit_temperature(probs, y)
    assert t > 1.0
    assert apply_temperature_matrix(probs, t)[0, 0] < 0.99


def test_to_level_thresholds():
    assert to_level_thresholds(np.array([0.7, 0.95]), IDS) == LevelThresholds(
        0.7, {"not_urgent": 0.95}
    )
    assert to_level_thresholds(np.array([0.8, 0.8]), IDS) == LevelThresholds(0.8, {})


def test_replay_cascade_uses_thresholds(tmp_path):
    labels = LabelSet((Label("urgent", "U", "u"), Label("not_urgent", "N", "n")))
    classic = {
        "1": Prediction("not_urgent", {"urgent": 0.1, "not_urgent": 0.9}),
        "2": Prediction("urgent", {"urgent": 0.9, "not_urgent": 0.1}),
    }
    claude = {
        "1": Prediction("urgent", {"urgent": 1.0}, cost_usd=0.001),
        "2": Prediction("urgent", {"urgent": 1.0}, cost_usd=0.001),
    }
    config = {
        "order": ["classic", "claude"],
        "jev_temperature": 1.0,
        "levels": {"classic": {"default": 0.8, "per_label": {"not_urgent": 0.95}}},
    }
    cascade = replay_cascade(config, {"classic": classic, "claude": claude})
    out = cascade.predict(["1", "2"], labels)
    assert [(p.label, p.level) for p in out] == [("urgent", "claude"), ("urgent", "classic")]
    assert ReplayClassifier("x", classic).predict(["2"], labels)[0].label == "urgent"
