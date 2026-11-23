import pytest

from triage_core.cascade import Cascade, LevelThresholds
from triage_core.classify import Prediction, StaleModelError
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("urgent", "U", "u"), Label("not_urgent", "N", "n")))


class Fixed:
    """Returns a fixed prediction per text, counting how many texts it saw."""

    def __init__(self, name, table, cost=0.0, stale=False):
        self.name = name
        self.cache_key = name
        self.table = table
        self.cost = cost
        self.stale = stale
        self.seen = []

    def predict(self, texts, labels):
        if self.stale:
            raise StaleModelError("stale")
        self.seen += texts
        return [
            Prediction(self.table[t][0], {self.table[t][0]: self.table[t][1]}, cost_usd=self.cost)
            for t in texts
        ]


def test_accepts_confident_and_escalates_rest():
    classic = Fixed("classic", {"x": ("urgent", 0.95), "y": ("urgent", 0.6)})
    claude = Fixed("claude", {"y": ("not_urgent", 1.0)}, cost=0.001)
    out = Cascade([(classic, LevelThresholds(0.9)), (claude, None)]).predict(["x", "y"], LABELS)
    assert [(p.label, p.level) for p in out] == [("urgent", "classic"), ("not_urgent", "claude")]
    assert claude.seen == ["y"]


def test_asymmetric_threshold_escalates_not_urgent():
    classic = Fixed("classic", {"x": ("not_urgent", 0.9), "y": ("urgent", 0.9)})
    claude = Fixed("claude", {"x": ("urgent", 1.0)})
    thr = LevelThresholds(0.8, {"not_urgent": 0.95})
    out = Cascade([(classic, thr), (claude, None)]).predict(["x", "y"], LABELS)
    assert [(p.label, p.level) for p in out] == [("urgent", "claude"), ("urgent", "classic")]


def test_cost_accumulates_across_levels():
    jev = Fixed("jev", {"x": ("urgent", 0.1)}, cost=0.0001)
    claude = Fixed("claude", {"x": ("urgent", 1.0)}, cost=0.001)
    [p] = Cascade([(jev, LevelThresholds(0.9)), (claude, None)]).predict(["x"], LABELS)
    assert p.cost_usd == pytest.approx(0.0011)


def test_stale_level_is_skipped():
    classic = Fixed("classic", {}, stale=True)
    claude = Fixed("claude", {"x": ("urgent", 1.0)})
    [p] = Cascade([(classic, LevelThresholds(0.5)), (claude, None)]).predict(["x"], LABELS)
    assert p.level == "claude"


def test_last_level_must_accept_everything():
    with pytest.raises(ValueError, match="last level"):
        Cascade([(Fixed("c", {}), LevelThresholds(0.5))])


def test_thresholds_roundtrip():
    thr = LevelThresholds(0.7, {"not_urgent": 0.95})
    assert LevelThresholds.from_dict(thr.to_dict()) == thr
    assert thr.threshold("urgent") == 0.7
    assert thr.threshold("not_urgent") == 0.95
