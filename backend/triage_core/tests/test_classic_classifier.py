import pytest

from triage_core.classifiers.classic import ClassicClassifier
from triage_core.classify import StaleModelError
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("a", "A", "first"), Label("b", "B", "second")))


class FakeModel:
    classes_ = ["a", "b"]

    def predict_proba(self, texts):
        return [[0.9, 0.1] if "alpha" in t else [0.2, 0.8] for t in texts]


def test_predict_returns_argmax_and_probs():
    clf = ClassicClassifier(FakeModel(), LABELS.labels_hash(), "v1")
    preds = clf.predict(["alpha text", "beta text"], LABELS)
    assert [p.label for p in preds] == ["a", "b"]
    assert preds[0].probs == {"a": 0.9, "b": 0.1}
    assert preds[0].cost_usd == 0.0
    assert preds[0].latency_ms >= 0.0


def test_cache_key_includes_version():
    assert ClassicClassifier(FakeModel(), "h", "v7").cache_key == "classic-v7-h"


def test_stale_labels_raise():
    clf = ClassicClassifier(FakeModel(), "other-hash", "v1")
    with pytest.raises(StaleModelError):
        clf.predict(["alpha"], LABELS)


def test_empty_input():
    assert ClassicClassifier(FakeModel(), LABELS.labels_hash(), "v1").predict([], LABELS) == []


def test_cache_key_differs_across_label_sets_with_same_version():
    other = LabelSet((Label("x", "X", "x"), Label("y", "Y", "y")))
    a = ClassicClassifier(FakeModel(), LABELS.labels_hash(), "v1")
    b = ClassicClassifier(FakeModel(), other.labels_hash(), "v1")
    assert a.cache_key != b.cache_key
