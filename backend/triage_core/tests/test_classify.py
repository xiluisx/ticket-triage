from triage_core.classify import Prediction, short_hash


def test_confidence_is_prob_of_label():
    p = Prediction("a", {"a": 0.7, "b": 0.3})
    assert p.confidence == 0.7


def test_confidence_missing_label_is_zero():
    assert Prediction("c", {"a": 1.0}).confidence == 0.0


def test_short_hash_is_stable_and_order_independent():
    assert short_hash({"x": 1, "y": [1, 2]}) == short_hash({"y": [1, 2], "x": 1})
    assert len(short_hash("a")) == 12
    assert short_hash("a") != short_hash("b")


def test_prediction_skipped_defaults_empty_and_is_tuple():
    assert Prediction("a", {"a": 1.0}).skipped == ()
    p = Prediction("a", {"a": 1.0}, skipped=["classic:stale"])
    assert p.skipped == ("classic:stale",)
    assert p == Prediction("a", {"a": 1.0}, skipped=("classic:stale",))
