import pytest

from training.cache import PredictionCache, RunFailedError, cache_for, confirm_cost, predict_cached
from training.jsonl import read_jsonl
from triage_core.classify import Prediction
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("a", "A", "a"), Label("b", "B", "b")))


class Counting:
    name = "fake"
    cache_key = "fake-1"

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)

    def predict(self, texts, labels):
        self.calls += 1
        if texts[0] in self.fail_on:
            raise RuntimeError("boom")
        return [Prediction("a", {"a": 1.0}, cost_usd=0.01)]


def test_second_run_hits_cache(tmp_path):
    cache = PredictionCache(tmp_path / "c.jsonl")
    clf = Counting()
    out = predict_cached(clf, cache, ["1", "2"], ["x", "y"], LABELS)
    assert set(out) == {"1", "2"} and clf.calls == 2
    reloaded = PredictionCache(tmp_path / "c.jsonl")
    predict_cached(clf, reloaded, ["1", "2"], ["x", "y"], LABELS)
    assert clf.calls == 2
    assert reloaded.get("1") == out["1"]


def test_failures_are_logged_and_tolerated_below_rate(tmp_path):
    cache = PredictionCache(tmp_path / "c.jsonl")
    ids = [str(i) for i in range(100)]
    texts = [f"t{i}" for i in range(100)]
    failed = tmp_path / "failed.jsonl"
    out = predict_cached(Counting(fail_on={"t7"}), cache, ids, texts, LABELS, failed_path=failed)
    assert len(out) == 99
    assert read_jsonl(failed)[0]["complaint_id"] == "7"


def test_too_many_failures_raise(tmp_path):
    cache = PredictionCache(tmp_path / "c.jsonl")
    with pytest.raises(RunFailedError):
        predict_cached(Counting(fail_on={"x"}), cache, ["1", "2"], ["x", "y"], LABELS)


def test_missing_lists_uncached(tmp_path):
    cache = PredictionCache(tmp_path / "c.jsonl")
    cache.put("1", Prediction("a", {"a": 1.0}))
    assert cache.missing(["1", "2"]) == ["2"]


def test_confirm_cost_aborts_without_yes():
    with pytest.raises(SystemExit):
        confirm_cost(10, 0.5, input_fn=lambda _: "n")
    confirm_cost(10, 0.5, input_fn=lambda _: "y")
    confirm_cost(10, 0.5, assume_yes=True, input_fn=lambda _: pytest.fail("asked"))
    confirm_cost(0, 0.0, input_fn=lambda _: pytest.fail("asked"))


OTHER = LabelSet((Label("a", "A", "a"), Label("b", "B", "changed description")))


def test_cache_for_separates_label_sets(tmp_path):
    clf = Counting()
    a = cache_for(clf, LABELS, root=tmp_path)
    b = cache_for(clf, OTHER, root=tmp_path)
    assert a.path != b.path
    assert a.path.name == f"fake-1-{LABELS.labels_hash()}.jsonl"


def test_cache_for_ignores_legacy_file_without_labels_hash(tmp_path):
    legacy = PredictionCache(tmp_path / "fake-1.jsonl")
    legacy.put("1", Prediction("a", {"a": 1.0}))
    cache = cache_for(Counting(), LABELS, root=tmp_path)
    assert "1" not in cache
    assert (tmp_path / "fake-1.jsonl").exists()
