import pytest

from triage_core.classifiers.jev import JevClassifier
from triage_core.classify import InvalidLabelError
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("a", "A", "first"), Label("b", "B", "second")))


def answer(choice, probs):
    return {
        "answers": {
            "label": {
                "choice": choice,
                "confidence": max(probs.values()),
                "probabilities": probs,
            }
        }
    }


class FakePost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        return self.responses.pop(0)


def make(post, **kw):
    return JevClassifier(post, "Which team?", usd_per_call=0.0001, sleep=lambda s: None, **kw)


def test_predict_parses_choice_and_probs():
    post = FakePost([(200, answer("b", {"a": 0.3, "b": 0.7}))])
    [p] = make(post).predict(["text"], LABELS)
    assert p.label == "b"
    assert p.probs == {"a": 0.3, "b": 0.7}
    assert p.cost_usd == 0.0001
    body = post.bodies[0]
    assert body["state"] == {"ticket": "text"}
    assert body["questions"]["label"]["criteria"] == {"a": "first", "b": "second"}


def test_truncates_long_text():
    post = FakePost([(200, answer("a", {"a": 1.0, "b": 0.0}))])
    make(post, max_chars=5).predict(["0123456789"], LABELS)
    assert post.bodies[0]["state"]["ticket"] == "01234"


def test_retries_on_429_and_5xx():
    ok = (200, answer("a", {"a": 0.9, "b": 0.1}))
    post = FakePost([(429, {}), (503, {}), ok])
    [p] = make(post).predict(["t"], LABELS)
    assert p.label == "a"
    assert len(post.bodies) == 3


def test_client_error_is_not_retried():
    post = FakePost([(400, {"error": "bad"})])
    with pytest.raises(RuntimeError, match="400"):
        make(post).predict(["t"], LABELS)
    assert len(post.bodies) == 1


def test_unknown_choice_raises():
    post = FakePost([(200, answer("zzz", {"a": 0.5, "b": 0.5}))])
    with pytest.raises(InvalidLabelError):
        make(post).predict(["t"], LABELS)


def test_temperature_is_applied():
    post = FakePost([(200, answer("a", {"a": 0.9, "b": 0.1}))])
    [p] = make(post, temperature=2.0).predict(["t"], LABELS)
    assert p.label == "a"
    assert p.probs["a"] < 0.9


def test_cache_key_changes_with_instructions():
    a = JevClassifier(FakePost([]), "Q1", usd_per_call=0)
    b = JevClassifier(FakePost([]), "Q2", usd_per_call=0)
    assert a.cache_key != b.cache_key
    assert a.cache_key.startswith("jev-")
