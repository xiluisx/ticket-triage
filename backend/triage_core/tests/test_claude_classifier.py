import json
from types import SimpleNamespace

import pytest

from triage_core.classifiers.claude import ClaudeClassifier
from triage_core.classify import InvalidLabelError
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("a", "A", "first"), Label("b", "B", "second")))


def response(payload, stop_reason="end_turn", tokens=(1000, 10)):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class FakeClient:
    def __init__(self, resp):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self.resp = resp

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


def make(client, **kw):
    return ClaudeClassifier(
        client, "anthropic.claude-haiku-5-5", "Which team?", usd_per_mtok=(0.10, 0.50), **kw
    )


def test_predict_parses_label_and_cost():
    client = FakeClient(response({"label": "b"}))
    [p] = make(client).predict(["the complaint"], LABELS)
    assert p.label == "b"
    assert p.probs == {"b": 1.0}
    assert p.rationale is None
    assert p.cost_usd == pytest.approx(1000 * 0.10 / 1e6 + 10 * 0.50 / 1e6)


def test_request_uses_enum_schema_and_wraps_text():
    client = FakeClient(response({"label": "a"}))
    make(
        client,
        request_options={
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
        },
    ).predict(["hi"], LABELS)
    call = client.calls[0]
    schema = call["output_config"]["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == ["a", "b"]
    assert schema["required"] == ["label"]
    assert call["output_config"]["effort"] == "low"
    assert call["thinking"] == {"type": "disabled"}
    assert "<complaint>\nhi\n</complaint>" in call["messages"][0]["content"]
    assert "- a: first" in call["system"]


def test_rationale_requested_when_enabled():
    client = FakeClient(response({"label": "a", "rationale": "because"}))
    [p] = make(client, with_rationale=True).predict(["x"], LABELS)
    assert p.rationale == "because"
    schema = client.calls[0]["output_config"]["format"]["schema"]
    assert schema["required"] == ["label", "rationale"]


def test_refusal_raises():
    client = FakeClient(response({}, stop_reason="refusal"))
    with pytest.raises(InvalidLabelError, match="refus"):
        make(client).predict(["x"], LABELS)


def test_label_outside_set_raises():
    client = FakeClient(response({"label": "zzz"}))
    with pytest.raises(InvalidLabelError):
        make(client).predict(["x"], LABELS)


def test_cache_key_depends_on_model_and_prompt():
    a = make(FakeClient(None))
    b = ClaudeClassifier(FakeClient(None), "anthropic.other", "Which team?", (1, 1))
    c = make(FakeClient(None), with_rationale=True)
    assert len({a.cache_key, b.cache_key, c.cache_key}) == 3


def test_long_text_is_truncated():
    client = FakeClient(response({"label": "a"}))
    make(client, max_chars=5).predict(["0123456789"], LABELS)
    assert "<complaint>\n01234\n</complaint>" in client.calls[0]["messages"][0]["content"]


def test_max_tokens_raises_invalid_label():
    client = FakeClient(response({"label": "a"}, stop_reason="max_tokens"))
    with pytest.raises(InvalidLabelError, match="max_tokens"):
        make(client).predict(["x"], LABELS)


def test_no_text_block_raises_invalid_label():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking", thinking="...")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    with pytest.raises(InvalidLabelError, match="no text"):
        make(FakeClient(resp)).predict(["x"], LABELS)


def test_invalid_json_raises_invalid_label():
    resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text='{"label": "a"')],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    with pytest.raises(InvalidLabelError, match="invalid JSON"):
        make(FakeClient(resp)).predict(["x"], LABELS)
