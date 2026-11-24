import pytest

from training import llm
from training.tasks import INSTRUCTIONS, TASKS, labels_for


def test_tasks_have_labels_and_instructions():
    assert TASKS == ("category", "priority")
    assert labels_for("priority").ids == ("urgent", "not_urgent")
    assert len(labels_for("category").ids) == 8
    assert set(INSTRUCTIONS) == set(TASKS)


def test_prices_by_model():
    assert llm.claude_prices("anthropic.claude-haiku-5-5") == (0.10, 0.50)
    assert llm.claude_prices("anthropic.claude-haiku-4-5") == (1.00, 5.00)
    with pytest.raises(ValueError, match="pricing"):
        llm.claude_prices("anthropic.unknown")


def test_request_options_disable_thinking_on_haiku_5_5():
    assert llm.claude_request_options("anthropic.claude-haiku-5-5") == {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }
    assert llm.claude_request_options("anthropic.claude-haiku-4-5") == {}


def test_estimate_grows_with_text():
    short = llm.estimate_claude_usd(["a" * 100], "anthropic.claude-haiku-5-5")
    long = llm.estimate_claude_usd(["a" * 10_000], "anthropic.claude-haiku-5-5")
    assert 0 < short < long


def test_jev_classifier_none_without_key(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert llm.jev_classifier("category") is None
