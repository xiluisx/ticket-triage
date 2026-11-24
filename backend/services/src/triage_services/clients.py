"""Jev and Claude classifiers for the Lambda (2 s / 8 s budget) and free local fakes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from triage_core.classifiers.claude import ClaudeClassifier
from triage_core.classifiers.jev import JEV_URL, JevClassifier
from triage_core.classify import Classifier, Prediction
from triage_core.labels import LabelSet
from triage_core.llm_config import INSTRUCTIONS, claude_prices, claude_request_options

JEV_TIMEOUT_S = 2.0
CLAUDE_TIMEOUT_S = 8.0
TASKS = ("category", "priority")


@dataclass(frozen=True)
class Classifiers:
    jev: Callable[[str, float], Classifier]  # (task, temperature) -> Jev
    claude: Callable[[str], Classifier]  # task -> Claude


def jev_post(api_key: str, timeout: float = JEV_TIMEOUT_S, transport=None):
    client = httpx.Client(
        timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}, transport=transport
    )

    def post(body: dict) -> tuple[int, dict]:
        try:
            resp = client.post(JEV_URL, json=body)
        except httpx.TransportError as err:
            return 599, {"error": type(err).__name__}
        if resp.status_code != 200:
            return resp.status_code, {"error": resp.text[:300]}
        return 200, resp.json()

    return post


def aws_classifiers(settings, ssm) -> Classifiers:
    """Built once per container: Jev key from SSM, Claude over Bedrock with the IAM role."""
    from anthropic import AnthropicBedrock

    param = ssm.get_parameter(Name=settings.jev_api_key_param, WithDecryption=True)
    post = jev_post(param["Parameter"]["Value"])
    bedrock = AnthropicBedrock(aws_region=settings.region, timeout=CLAUDE_TIMEOUT_S, max_retries=1)
    prices = claude_prices(settings.claude_model_id)
    options = claude_request_options(settings.claude_model_id)
    claude = {
        task: ClaudeClassifier(
            bedrock,
            settings.claude_model_id,
            INSTRUCTIONS[task],
            usd_per_mtok=prices,
            request_options=options,
        )
        for task in TASKS
    }

    def jev(task: str, temperature: float) -> Classifier:
        return JevClassifier(
            post,
            INSTRUCTIONS[task],
            settings.jev_usd_per_call,
            temperature=temperature,
            attempts=1,  # no retries inside a synchronous request
        )

    return Classifiers(jev=jev, claude=claude.__getitem__)


class KeywordClassifier:
    """Local stand-in for Jev/Claude ($0): first label whose id words appear in the text."""

    def __init__(self, name: str):
        self.name = name
        self.cache_key = f"fake-{name}"

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]:
        out = []
        rest = 0.1 / max(len(labels.ids) - 1, 1)
        for text in texts:
            low = text.lower()
            hits = [i for i in labels.ids if any(w in low for w in i.split("_"))]
            label = hits[0] if hits else labels.ids[0]
            probs = {i: (0.9 if i == label else rest) for i in labels.ids}
            out.append(Prediction(label, probs, cost_usd=0.0, latency_ms=1.0))
        return out


def fake_classifiers() -> Classifiers:
    return Classifiers(
        jev=lambda task, temperature: KeywordClassifier("jev"),
        claude=lambda task: KeywordClassifier("claude"),
    )
