"""Factories for the paid classifiers (Claude on Bedrock, Jev) plus pricing and estimates."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from training.paths import ROOT
from triage_core.classifiers.claude import ClaudeClassifier
from triage_core.classifiers.jev import JEV_URL, JevClassifier
from triage_core.llm_config import (  # noqa: F401 - re-exported for training scripts
    CLAUDE_USD_PER_MTOK,
    DEFAULT_CLAUDE_MODEL,
    INSTRUCTIONS,
    claude_prices,
    claude_request_options,
)

REGION = "us-east-1"
SYSTEM_TOKENS_ESTIMATE = 400
OUTPUT_TOKENS_ESTIMATE = 20
CHARS_PER_TOKEN = 3  # conservative: newer tokenizers count more tokens per character


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def claude_model_id() -> str:
    load_env()
    return os.environ.get("CLAUDE_MODEL_ID") or DEFAULT_CLAUDE_MODEL


def estimate_claude_usd(texts: list[str], model_id: str) -> float:
    price_in, price_out = claude_prices(model_id)
    tokens_in = sum(len(t) / CHARS_PER_TOKEN + SYSTEM_TOKENS_ESTIMATE for t in texts)
    tokens_out = OUTPUT_TOKENS_ESTIMATE * len(texts)
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def claude_classifier(task: str, with_rationale: bool = False) -> ClaudeClassifier:
    from anthropic import AnthropicBedrock

    model_id = claude_model_id()
    client = AnthropicBedrock(aws_region=REGION, max_retries=5)
    return ClaudeClassifier(
        client,
        model_id,
        INSTRUCTIONS[task],
        claude_prices(model_id),
        with_rationale=with_rationale,
        request_options=claude_request_options(model_id),
    )


def jev_post(api_key: str, timeout: float = 30.0):
    import httpx

    client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def post(body: dict) -> tuple[int, dict]:
        try:
            resp = client.post(JEV_URL, json=body)
        except httpx.TransportError as err:
            return 599, {"error": type(err).__name__}
        if resp.status_code != 200:
            return resp.status_code, {"error": resp.text[:300]}
        return 200, resp.json()

    return post


def jev_usd_per_call() -> float:
    value = os.environ.get("JEV_USD_PER_CALL")
    if not value:
        print("JEV_USD_PER_CALL not set in .env; Jev cost reported as $0")
        return 0.0
    return float(value)


def jev_classifier(task: str, temperature: float = 1.0) -> JevClassifier | None:
    load_env()
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("TYPESAFE_API_KEY not set; skipping Jev")
        return None
    return JevClassifier(
        jev_post(api_key), INSTRUCTIONS[task], jev_usd_per_call(), temperature=temperature
    )
