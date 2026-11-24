"""Per-task instructions and Claude pricing/options, shared by training and the Lambdas."""

from __future__ import annotations

INSTRUCTIONS = {
    "category": "Which support team should handle this customer complaint?",
    "priority": (
        "How urgent is this customer complaint? Choose urgent only when the customer faces "
        "active or imminent harm as described in the label definitions; otherwise choose "
        "not_urgent."
    ),
}

DEFAULT_CLAUDE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# USD per million tokens (input, output). First-party rates; check the Bedrock pricing page
# and update if Bedrock differs.
CLAUDE_USD_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


def claude_prices(model_id: str) -> tuple[float, float]:
    for key, prices in CLAUDE_USD_PER_MTOK.items():
        if key in model_id:
            return prices
    raise ValueError(f"no pricing for {model_id}; add it to CLAUDE_USD_PER_MTOK")


def claude_request_options(model_id: str) -> dict:
    """Haiku 5.5 thinks by default; classification needs no thinking (allowed at effort <= high)."""
    if "haiku-5-5" in model_id:
        return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    return {}
