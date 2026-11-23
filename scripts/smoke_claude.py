"""Smoke test: Claude on Bedrock via the Anthropic SDK (Mantle) with structured outputs.

One call on a synthetic complaint (~$0.0001). Exits non-zero on failure.
"""

import sys

from training.llm import claude_classifier
from training.tasks import labels_for

TEXT = "Someone stole my debit card yesterday and keeps charging it. The bank won't freeze it."


def main() -> int:
    clf = claude_classifier("priority")
    [pred] = clf.predict([TEXT], labels_for("priority"))
    print(
        f"model={clf.model_id} label={pred.label} cost=${pred.cost_usd:.6f} "
        f"latency={pred.latency_ms:.0f}ms"
    )
    return 0 if pred.label == "urgent" else 1


if __name__ == "__main__":
    sys.exit(main())
