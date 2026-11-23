"""Per-task label sets and instructions for zero-shot models."""

from training.paths import CATEGORIES, PRIORITIES
from triage_core.labels import LabelSet

TASKS = ("category", "priority")

INSTRUCTIONS = {
    "category": "Which support team should handle this customer complaint?",
    "priority": (
        "How urgent is this customer complaint? Choose urgent only when the customer faces "
        "active or imminent harm as described in the label definitions; otherwise choose "
        "not_urgent."
    ),
}


def labels_for(task: str) -> LabelSet:
    return LabelSet.from_json({"category": CATEGORIES, "priority": PRIORITIES}[task])
