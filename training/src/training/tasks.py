"""Per-task label sets and instructions for zero-shot models."""

from training.paths import CATEGORIES, PRIORITIES
from triage_core.labels import LabelSet
from triage_core.llm_config import INSTRUCTIONS  # noqa: F401 - re-exported

TASKS = ("category", "priority")


def labels_for(task: str) -> LabelSet:
    return LabelSet.from_json({"category": CATEGORIES, "priority": PRIORITIES}[task])
