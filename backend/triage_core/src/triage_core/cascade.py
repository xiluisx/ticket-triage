"""Cost-aware cascade: each level accepts confident predictions, the rest escalate.

Per-label thresholds make the priority cascade asymmetric: `not_urgent` needs a higher
confidence than `urgent` to be accepted early (safety over cost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from triage_core.classify import Classifier, Prediction, StaleModelError
from triage_core.labels import LabelSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelThresholds:
    default: float
    per_label: dict[str, float] = field(default_factory=dict)

    def threshold(self, label: str) -> float:
        return self.per_label.get(label, self.default)

    def to_dict(self) -> dict:
        return {"default": self.default, "per_label": dict(self.per_label)}

    @classmethod
    def from_dict(cls, data: dict) -> LevelThresholds:
        per_label = {k: float(v) for k, v in data.get("per_label", {}).items()}
        return cls(float(data["default"]), per_label)


class Cascade:
    def __init__(self, levels: list[tuple[Classifier, LevelThresholds | None]]):
        if not levels or levels[-1][1] is not None:
            raise ValueError("last level must accept everything (thresholds=None)")
        self.levels = levels

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]:
        results: list[Prediction | None] = [None] * len(texts)
        spent = [0.0] * len(texts)
        waited = [0.0] * len(texts)
        skipped: list[list[str]] = [[] for _ in texts]
        pending = list(range(len(texts)))
        for clf, thresholds in self.levels:
            if not pending:
                break
            try:
                preds = clf.predict([texts[i] for i in pending], labels)
            except StaleModelError as err:
                logger.warning("cascade level %s skipped (stale model): %s", clf.name, err)
                for i in pending:
                    skipped[i].append(f"{clf.name}:stale")
                continue
            except Exception as err:
                # Never log ticket text; the error type is enough to debug a level outage.
                logger.warning("cascade level %s failed: %s", clf.name, type(err).__name__)
                for i in pending:
                    skipped[i].append(f"{clf.name}:error")
                continue
            still = []
            for i, pred in zip(pending, preds, strict=True):
                spent[i] += pred.cost_usd
                waited[i] += pred.latency_ms
                if thresholds is None or pred.confidence >= thresholds.threshold(pred.label):
                    results[i] = replace(
                        pred,
                        level=clf.name,
                        cost_usd=spent[i],
                        latency_ms=waited[i],
                        skipped=tuple(skipped[i]),
                    )
                else:
                    still.append(i)
            pending = still
        for i in pending:  # every level failed: the ticket is kept as unclassified
            results[i] = Prediction(
                None,
                {},
                level="none",
                cost_usd=spent[i],
                latency_ms=waited[i],
                skipped=tuple(skipped[i]),
            )
        return results  # type: ignore[return-value]
