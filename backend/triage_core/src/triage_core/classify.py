"""Classifier contract shared by training, the cascade and the inference Lambda."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from triage_core.labels import LabelSet


@dataclass(frozen=True)
class Prediction:
    label: str | None
    probs: dict[str, float]
    rationale: str | None = None
    level: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    skipped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # JSON round-trips (caches, preds files) give a list; keep equality stable.
        object.__setattr__(self, "skipped", tuple(self.skipped))

    @property
    def confidence(self) -> float:
        return 0.0 if self.label is None else self.probs.get(self.label, 0.0)


class Classifier(Protocol):
    name: str
    cache_key: str

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]: ...


class InvalidLabelError(ValueError):
    """The model answered with a label outside the LabelSet, or refused to answer."""


class StaleModelError(RuntimeError):
    """The model was trained on another LabelSet (labels_hash mismatch)."""


def short_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
