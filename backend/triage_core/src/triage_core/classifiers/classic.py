"""Classic text classifier: any fitted model exposing predict_proba and classes_."""

from __future__ import annotations

import time

from triage_core.classify import Prediction, StaleModelError
from triage_core.labels import LabelSet


class ClassicClassifier:
    name = "classic"

    def __init__(self, model, labels_hash: str, version: str):
        self.model = model
        self.labels_hash = labels_hash
        self.cache_key = f"classic-{version}-{labels_hash}"

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]:
        if labels.labels_hash() != self.labels_hash:
            raise StaleModelError(
                f"model trained on labels {self.labels_hash}, tenant has {labels.labels_hash()}"
            )
        if not texts:
            return []
        start = time.perf_counter()
        proba = self.model.predict_proba(list(texts))
        per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
        classes = [str(c) for c in self.model.classes_]
        out = []
        for row in proba:
            probs = {c: float(p) for c, p in zip(classes, row, strict=True)}
            out.append(Prediction(max(probs, key=probs.get), probs, latency_ms=per_item_ms))
        return out
