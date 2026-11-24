"""Replay saved predictions through the production Cascade (no API calls).

`texts` passed to a ReplayClassifier are complaint ids.
"""

from __future__ import annotations

from dataclasses import replace

from triage_core.calibration import apply_temperature
from triage_core.cascade import Cascade, LevelThresholds
from triage_core.classify import Prediction
from triage_core.labels import LabelSet


class ReplayClassifier:
    def __init__(self, name: str, preds: dict[str, Prediction], temperature: float = 1.0):
        self.name = name
        self.cache_key = f"replay-{name}"
        self.preds = preds
        self.temperature = temperature

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]:
        out = []
        for cid in texts:
            pred = self.preds[cid]
            if self.temperature != 1.0:
                pred = replace(pred, probs=apply_temperature(pred.probs, self.temperature))
            out.append(pred)
        return out


def replay_cascade(config: dict, preds_by_model: dict[str, dict[str, Prediction]]) -> Cascade:
    levels = []
    for name in config["order"][:-1]:
        if name not in preds_by_model:
            continue
        temperature = config.get("jev_temperature", 1.0) if name == "jev" else 1.0
        levels.append(
            (
                ReplayClassifier(name, preds_by_model[name], temperature),
                LevelThresholds.from_dict(config["levels"][name]),
            )
        )
    final = config["order"][-1]
    levels.append((ReplayClassifier(final, preds_by_model[final]), None))
    return Cascade(levels)


def common_ids(ids: list[str], preds_by_model: dict[str, dict[str, Prediction]]) -> list[str]:
    return [cid for cid in ids if all(cid in p for p in preds_by_model.values())]
