"""Classification engine: classic models baked into the image + Jev + Claude, per task.

Each task runs its own cascade (thresholds.json of the model version); both tasks run in
parallel. A failing level is skipped and recorded; if every level fails the task is "none".
"""

from __future__ import annotations

import functools
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import joblib

from triage_core.cascade import Cascade, LevelThresholds
from triage_core.classifiers.classic import ClassicClassifier
from triage_core.classify import Classifier, Prediction
from triage_core.labels import LabelSet
from triage_services.clients import Classifiers
from triage_services.schemas import TaskPrediction

logger = logging.getLogger(__name__)

LABEL_FILES = {"category": "categories_v1.json", "priority": "priorities_v2.json"}
# Jev thresholds when no classic model is available (priority: not_urgent must be confident).
FALLBACK_JEV = {
    "category": {"default": 0.5, "per_label": {}},
    "priority": {"default": 0.5, "per_label": {"not_urgent": 0.9}},
}


@dataclass(frozen=True)
class TaskEngine:
    task: str
    labels: LabelSet
    cascade: Cascade
    model_version: str | None = None


def latest_version_dir(task_dir: Path) -> Path | None:
    """Newest version by metadata created_at (legacy versions without it sort first)."""
    if not task_dir.exists():
        return None
    dirs = [p for p in task_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]

    def created(p: Path) -> tuple[str, str]:
        meta = p / "metadata.json"
        stamp = (
            json.loads(meta.read_text(encoding="utf-8")).get("created_at", "")
            if meta.exists()
            else ""
        )
        return stamp, p.name

    return max(dirs, key=created, default=None)


def build_engine(task: str, settings, classifiers: Classifiers) -> TaskEngine:
    labels = LabelSet.from_json(settings.datasets_dir / LABEL_FILES[task])
    jev = functools.partial(classifiers.jev, task)
    levels: list[tuple[Classifier, LevelThresholds | None]] = []
    version = None
    path = latest_version_dir(settings.models_dir / task)
    if path is None:
        logger.warning("no classic model for %s; cascade starts at Jev", task)
        levels.append((jev(1.0), LevelThresholds.from_dict(FALLBACK_JEV[task])))
    else:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        thresholds = json.loads((path / "thresholds.json").read_text(encoding="utf-8"))
        version = meta["version"]
        model = joblib.load(path / "model.joblib")  # noqa: S301 - artifact baked into our image
        classic = ClassicClassifier(model, meta["labels_hash"], version)
        temperature = float(thresholds.get("jev_temperature", 1.0))
        for name in thresholds["order"][:-1]:
            clf = classic if name == "classic" else jev(temperature)
            levels.append((clf, LevelThresholds.from_dict(thresholds["levels"][name])))
    levels.append((classifiers.claude(task), None))
    return TaskEngine(task, labels, Cascade(levels), version)


def compose_text(subject: str, body: str) -> str:
    return f"{subject}\n\n{body}" if subject.strip() else body


def to_task_prediction(pred: Prediction, engine: TaskEngine) -> TaskPrediction:
    level = pred.level or "none"
    return TaskPrediction(
        label=pred.label,
        confidence=pred.confidence,
        level=level,
        model_version=engine.model_version if level == "classic" else None,
        latency_ms=pred.latency_ms,
        cost_usd=pred.cost_usd,
        rationale=pred.rationale if level == "claude" else None,
        skipped=list(pred.skipped),
    )


def _run(engine: TaskEngine, text: str) -> TaskPrediction:
    try:
        [pred] = engine.cascade.predict([text], engine.labels)
    except Exception as err:  # defensive: Cascade already turns level errors into "none"
        logger.error("cascade %s crashed: %s", engine.task, type(err).__name__)
        return TaskPrediction(
            label=None,
            confidence=0.0,
            level="none",
            latency_ms=0.0,
            cost_usd=0.0,
            skipped=["cascade:error"],
        )
    return to_task_prediction(pred, engine)


def classify(engines: dict[str, TaskEngine], text: str) -> dict[str, TaskPrediction]:
    with ThreadPoolExecutor(max_workers=max(1, len(engines))) as pool:
        futures = {task: pool.submit(_run, engine, text) for task, engine in engines.items()}
        return {task: f.result() for task, f in futures.items()}
