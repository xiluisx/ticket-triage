"""Train classic models (TF-IDF + LogReg / LinearSVC, calibrated) and save versioned artifacts.

python -m training.train category   # ~50k product-mapped labels, isotonic calibration
python -m training.train priority   # ~1.6k Claude labels, sigmoid calibration
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess  # noqa: S404 - fixed git command
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from training.jsonl import read_jsonl, write_json
from training.metrics import ece, macro_f1
from training.paths import DISTILLED, MODELS
from training.splits import load_meta, read_ids, with_text
from training.tasks import labels_for
from triage_core.classifiers.classic import ClassicClassifier
from triage_core.labels import LabelSet

N_FEATURES = 2**18  # keeps the 8-class LogReg artifact ~16 MB (Lambda-friendly)
CALIBRATION = {"category": "isotonic", "priority": "sigmoid"}
ESTIMATORS = {
    "logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=1000),
    "svc": lambda: LinearSVC(class_weight="balanced"),
}


def build_pipeline(kind: str, calibration: str, n_features: int = N_FEATURES) -> Pipeline:
    return Pipeline(
        [
            (
                "hash",
                HashingVectorizer(
                    ngram_range=(1, 2), n_features=n_features, alternate_sign=False, norm=None
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            (
                "clf",
                CalibratedClassifierCV(
                    ESTIMATORS[kind](), method=calibration, cv=5, ensemble=False
                ),
            ),
        ]
    )


def fit_select(
    x_train: list[str],
    y_train: list[str],
    x_val: list[str],
    y_val: list[str],
    calibration: str,
    n_features: int = N_FEATURES,
) -> tuple[str, Pipeline, dict]:
    results, models = {}, {}
    classes = sorted(set(y_train))
    for kind in ESTIMATORS:
        model = build_pipeline(kind, calibration, n_features).fit(x_train, y_train)
        proba = model.predict_proba(x_val)
        pred = [str(c) for c in model.classes_[proba.argmax(axis=1)]]
        results[kind] = {
            "macro_f1": macro_f1(y_val, pred, classes),
            "ece": ece(y_val, proba, [str(c) for c in model.classes_]),
        }
        models[kind] = model
        print(f"  {kind}: macro-F1={results[kind]['macro_f1']:.3f} ECE={results[kind]['ece']:.3f}")
    best = max(results, key=lambda k: results[k]["macro_f1"])
    return best, models[best], results


def git_sha() -> str:
    out = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "--short=7", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def git_dirty() -> bool:
    out = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(out.stdout.strip())


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def version_tag(sha: str, now: dt.datetime | None = None) -> str:
    return f"v{(now or utc_now()):%Y%m%dT%H%M%SZ}-{sha}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(
    model,
    task: str,
    labels: LabelSet,
    metadata: dict,
    version: str,
    root: Path = MODELS,
    created_at: str | None = None,
) -> Path:
    """Write an immutable versioned artifact; an existing version is never overwritten.

    Files go to a hidden staging dir that is renamed into place last, so a crash never
    leaves a half-written version behind.
    """
    path = root / task / version
    if path.exists():
        raise FileExistsError(path)
    staging = root / task / f".{version}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        joblib.dump(model, staging / "model.joblib")
        meta = {
            "task": task,
            "version": version,
            "labels_hash": labels.labels_hash(),
            "created_at": created_at or f"{utc_now():%Y-%m-%dT%H:%M:%SZ}",
            "artifact_sha256": file_sha256(staging / "model.joblib"),
            **metadata,
        }
        write_json(staging / "metadata.json", meta)
        try:
            staging.rename(path)
        except OSError as err:
            raise FileExistsError(path) from err
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


def _created_at(path: Path) -> str:
    meta = path / "metadata.json"
    if not meta.exists():
        return ""
    return json.loads(meta.read_text(encoding="utf-8")).get("created_at", "")


def latest_artifact(task: str, root: Path = MODELS) -> Path:
    """Newest version by metadata created_at; legacy versions without it sort first, by name."""
    versions = [p for p in (root / task).iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not versions:
        raise FileNotFoundError(f"no artifacts under {root / task}")
    return max(versions, key=lambda p: (_created_at(p), p.name))


def load_classic(path: Path) -> ClassicClassifier:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return ClassicClassifier(
        joblib.load(path / "model.joblib"), meta["labels_hash"], meta["version"]
    )


def training_frames(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = load_meta()
    if task == "category":
        train = with_text(meta, read_ids("train_category"))
        val = with_text(meta, read_ids("val_category"))
        return train.assign(label=train["category"]), val.assign(label=val["category"])
    rows = pd.DataFrame(read_jsonl(DISTILLED))
    frames = []
    for split in ("train", "val"):
        part = rows[rows["split"] == split]
        frames.append(
            with_text(meta, part["complaint_id"].tolist()).assign(label=part["label"].to_numpy())
        )
    return frames[0], frames[1]


def main(task: str) -> None:
    labels = labels_for(task)
    train, val = training_frames(task)
    print(f"{task}: train={len(train):,} val={len(val):,}")
    best, model, results = fit_select(
        train["text"].tolist(),
        train["label"].tolist(),
        val["text"].tolist(),
        val["label"].tolist(),
        CALIBRATION[task],
    )
    now = utc_now()
    sha = git_sha()
    path = save_artifact(
        model,
        task,
        labels,
        {
            "estimator": best,
            "calibration": CALIBRATION[task],
            "n_features": N_FEATURES,
            "train_rows": len(train),
            "val_rows": len(val),
            "val": results[best],
            "candidates": results,
            "label_source": "cfpb_product_map" if task == "category" else "claude_distilled",
            "git_sha": sha,
            "git_dirty": git_dirty(),
        },
        version_tag(sha, now),
        created_at=f"{now:%Y-%m-%dT%H:%M:%SZ}",
    )
    print(f"best={best} → {path}")


if __name__ == "__main__":
    main(sys.argv[1])
