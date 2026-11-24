import dataclasses
import json

import joblib
import pytest
from sklearn.dummy import DummyClassifier

from triage_core.labels import LabelSet
from triage_services.clients import KeywordClassifier, fake_classifiers
from triage_services.engine import build_engine, classify, latest_version_dir
from triage_services.settings import REPO_ROOT


def write_model(root, task, version, labels, thresholds, created_at=None):
    path = root / task / version
    path.mkdir(parents=True)
    model = DummyClassifier(strategy="prior").fit(
        [[0], [0], [1]], list(labels.ids)[:1] * 2 + [labels.ids[1]]
    )
    joblib.dump(model, path / "model.joblib")
    meta = {"version": version, "labels_hash": labels.labels_hash()}
    if created_at:
        meta["created_at"] = created_at
    (path / "metadata.json").write_text(json.dumps(meta))
    (path / "thresholds.json").write_text(json.dumps(thresholds))
    return path


def priority_labels(settings):
    return LabelSet.from_json(settings.datasets_dir / "priorities_v2.json")


def test_latest_version_uses_created_at(tmp_path, settings):
    labels = priority_labels(settings)
    thr = {"order": ["classic", "claude"], "levels": {"classic": {"default": 0.5}}}
    write_model(tmp_path, "priority", "v20261123-3c155e3", labels, thr)
    write_model(
        tmp_path, "priority", "v20261125T100000Z-aaaaaaa", labels, thr, "2026-11-25T10:00:00Z"
    )
    (tmp_path / "priority" / ".v20261126T000000Z-bbbbbbb.tmp").mkdir()
    assert latest_version_dir(tmp_path / "priority").name == "v20261125T100000Z-aaaaaaa"
    assert latest_version_dir(tmp_path / "missing") is None


def test_classic_level_accepts_with_version(tmp_path, settings):
    labels = priority_labels(settings)
    thr = {
        "order": ["classic", "jev", "claude"],
        "levels": {"classic": {"default": 0.5}, "jev": {"default": 0.5}},
        "jev_temperature": 4.0,
    }
    write_model(tmp_path, "priority", "v20261123-3c155e3", labels, thr)
    seen = []

    class Spy:
        def jev(self, task, temperature):
            seen.append((task, temperature))
            return KeywordClassifier("jev")

        def claude(self, task):
            return KeywordClassifier("claude")

    s = dataclasses.replace(settings, models_dir=tmp_path)
    engine = build_engine("priority", s, Spy())
    assert seen == [("priority", 4.0)]
    out = classify({"priority": engine}, "anything")["priority"]
    assert (out.level, out.model_version, out.label) == ("classic", "v20261123-3c155e3", "urgent")


def test_stale_classic_is_skipped(tmp_path, settings):
    other = LabelSet.from_json(settings.datasets_dir / "priorities_v1.json")
    thr = {"order": ["classic", "claude"], "levels": {"classic": {"default": 0.0}}}
    write_model(tmp_path, "priority", "v20261123-3c155e3", other, thr)
    s = dataclasses.replace(settings, models_dir=tmp_path)
    out = classify({"priority": build_engine("priority", s, fake_classifiers())}, "x")["priority"]
    assert out.level == "claude"
    assert out.skipped == ["classic:stale"]


@pytest.mark.skipif(not (REPO_ROOT / "models").exists(), reason="models/ is local only")
def test_real_models_load_and_classify(settings):
    s = dataclasses.replace(settings, models_dir=REPO_ROOT / "models")
    engines = {t: build_engine(t, s, fake_classifiers()) for t in ("category", "priority")}
    out = classify(engines, "Someone stole my identity and opened a credit card in my name")
    assert out["category"].label is not None
    assert out["priority"].label in ("urgent", "not_urgent")
