import datetime as dt
import json

import numpy as np
import pytest

from training.train import (
    build_pipeline,
    file_sha256,
    fit_select,
    latest_artifact,
    load_classic,
    save_artifact,
    version_tag,
)
from triage_core.labels import Label, LabelSet

LABELS = LabelSet((Label("cards", "C", "c"), Label("mortgage", "M", "m")))
CARD = [
    "card fee charged twice",
    "credit card interest wrong",
    "card declined fee",
    "my card bill",
]
HOME = [
    "mortgage escrow error",
    "foreclosure on my home",
    "loan servicer mortgage",
    "home loan late",
]


def data(n=10):
    texts = [f"{t} {i}" for i in range(n) for t in CARD + HOME]
    labels = [lab for _ in range(n) for lab in ["cards"] * 4 + ["mortgage"] * 4]
    return texts, labels


def test_pipeline_fits_and_returns_probabilities():
    texts, y = data()
    model = build_pipeline("logreg", "sigmoid", n_features=2**10).fit(texts, y)
    proba = model.predict_proba(["card fee"])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == ["cards", "mortgage"]


def test_fit_select_reports_both_candidates():
    texts, y = data()
    name, model, results = fit_select(texts, y, texts[:16], y[:16], "sigmoid", n_features=2**10)
    assert name in {"logreg", "svc"}
    assert set(results) == {"logreg", "svc"}
    assert results[name]["macro_f1"] == max(r["macro_f1"] for r in results.values())


def test_artifact_roundtrip(tmp_path):
    texts, y = data()
    model = build_pipeline("svc", "sigmoid", n_features=2**10).fit(texts, y)
    path = save_artifact(model, "category", LABELS, {"note": "x"}, "v20261123-abc", root=tmp_path)
    assert latest_artifact("category", root=tmp_path) == path
    clf = load_classic(path)
    [pred] = clf.predict(["foreclosure home"], LABELS)
    assert pred.label == "mortgage"
    assert clf.cache_key == f"classic-v20261123-abc-{LABELS.labels_hash()}"


def test_version_tag_has_utc_time():
    now = dt.datetime(2026, 11, 25, 14, 30, 12, tzinfo=dt.UTC)
    assert version_tag("abc1234", now) == "v20261125T143012Z-abc1234"


def test_save_artifact_writes_provenance_and_is_immutable(tmp_path):
    texts, y = data()
    model = build_pipeline("svc", "sigmoid", n_features=2**10).fit(texts, y)
    path = save_artifact(
        model,
        "category",
        LABELS,
        {},
        "v20261125T143012Z-abc1234",
        root=tmp_path,
        created_at="2026-11-25T14:30:12Z",
    )
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["created_at"] == "2026-11-25T14:30:12Z"
    assert meta["artifact_sha256"] == file_sha256(path / "model.joblib")
    assert (path / "metadata.json").read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        save_artifact(model, "category", LABELS, {}, "v20261125T143012Z-abc1234", root=tmp_path)


def _fake_version(root, task, name, created_at=None):
    path = root / task / name
    path.mkdir(parents=True)
    if created_at:
        (path / "metadata.json").write_text(json.dumps({"created_at": created_at}))
    return path


def test_latest_artifact_uses_created_at_not_name(tmp_path):
    _fake_version(tmp_path, "priority", "v20261125T100000Z-fff0000", "2026-11-25T10:00:00Z")
    newer = _fake_version(tmp_path, "priority", "v20261125T120000Z-0000aaa", "2026-11-25T12:00:00Z")
    assert latest_artifact("priority", root=tmp_path) == newer


def test_latest_artifact_prefers_new_format_over_legacy(tmp_path):
    _fake_version(tmp_path, "priority", "v20261123-3c155e3")
    new = _fake_version(tmp_path, "priority", "v20261124T000000Z-aaa0000", "2026-11-24T00:00:00Z")
    assert latest_artifact("priority", root=tmp_path) == new


def test_save_artifact_crash_leaves_no_version_and_retry_works(tmp_path, monkeypatch):
    texts, y = data()
    model = build_pipeline("svc", "sigmoid", n_features=2**10).fit(texts, y)
    version = "v20261125T143012Z-abc1234"

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("training.train.joblib.dump", boom)
    with pytest.raises(OSError, match="disk full"):
        save_artifact(model, "category", LABELS, {}, version, root=tmp_path)
    assert not (tmp_path / "category" / version).exists()
    monkeypatch.undo()
    path = save_artifact(model, "category", LABELS, {}, version, root=tmp_path)
    assert (path / "metadata.json").exists()
    assert [p.name for p in (tmp_path / "category").iterdir()] == [version]


def test_latest_artifact_ignores_hidden_staging_dirs(tmp_path):
    _fake_version(tmp_path, "priority", "v20261125T100000Z-fff0000", "2026-11-25T10:00:00Z")
    _fake_version(tmp_path, "priority", ".v20261126T100000Z-aaa0000.tmp", "2026-11-26T10:00:00Z")
    assert latest_artifact("priority", root=tmp_path).name == "v20261125T100000Z-fff0000"
