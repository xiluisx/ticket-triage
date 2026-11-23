import numpy as np

from training.train import (
    build_pipeline,
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
    assert clf.cache_key == "classic-v20261123-abc"


def test_latest_artifact_picks_newest(tmp_path):
    for v in ("v20261101-aaa", "v20261123-bbb"):
        (tmp_path / "priority" / v).mkdir(parents=True)
    assert latest_artifact("priority", root=tmp_path).name == "v20261123-bbb"


def test_version_tag_format():
    import datetime as dt

    assert version_tag("abc1234", dt.date(2026, 11, 23)) == "v20261123-abc1234"
