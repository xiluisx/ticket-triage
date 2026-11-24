import pandas as pd

from training.evalsets import golden_frame
from training.predict import read_preds, write_preds
from triage_core.classify import Prediction


def test_golden_frame_joins_tags_and_product_category():
    rows = [{"complaint_id": "1", "text": "t1", "category": "cards", "priority": "urgent"}]
    meta = pd.DataFrame(
        {"complaint_id": ["1"], "tags": ["Servicemember"], "category": ["bank_accounts"]}
    ).set_index("complaint_id", drop=False)
    frame = golden_frame(rows, meta, "priority")
    assert frame.to_dict("records") == [
        {
            "complaint_id": "1",
            "text": "t1",
            "truth": "urgent",
            "tags": "Servicemember",
            "product_category": "bank_accounts",
        }
    ]


def test_preds_roundtrip(tmp_path):
    preds = {"1": Prediction("a", {"a": 0.9, "b": 0.1}, cost_usd=0.001, latency_ms=12.0)}
    write_preds("category", "test", "jev", preds, root=tmp_path)
    assert read_preds("category", "test", "jev", root=tmp_path) == preds
