import pandas as pd

from training.cfpb import COLUMN_MAP, filter_chunk, map_categories

HEADER = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]


def row(date: str, product: str, narrative: str, tags: str, cid: str) -> list[str]:
    return [date, product, "", "Issue", "", narrative, "", "ACME", "TX", "75001", tags, "Web",
            date, "Closed", "Yes", cid]  # fmt: skip


def raw_chunk() -> pd.DataFrame:
    rows = [
        row("2024-05-02", "Debt collection", "They call me daily XXXX", "Older American", "1"),
        row("2024-05-03", "Mortgage", "", "", "2"),  # no narrative
        row("2023-12-31", "Credit card", "Charged twice", "", "3"),  # out of range
        row("2024-06-01", "Credit card", "   ", "", "4"),  # blank narrative
    ]
    return pd.DataFrame(rows, columns=HEADER)


def test_filter_chunk_keeps_valid_rows_and_renames():
    out = filter_chunk(raw_chunk(), date_min="2024-01-01", date_max="2026-08-31")
    assert out["complaint_id"].tolist() == ["1"]
    assert list(out.columns) == list(COLUMN_MAP.values())
    assert out.loc[0, "tags"] == "Older American"
    assert out.loc[0, "date_received"] == pd.Timestamp("2024-05-02")


def test_filter_chunk_fills_empty_tags():
    chunk = raw_chunk()
    chunk.loc[0, "Tags"] = None
    out = filter_chunk(chunk, date_min="2024-01-01", date_max="2026-08-31")
    assert out.loc[0, "tags"] == ""


def test_map_categories_returns_none_for_unknown():
    products = pd.Series(["Debt collection", "Brand new product"])
    mapped = map_categories(products, {"Debt collection": "debt_collection"})
    assert mapped.tolist() == ["debt_collection", None]
