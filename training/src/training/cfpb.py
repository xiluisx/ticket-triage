"""CFPB Consumer Complaint Database narratives archive: download, filter and label mapping.

CFPB stopped publishing narratives on 2026-08-14 (bulk file and API no longer include them).
The FOIA narratives archive keeps every previously published narrative.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ARCHIVE_BASE_URL = "https://files.consumerfinance.gov/f/documents/"

EXPORTS = (
    "CCDB_Export_5_September_2023_through_March_2024",
    "CCDB_Export_6_April_2024_through_July_2024",
    "CCDB_Export_7_August_2024_through_October_2024",
    "CCDB_Export_8_November_2024_through_December_2024",
    "CCDB_Export_9_January_2025_through_February_2025",
    "CCDB_Export_10_March_2025_through_April_2025",
    "CCDB_Export_11_May_2025_through_June_2025",
    "CCDB_Export_12_July_2025_through_August_2025",
    "CCDB_Export_13_September_2025_through_October_2025",
    "CCDB_Export_14_November_2025_through_December_2025",
    "CCDB_Export_15_January_2026_through_February_2026",
    "CCDB_Export_16_March_2026",
    "CCDB_Export_17_April_2026",
    "CCDB_Export_18_May_2026",
    "CCDB_Export_19_June_2026",
    "CCDB_Export_20_July_2026",
    "CCDB_Export_21_August_2026",
)

COLUMN_MAP = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Tags": "tags",
}

DATE_MIN = "2024-01-01"
DATE_MAX = "2026-08-31"


def filter_chunk(
    chunk: pd.DataFrame, date_min: str = DATE_MIN, date_max: str = DATE_MAX
) -> pd.DataFrame:
    """Keep rows with a non-blank narrative inside the date range; rename to snake_case."""
    out = chunk[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    out["narrative"] = out["narrative"].fillna("").str.strip()
    out["tags"] = out["tags"].fillna("")
    out["date_received"] = pd.to_datetime(out["date_received"])
    keep = (out["narrative"] != "") & out["date_received"].between(date_min, date_max)
    return out[keep].reset_index(drop=True)


def map_categories(products: pd.Series, product_map: dict[str, str]) -> pd.Series:
    """Map CFPB product names to category ids; unknown products become None."""
    return products.map(product_map).astype(object).where(products.isin(product_map), None)


def load_product_map(path: Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cfpb_product_map"]


def download_export(name: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"{name}.zip"
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(f"{ARCHIVE_BASE_URL}{name}.zip", tmp)  # noqa: S310 - fixed https URL
    tmp.rename(dest)
    return dest


def build_dataset(raw_dir: Path, out_path: Path, chunksize: int = 100_000) -> dict:
    """Download every export, filter in chunks and write one parquet. Returns summary stats."""
    frames = []
    for name in EXPORTS:
        zip_path = download_export(name, raw_dir)
        reader = pd.read_csv(
            zip_path,
            compression="zip",
            encoding="utf-8-sig",
            usecols=list(COLUMN_MAP),
            dtype=str,
            chunksize=chunksize,
        )
        for chunk in reader:
            frames.append(filter_chunk(chunk))
        print(f"{name}: done")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("complaint_id")
    df = df.sort_values("date_received").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return {
        "rows": len(df),
        "date_min": str(df["date_received"].min().date()),
        "date_max": str(df["date_received"].max().date()),
        "products": df["product"].value_counts().to_dict(),
    }


def read_narratives(path: Path, rows, batch_size: int = 100_000) -> list[str]:
    """Return the narratives at the given row positions, in the given order.

    Streams the parquet in batches so memory stays bounded by the batch size; loading the whole
    narrative column of 2.16M complaints costs >4 GB and Arrow does not return it to the OS.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    out: list[str] = [""] * len(rows)
    start, done = 0, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["narrative"]):
        if done == len(rows):
            break
        end = start + batch.num_rows
        stop = int(np.searchsorted(sorted_rows, end, side="left"))
        if stop > done:
            local = pa.array(sorted_rows[done:stop] - start)
            texts = batch.column(0).take(local).to_pylist()
            for k, text in zip(order[done:stop], texts, strict=True):
                out[k] = text
            done = stop
        start = end
    return out


def narrative_word_counts(path: Path, batch_size: int = 200_000) -> np.ndarray:
    """Whitespace-delimited word count of every narrative, streamed in batches."""
    batches = pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["narrative"])
    counts = [pc.add(pc.count_substring_regex(b.column(0), r"\s+"), 1).to_numpy() for b in batches]
    return np.concatenate(counts) if counts else np.array([], dtype=np.int64)
