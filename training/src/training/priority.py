"""Binary priority: map the 3-level golden set (v1) to urgent / not_urgent (v2).

The golden set came out 67 P1 / 32 P2 / 1 P3, so P3 cannot be measured on its own.
"""

from __future__ import annotations

from collections import Counter

from training.jsonl import read_jsonl, write_jsonl
from training.paths import GOLDEN_V1, GOLDEN_V2

V1_TO_V2 = {"P1": "urgent", "P2": "not_urgent", "P3": "not_urgent"}


def to_binary(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["priority"] not in V1_TO_V2:
            raise ValueError(f"unknown priority {row['priority']!r} in {row['complaint_id']}")
        out.append({**row, "priority": V1_TO_V2[row["priority"]]})
    return out


if __name__ == "__main__":
    rows = to_binary(read_jsonl(GOLDEN_V1))
    write_jsonl(GOLDEN_V2, rows)
    print(f"{len(rows)} rows → {GOLDEN_V2}: {dict(Counter(r['priority'] for r in rows))}")
