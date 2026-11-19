"""Golden set: stratified sampling, Label Studio config and export conversion."""

from __future__ import annotations

from html import escape

import pandas as pd

from triage_core.labels import LabelSet

TAG_GROUPS = ("Older American", "Servicemember")


def sample_golden(
    df: pd.DataFrame, n: int, min_per_category: int, min_per_tag: int, seed: int
) -> pd.DataFrame:
    """Sample n rows guaranteeing minimum counts per fairness tag group and per category."""
    if min_per_tag * len(TAG_GROUPS) + min_per_category * df["category"].nunique() > n:
        raise ValueError(f"Quotas exceed n={n}")

    picked: list[pd.DataFrame] = []

    def taken() -> pd.Series:
        if not picked:
            return pd.Series(False, index=df.index)
        return df["complaint_id"].isin(pd.concat(picked)["complaint_id"])

    for tag in TAG_GROUPS:
        pool = df[df["tags"].str.contains(tag) & ~taken()]
        picked.append(pool.sample(min(min_per_tag, len(pool)), random_state=seed))

    for category in sorted(df["category"].unique()):
        missing = min_per_category - int((pd.concat(picked)["category"] == category).sum())
        if missing > 0:
            pool = df[(df["category"] == category) & ~taken()]
            picked.append(pool.sample(min(missing, len(pool)), random_state=seed))

    remaining = n - len(pd.concat(picked))
    picked.append(df[~taken()].sample(remaining, random_state=seed))
    out = pd.concat(picked)
    return out.iloc[out["complaint_id"].astype(int).argsort()].reset_index(drop=True)


def subsample_golden(
    df: pd.DataFrame,
    labeled: set[str],
    n: int,
    category_quotas: dict[str, int],
    min_per_tag: int,
    seed: int,
) -> pd.DataFrame:
    """Shrink a golden pool to n rows: keep every labeled row, then top up quotas, then fill.

    Labeled rows count toward the per-category and per-tag minimums, so only the
    shortfall is drawn from unlabeled rows.
    """
    if missing := labeled - set(df["complaint_id"]):
        raise ValueError(f"labeled ids not in pool: {sorted(missing)[:5]}")
    if len(labeled) > n:
        raise ValueError(f"{len(labeled)} labeled rows exceed n={n}")
    if min_per_tag * len(TAG_GROUPS) + sum(category_quotas.values()) > n:
        raise ValueError(f"Quotas exceed n={n}")

    chosen = df["complaint_id"].isin(labeled)

    def top_up(mask: pd.Series, quota: int) -> None:
        nonlocal chosen
        missing = quota - int((mask & chosen).sum())
        pool = df[mask & ~chosen]
        if missing > 0:
            picked = pool.sample(min(missing, len(pool)), random_state=seed)
            chosen = chosen | df.index.isin(picked.index)

    for tag in TAG_GROUPS:
        top_up(df["tags"].str.contains(tag), min_per_tag)
    for category in sorted(category_quotas):
        top_up(df["category"] == category, category_quotas[category])

    if int(chosen.sum()) > n:
        raise ValueError(f"Quotas plus labeled rows exceed n={n}")
    rest = df[~chosen].sample(n - int(chosen.sum()), random_state=seed)
    out = df[chosen | df.index.isin(rest.index)]
    return out.iloc[out["complaint_id"].astype(int).argsort()].reset_index(drop=True)


def merge_annotations(
    pool_tasks: list[dict], exported: list[dict], selected: list[str]
) -> list[dict]:
    """Build Label Studio import tasks for the selected ids, carrying over finished annotations."""
    results = {}
    for task in exported:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if done:
            results[task["data"]["complaint_id"]] = done[-1]["result"]
    wanted = set(selected)
    out = []
    for task in pool_tasks:
        cid = task["data"]["complaint_id"]
        if cid not in wanted:
            continue
        new = {"data": task["data"]}
        if cid in results:
            new["annotations"] = [{"result": results[cid]}]
        out.append(new)
    return sorted(out, key=lambda t: int(t["data"]["complaint_id"]))


def _choices(name: str, labels: LabelSet) -> str:
    items = "\n".join(
        f'    <Choice value="{escape(lb.id)}" hint="{escape(lb.name)}: {escape(lb.description)}"/>'
        for lb in labels.labels
    )
    return (
        f'  <Choices name="{name}" toName="text" choice="single" required="true">\n'
        f"{items}\n  </Choices>"
    )


def label_studio_config(categories: LabelSet, priorities: LabelSet) -> str:
    return "\n".join(
        [
            "<View>",
            '  <Header value="Complaint"/>',
            '  <Text name="text" value="$text"/>',
            '  <Header value="Category (team)"/>',
            _choices("category", categories),
            '  <Header value="Priority"/>',
            _choices("priority", priorities),
            "</View>",
        ]
    )


def export_annotations(tasks: list[dict], categories: LabelSet, priorities: LabelSet) -> list[dict]:
    """Convert a Label Studio JSON export to golden rows; skip unfinished or cancelled tasks."""
    rows = []
    for task in tasks:
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        values = {r["from_name"]: r["value"]["choices"][0] for r in annotations[-1]["result"]}
        if "category" not in values or "priority" not in values:
            continue
        if values["category"] not in categories.ids or values["priority"] not in priorities.ids:
            raise ValueError(f"unknown label in task {task['data']['complaint_id']}: {values}")
        rows.append(
            {
                "complaint_id": task["data"]["complaint_id"],
                "text": task["data"]["text"],
                "product_original": task["data"]["product"],
                "category": values["category"],
                "priority": values["priority"],
            }
        )
    return sorted(rows, key=lambda r: int(r["complaint_id"]))
