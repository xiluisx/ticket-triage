# %% [markdown]
# # 01 · EDA — CFPB complaint narratives (2024-01 → 2026-08)
# Fuente: CFPB Narratives Archive. Objetivo: entender volumen, desbalance, longitud y grupos
# de equidad, y validar el mapeo producto → 8 categorías (`datasets/categories_v1.json`).
#
# Memoria: las narrativas nunca se cargan completas (2.16M textos > 4 GB); se leen por lotes.
# A pandas solo pasan columnas pequeñas como categorías.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from training.cfpb import load_product_map, map_categories, narrative_word_counts, read_narratives

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data" / "cfpb" / "complaints.parquet"
df = pq.read_table(DATA, columns=["date_received", "product", "tags"]).to_pandas(
    strings_to_categorical=True
)
df["words"] = narrative_word_counts(DATA)

product_map = load_product_map(ROOT / "datasets" / "categories_v1.json")
df["category"] = map_categories(
    df["product"].astype(str), product_map
)  # 8 values: plain str is cheap
df["month"] = df["date_received"].dt.to_period("M").dt.to_timestamp()
date_range = f"{df['date_received'].min().date()} → {df['date_received'].max().date()}"
print(f"{len(df):,} complaints with narrative | {date_range}")

# %% [markdown]
# ## Productos sin mapear (debe ser 0)

# %%
unmapped = df[df["category"].isna()]["product"].value_counts()
unmapped = unmapped[unmapped > 0]
print(unmapped if len(unmapped) else "All products mapped ✔")
assert unmapped.empty, "Agregar productos faltantes a cfpb_product_map"

# %% [markdown]
# ## Volumen por mes

# %%
fig, ax = plt.subplots(figsize=(10, 3.5))
df.groupby("month").size().plot(ax=ax, marker="o", title="Complaints with narrative per month")
ax.set_xlabel("")
ax.set_ylabel("complaints")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_volume_by_month.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Desbalance de categorías

# %%
counts = df["category"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
counts.sort_values().plot.barh(ax=ax, title="Complaints per category")
for i, v in enumerate(counts.sort_values()):
    ax.text(v, i, f" {v:,} ({v / len(df):.0%})", va="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_category_balance.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Mapeo producto → categoría

# %%
mapping_table = (
    df.groupby(["category", "product"], observed=True).size().rename("complaints").reset_index()
)
mapping_table

# %% [markdown]
# ## Longitud de narrativas

# %%
print(df["words"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]).round(0))
fig, ax = plt.subplots(figsize=(8, 3.5))
df["words"].clip(upper=1500).plot.hist(
    ax=ax, bins=60, title="Words per narrative (clipped at 1500)"
)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_narrative_length.png", dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# ## Grupos de equidad (`Tags`)

# %%
tags = df["tags"].astype(str)
df["older_american"] = tags.str.contains("Older American")
df["servicemember"] = tags.str.contains("Servicemember")
print(tags.replace("", "(none)").value_counts())
by_cat = (
    df.groupby("category", observed=True)[["older_american", "servicemember"]]
    .mean()
    .mul(100)
    .round(1)
)
fig, ax = plt.subplots(figsize=(8, 4))
by_cat.plot.barh(ax=ax, title="% of complaints tagged, by category")
ax.set_xlabel("%")
plt.tight_layout()
plt.savefig(FIG_DIR / "eda_tags_by_category.png", dpi=150)
plt.show()
plt.close(fig)
by_cat

# %% [markdown]
# ## Ejemplos por categoría (texto ya enmascarado por CFPB con XXXX)

# %%
example_rows = df.groupby("category", observed=True).sample(1, random_state=1)
examples = read_narratives(DATA, example_rows.index.to_numpy())
for cat, text in zip(example_rows["category"], examples, strict=True):
    print(f"--- {cat} ---\n{text[:400]}\n")

# %%
summary = {
    "rows": len(df),
    "date_min": str(df["date_received"].min().date()),
    "date_max": str(df["date_received"].max().date()),
    "categories": {str(k): int(v) for k, v in counts.items()},
    "tags": {str(k): int(v) for k, v in tags.replace("", "(none)").value_counts().items()},
}
print(json.dumps(summary, indent=2))
