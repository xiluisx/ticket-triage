"""Cascade diagram for the poster/report: PII -> Classic -> Jev -> Claude Haiku 4.5.

uv run python scripts/cascade_diagram.py  ->  docs/figures/cascade_diagram.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "cascade_diagram.png"

STEPS = [
    ("Redacción\nde PII", "regex · email, tel., SSN,\ntarjeta (Luhn), cuenta", "#556270", None),
    ("Clásico", "TF-IDF + SVM calibrado\n6 ms · $0", "#1e7a4c", "89%"),
    ("Jev", "TypeSafe zero-shot\n~170 ms · $0.00006", "#a86b00", "7%"),
    ("Claude\nHaiku 4.5", "Amazon Bedrock\n~740 ms · $0.0009", "#6b3fb5", "4%"),
]

fig, ax = plt.subplots(figsize=(15, 4.2), dpi=200)
ax.set_xlim(0, 15)
ax.set_ylim(0, 4.2)
ax.axis("off")

w, h, gap, y = 2.7, 1.9, 1.3, 1.6
xs = [0.25 + i * (w + gap) for i in range(len(STEPS))]
for x, (title, sub, color, coverage) in zip(xs, STEPS, strict=True):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=color, ec="none"
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=19,
        fontweight="bold",
        linespacing=1.05,
    )
    ax.text(
        x + w / 2,
        y + h * 0.2,
        sub,
        ha="center",
        va="center",
        color="white",
        fontsize=11.5,
        linespacing=1.2,
    )
    if coverage:
        ax.add_patch(
            FancyArrowPatch(
                (x + w / 2, y - 0.05),
                (x + w / 2, 0.75),
                arrowstyle="-|>",
                mutation_scale=18,
                color=color,
                lw=2,
            )
        )
        ax.text(
            x + w / 2,
            0.42,
            f"responde {coverage}",
            ha="center",
            va="center",
            color=color,
            fontsize=13.5,
            fontweight="bold",
        )

labels = ["texto\nlimpio", "confianza\n< umbral", "confianza\n< umbral"]
for i, label in enumerate(labels):
    x0, x1 = xs[i] + w + 0.06, xs[i + 1] - 0.06
    ax.add_patch(
        FancyArrowPatch(
            (x0, y + h / 2),
            (x1, y + h / 2),
            arrowstyle="-|>",
            mutation_scale=22,
            color="#556270",
            lw=2.2,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y + h / 2 - 0.22,
        label,
        ha="center",
        va="top",
        color="#556270",
        fontsize=10.5,
        style="italic",
    )

ax.text(
    7.5,
    3.95,
    "Cada nivel responde solo si su confianza calibrada supera el umbral; si no, el ticket escala",
    ha="center",
    va="center",
    fontsize=13,
    color="#17212b",
)
ax.text(
    0.3,
    0.42,
    "Cobertura de la cascada\nde categoría (validación)",
    ha="left",
    va="center",
    fontsize=9,
    color="#8a96a3",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
