"""Deployed architecture with official AWS icons (from the `diagrams` package), no Graphviz.

uv run --no-project --with diagrams --with matplotlib python scripts/architecture_diagram.py
-> docs/figures/architecture.png
"""

from pathlib import Path

import diagrams
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ICONS = Path(diagrams.__file__).resolve().parent.parent / "resources"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "architecture.png"

INK, MUTED, AWS_ORANGE = "#232f3e", "#5b6676", "#ff9900"
ICON = 1.0  # icon size in data units (1 unit ~ 1 inch)

NODES = {
    # key: (icon path, x, y, label)
    "client": ("aws/general/users.png", 1.1, 4.6, "Navegador /\ncliente HTTP"),
    "github": ("onprem/vcs/github.png", 1.1, 7.6, "GitHub Actions\nCI · OIDC"),
    "terraform": ("onprem/iac/terraform.png", 1.1, 1.6, "Terraform\n(IaC)"),
    "ecr": ("aws/compute/ec2-container-registry.png", 5.0, 7.3, "ECR\nimagen del servicio"),
    "lambda": (
        "aws/compute/lambda.png",
        5.0,
        4.6,
        "Lambda\nFastAPI · PII · cascada\nclásico en la imagen",
    ),
    "cloudwatch": (
        "aws/management/cloudwatch.png",
        5.0,
        1.6,
        "CloudWatch\nlogs · métricas EMF\ndashboard · alarmas",
    ),
    "bedrock": ("aws/ml/bedrock.png", 8.7, 7.3, "Bedrock\nClaude Haiku 4.5"),
    "dynamodb": ("aws/database/dynamodb.png", 8.7, 4.6, "DynamoDB\ntickets"),
    "ssm": (
        "aws/management/systems-manager-parameter-store.png",
        8.7,
        1.6,
        "SSM Parameter\nStore · key de Jev",
    ),
    "s3": ("aws/storage/simple-storage-service-s3.png", 12.4, 7.3, "S3\ndatos · estado TF"),
    "budgets": ("aws/cost/budgets.png", 12.4, 4.6, "Budgets\n$5 / mes"),
    "sns": ("aws/integration/simple-notification-service-sns.png", 12.4, 1.6, "SNS\nalertas"),
    "email": ("aws/general/user.png", 15.6, 1.6, "Email del\nresponsable"),
}


def icon(ax, path: str, x: float, y: float, label: str) -> None:
    img = mpimg.imread(ICONS / path)
    ax.imshow(img, extent=(x - ICON / 2, x + ICON / 2, y - ICON / 2, y + ICON / 2), zorder=3)
    ax.text(
        x,
        y - ICON / 2 - 0.1,
        label,
        ha="center",
        va="top",
        fontsize=10.5,
        color=INK,
        linespacing=1.15,
        zorder=3,
    )


def arrow(ax, a, b, label=None, rad=0.0, color=MUTED, offset=(0, 0.18), shrink=34):
    (x0, y0), (x1, y1) = a, b
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            lw=1.8,
            shrinkA=shrink,
            shrinkB=shrink,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )
    if label:
        ax.text(
            (x0 + x1) / 2 + offset[0],
            (y0 + y1) / 2 + offset[1],
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=color,
            style="italic",
            zorder=4,
            bbox={"fc": "white", "ec": "none", "pad": 1.2, "alpha": 0.9},
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 11.2), dpi=170)
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.55, 10.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # AWS cloud frame (draw.io style) and the external SaaS box above it.
    ax.add_patch(
        FancyBboxPatch(
            (2.6, -0.45),
            11.6,
            9.35,
            boxstyle="round,pad=0,rounding_size=0.12",
            fc="#fbfbfc",
            ec=INK,
            lw=1.4,
            zorder=0,
        )
    )
    ax.text(
        2.8, 8.68, "AWS Cloud · us-east-1", fontsize=12.5, fontweight="bold", color=INK, va="center"
    )
    ax.add_patch(
        FancyBboxPatch(
            (7.2, 9.25),
            3.0,
            1.3,
            boxstyle="round,pad=0,rounding_size=0.12",
            fc="#fff7e8",
            ec=AWS_ORANGE,
            lw=1.4,
            ls="--",
            zorder=0,
        )
    )
    ax.text(
        8.7, 10.1, "Jev", ha="center", va="center", fontsize=17, fontweight="bold", color="#a86b00"
    )
    ax.text(
        8.7, 9.6, "TypeSafe (zero-shot, externo)", ha="center", va="center", fontsize=9.5, color=INK
    )

    for path, x, y, label in NODES.values():
        icon(ax, path, x, y, label)
    p = {k: (v[1], v[2]) for k, v in NODES.items()}

    def below(key, depth):  # point just under a node's label
        x, y = p[key]
        return (x, y - ICON / 2 - depth)

    arrow(ax, p["client"], p["lambda"], "HTTPS · Function URL\nX-Api-Key", offset=(-0.1, 0.12))
    arrow(ax, below("ecr", 0.62), (5.0, 5.15), "imagen", offset=(0.45, -0.15), shrink=2)
    arrow(ax, p["lambda"], p["bedrock"], "3. nivel", offset=(-0.45, 0.05))
    arrow(ax, (5.35, 5.15), (8.0, 9.25), "2. nivel", offset=(-0.55, 0.1), shrink=2)
    arrow(ax, p["lambda"], p["dynamodb"], "guarda ticket", offset=(0, 0.1))
    arrow(ax, (5.55, 4.55), (8.2, 2.05), "lee key", offset=(0.55, -0.05), shrink=2)
    arrow(ax, below("lambda", 0.95), (5.0, 2.15), "EMF", offset=(0.35, -0.15), shrink=2)
    arrow(ax, below("budgets", 0.62), (12.4, 2.15), "50/80/100%", offset=(0.6, -0.15), shrink=2)
    arrow(ax, below("cloudwatch", 0.95), (11.9, 1.2), None, rad=0.28, shrink=2)
    ax.text(
        8.7,
        -0.32,
        "alarmas: Errors > 0, Unclassified > 0",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
        bbox={"fc": "#fbfbfc", "ec": "none", "pad": 1.2},
    )
    arrow(ax, p["sns"], p["email"])
    arrow(ax, (1.75, 7.6), (2.6, 7.6), shrink=2)
    arrow(ax, (1.75, 1.6), (2.6, 1.6), shrink=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
