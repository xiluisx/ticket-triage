"""Temperature scaling for probability dicts (used on Jev outputs)."""


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature == 1.0:
        return dict(probs)
    scaled = {k: max(p, 1e-12) ** (1.0 / temperature) for k, p in probs.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}
