"""Label sets shared by training, serving and labeling tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        ids = [label.id for label in self.labels]
        if not ids:
            raise ValueError("LabelSet needs at least one label")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate label ids: {ids}")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(label.id for label in self.labels)

    def descriptions(self) -> dict[str, str]:
        return {label.id: label.description for label in self.labels}

    def labels_hash(self) -> str:
        """Short hash of ids, names and descriptions; a model trained on another hash is stale."""
        payload = json.dumps(
            [[lb.id, lb.name, lb.description] for lb in self.labels], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict) -> LabelSet:
        return cls(
            tuple(Label(item["id"], item["name"], item["description"]) for item in data["labels"])
        )

    @classmethod
    def from_json(cls, path: Path) -> LabelSet:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
