"""Jev (TypeSafe System One) zero-shot classifier using the Choice primitive.

`post(body) -> (status_code, json_dict)` is injected so this module has no HTTP dependency.
Transport errors should be reported by `post` as status 599.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from triage_core.calibration import apply_temperature
from triage_core.classify import InvalidLabelError, Prediction, short_hash
from triage_core.labels import LabelSet
from triage_core.retry import RetryableError, call_with_retries

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 599})


class JevClassifier:
    name = "jev"

    def __init__(
        self,
        post: Callable[[dict], tuple[int, dict]],
        instructions: str,
        usd_per_call: float,
        temperature: float = 1.0,
        max_chars: int = 4000,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.post = post
        self.instructions = instructions
        self.usd_per_call = usd_per_call
        self.temperature = temperature
        self.max_chars = max_chars
        self.sleep = sleep
        key_parts: list = [JEV_MODEL, instructions, max_chars]
        if temperature != 1.0:
            # Cached probs are post-temperature; T=1 keeps the legacy key (existing paid caches).
            key_parts.append(float(temperature))
        self.cache_key = "jev-" + short_hash(key_parts)

    def predict(self, texts: list[str], labels: LabelSet) -> list[Prediction]:
        return [self._one(text, labels) for text in texts]

    def _one(self, text: str, labels: LabelSet) -> Prediction:
        body = {
            "model": JEV_MODEL,
            "state": {"ticket": text[: self.max_chars]},
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": labels.descriptions(),
                }
            },
        }

        def call() -> dict:
            status, data = self.post(body)
            if status in RETRYABLE_STATUS:
                raise RetryableError(f"Jev HTTP {status}")
            if status != 200:
                raise RuntimeError(f"Jev HTTP {status}: {data}")
            return data

        start = time.perf_counter()
        data = call_with_retries(
            call, is_retryable=lambda e: isinstance(e, RetryableError), sleep=self.sleep
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = data["answers"]["label"]
        if answer["choice"] not in labels.ids:
            raise InvalidLabelError(f"Jev answered {answer['choice']!r}")
        raw = {k: float(answer["probabilities"].get(k, 0.0)) for k in labels.ids}
        return Prediction(
            answer["choice"],
            apply_temperature(raw, self.temperature),
            cost_usd=self.usd_per_call,
            latency_ms=latency_ms,
        )
