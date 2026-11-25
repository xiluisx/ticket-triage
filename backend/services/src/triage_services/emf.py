"""CloudWatch Embedded Metric Format: one JSON line per event, metrics without dimensions.

Free tier allows 10 custom metrics, so tenant/level go as log properties (Logs Insights),
never as dimensions. Never pass subject, body, API keys or tokens in `event`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from typing import TextIO

NAMESPACE = "TicketTriage"
METRIC_UNITS = {
    "TicketsProcessed": "Count",
    "ClassificationLatency": "Milliseconds",
    "ClassificationCost": "None",
    "Unclassified": "Count",
    "Errors": "Count",
    "PiiRedacted": "Count",
}


class EmfLogger:
    def __init__(self, stream: TextIO | None = None, clock: Callable[[], float] = time.time):
        self.stream = stream or sys.stdout
        self.clock = clock

    def emit(self, event: dict, metrics: dict[str, float] | None = None) -> None:
        metrics = metrics or {}
        unknown = set(metrics) - set(METRIC_UNITS)
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)}")
        line = dict(event)
        if metrics:
            line["_aws"] = {
                "Timestamp": int(self.clock() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": NAMESPACE,
                        "Dimensions": [[]],
                        "Metrics": [{"Name": n, "Unit": METRIC_UNITS[n]} for n in metrics],
                    }
                ],
            }
            line.update(metrics)
        self.stream.write(json.dumps(line, default=str) + "\n")
        self.stream.flush()
