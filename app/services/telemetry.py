from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any


@dataclass(frozen=True)
class _OpenStage:
    started_at_utc: datetime
    started_at_monotonic: float


class InferenceTelemetryCollector:
    def __init__(self) -> None:
        self._open_stages: dict[str, _OpenStage] = {}
        self._stage_rows: list[dict[str, Any]] = []

    def start_stage(self, stage_name: str) -> None:
        self._open_stages[stage_name] = _OpenStage(
            started_at_utc=datetime.now(timezone.utc),
            started_at_monotonic=time.perf_counter(),
        )

    def end_stage(
        self,
        stage_name: str,
        *,
        status: str,
        provider: str | None = None,
        model_name: str | None = None,
        usage: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        started = self._open_stages.pop(stage_name)
        finished_at = datetime.now(timezone.utc)
        duration_ms = int((time.perf_counter() - started.started_at_monotonic) * 1000)
        self._stage_rows.append(
            {
                'stage_name': stage_name,
                'started_at': started.started_at_utc,
                'finished_at': finished_at,
                'duration_ms': max(duration_ms, 0),
                'provider': provider,
                'model_name': model_name,
                'usage_json': usage or {},
                'estimated_cost_usd': estimated_cost_usd,
                'status': status,
                'error_code': error_code,
                'error_message': error_message,
            }
        )

    def stage_rows(self) -> list[dict[str, Any]]:
        return list(self._stage_rows)


def estimate_stage_cost(
    *,
    usage: dict[str, Any] | None,
    unit_type: str | None,
    unit_price_usd: float | None,
) -> float | None:
    if usage is None or unit_type is None or unit_price_usd is None:
        return None

    price = float(unit_price_usd)
    if unit_type == 'token':
        tokens = usage.get('tokens')
        return float(tokens) * price if tokens is not None else None
    if unit_type == 'audio_second':
        seconds = usage.get('audio_seconds')
        return float(seconds) * price if seconds is not None else None
    if unit_type == 'candidate':
        count = usage.get('candidate_count')
        return float(count) * price if count is not None else None
    if unit_type == 'request':
        requests = usage.get('request_count')
        return float(requests) * price if requests is not None else None
    return None
