"""Courier tracking.

`HttpCourier` talks to any courier exposing `GET {base}/track/{tracking_id}` that returns
the normalised shape below. `MockCourier` is the built-in stand-in: it derives a realistic
status timeline from shipped_at / eta in Postgres, so the demo stays "live" as time passes.
Adapters for Pathao / Steadfast / RedX would map their responses into the same shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import httpx
from fastapi import APIRouter, HTTPException

from .config import settings
from .db import connection


class CourierError(Exception):
    pass


class Courier(Protocol):
    def track(self, tracking_id: str) -> dict[str, Any] | None: ...


def build_timeline(row: dict[str, Any], now: datetime) -> dict[str, Any]:
    shipped, eta = row["shipped_at"], row["eta"]
    span = (eta - shipped).total_seconds() or 1
    progress = (now - shipped).total_seconds() / span
    events = [{"at": shipped.isoformat(), "status": "picked_up", "location": "Seller warehouse"}]
    if progress >= 0.15:
        events.append({"at": (shipped + (eta - shipped) * 0.15).isoformat(),
                       "status": "at_hub", "location": row["hub"]})
    if progress >= 0.6:
        events.append({"at": (shipped + (eta - shipped) * 0.6).isoformat(),
                       "status": "out_for_delivery", "location": row["destination_area"]})
    if row["exception"] and progress >= 0.8:
        events.append({"at": (shipped + (eta - shipped) * 0.8).isoformat(),
                       "status": "delivery_attempt_failed", "location": row["destination_area"],
                       "note": row["exception"]})
    elif progress >= 1.0:
        events.append({"at": eta.isoformat(), "status": "delivered", "location": row["destination_area"]})
    last = events[-1]
    return {
        "tracking_id": row["tracking_id"],
        "courier": row["courier"],
        "status": last["status"],
        "current_location": last["location"],
        "destination_area": row["destination_area"],
        "eta": None if last["status"] == "delivered" else eta.isoformat(),
        "events": events,
        "checked_at": now.isoformat(),
    }


class MockCourier:
    def __init__(self, clock=None):
        from . import timefmt
        self.clock = clock or (lambda: timefmt._now())

    def track(self, tracking_id: str) -> dict[str, Any] | None:
        with connection() as conn:
            row = conn.execute(
                "SELECT * FROM courier_shipments WHERE tracking_id = %s", (tracking_id,)
            ).fetchone()
        return build_timeline(row, self.clock()) if row else None


class HttpCourier:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 8.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(headers=headers, timeout=timeout)

    def track(self, tracking_id: str) -> dict[str, Any] | None:
        try:
            r = self.client.get(f"{self.base_url}/track/{tracking_id}")
        except httpx.HTTPError as e:
            raise CourierError(f"courier unreachable: {e}") from e
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise CourierError(f"courier error {r.status_code}")
        return r.json()


def get_courier() -> Courier:
    if settings.courier_api_url:
        return HttpCourier(settings.courier_api_url, settings.courier_api_key)
    return MockCourier()


# HTTP face of the mock, so the HttpCourier path can be exercised end to end.
mock_router = APIRouter(prefix="/mock-courier", tags=["mock courier"])


@mock_router.get("/track/{tracking_id}")
def mock_track(tracking_id: str) -> dict[str, Any]:
    result = MockCourier().track(tracking_id)
    if result is None:
        raise HTTPException(404, "unknown tracking id")
    return result
