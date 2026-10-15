"""The three grounding tools the agent can call, and source bookkeeping."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import helpdocs, orders
from .courier import Courier, CourierError
from .timefmt import part_of_day, when

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": (
            "Look up one of this shop's orders by its order number. Returns status, items, "
            "total, payment, delivery area, courier and tracking id. Call this whenever the "
            "customer mentions an order number. Never guess order details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"order_no": {"type": "integer", "description": "e.g. 4821"}},
            "required": ["order_no"],
        },
    },
    {
        "name": "track_shipment",
        "description": (
            "Live courier tracking for a tracking id from get_order. Returns current status, "
            "location, event history and ETA with ready-made local-time phrases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"tracking_id": {"type": "string"}},
            "required": ["tracking_id"],
        },
    },
    {
        "name": "search_help",
        "description": (
            "Search the shop's own help docs (delivery, returns, refunds, payment, cancellation, "
            "warranty). Use for any policy question. Query in the customer's language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

SOURCE_LABELS = {"order_db": "order DB", "courier": "courier API", "help": "help docs"}


@dataclass
class Source:
    kind: str           # order_db | courier | help
    ref: str            # order no, tracking id, or doc#section

    def label(self) -> str:
        return SOURCE_LABELS[self.kind]


@dataclass
class Toolbox:
    shop_id: str
    courier: Courier
    lang: str = "en"
    sources: list[Source] = field(default_factory=list)

    def _cite(self, kind: str, ref: str) -> None:
        if not any(s.kind == kind and s.ref == ref for s in self.sources):
            self.sources.append(Source(kind, ref))

    def get_order(self, order_no: int) -> dict[str, Any]:
        order = orders.get_order(self.shop_id, int(order_no))
        if order is None:
            return {"found": False, "order_no": order_no,
                    "note": "No order with this number in this shop."}
        self._cite("order_db", f"#{order_no}")
        order["placed_phrase"] = {"en": part_of_day(order["placed_at"], "en"),
                                  "bn": part_of_day(order["placed_at"], "bn")}
        return {"found": True, **order}

    def track_shipment(self, tracking_id: str) -> dict[str, Any]:
        try:
            t = self.courier.track(tracking_id)
        except CourierError as e:
            return {"found": False, "error": str(e),
                    "note": "Courier API is down. Say you couldn't reach the courier right now."}
        if t is None:
            return {"found": False, "tracking_id": tracking_id}
        self._cite("courier", tracking_id)
        shipped = t["events"][0]["at"]
        t["shipped_phrase"] = {"en": part_of_day(shipped, "en"), "bn": part_of_day(shipped, "bn")}
        if t.get("eta"):
            t["eta_phrase"] = {"en": when(t["eta"], "en", round_up=True),
                              "bn": when(t["eta"], "bn", round_up=True)}
        if t["status"] == "delivered":
            t["delivered_phrase"] = {"en": part_of_day(t["events"][-1]["at"], "en"),
                                     "bn": part_of_day(t["events"][-1]["at"], "bn")}
        return {"found": True, **t}

    def search_help(self, query: str, cite: int | None = None) -> dict[str, Any]:
        """Cites every hit by default (the model saw them all); `cite=n` cites the top n."""
        hits = helpdocs.search(self.shop_id, query, lang=self.lang)
        for h in hits[:cite]:
            self._cite("help", f"{h.doc}#{h.section}")
        return {"results": [{"doc": h.doc, "section": h.section, "lang": h.lang,
                             "content": h.content, "score": round(h.score, 3)} for h in hits]}

    def call(self, name: str, args: dict[str, Any]) -> str:
        fn = {"get_order": self.get_order, "track_shipment": self.track_shipment,
              "search_help": self.search_help}.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), ensure_ascii=False, default=str)
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"})

    def source_summary(self) -> dict[str, Any]:
        labels: list[str] = []
        for s in self.sources:
            if s.label() not in labels:
                labels.append(s.label())
        return {"count": len(labels), "labels": labels,
                "items": [{"kind": s.kind, "label": s.label(), "ref": s.ref} for s in self.sources]}
