"""Read-only access to the shop's live order data."""
from __future__ import annotations

from typing import Any

from .db import connection


def mask_phone(phone: str) -> str:
    digits = [c for c in phone if c.isdigit()]
    return "•••• " + "".join(digits[-3:]) if len(digits) >= 3 else "••••"


def get_order(shop_id: str, order_no: int) -> dict[str, Any] | None:
    """Order summary safe to show a customer: no full name, phone or street address."""
    with connection() as conn:
        row = conn.execute(
            """SELECT order_no, customer_name, customer_phone, delivery_area, items, total_bdt,
                      payment_method, payment_status, status, courier, tracking_id,
                      created_at, updated_at
               FROM orders WHERE shop_id = %s AND order_no = %s""",
            (shop_id, order_no),
        ).fetchone()
    if not row:
        return None
    return {
        "order_no": row["order_no"],
        "status": row["status"],
        "items": row["items"],
        "total_bdt": float(row["total_bdt"]),
        "payment_method": row["payment_method"],
        "payment_status": row["payment_status"],
        "delivery_area": row["delivery_area"],
        "courier": row["courier"],
        "tracking_id": row["tracking_id"],
        "placed_at": row["created_at"].isoformat(),
        "last_updated": row["updated_at"].isoformat(),
        "customer_first_name": row["customer_name"].split()[0],
        "phone_on_file": mask_phone(row["customer_phone"]),
    }
