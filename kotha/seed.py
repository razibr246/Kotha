"""Seed a demo shop: orders, courier shipments and help docs.

    python -m kotha.seed            # create/refresh demo data
    python -m kotha.seed --reset    # drop all tables first
    python -m kotha.seed --docs-only --shop myshop --docs path/to/help
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings
from .db import close_pool, connection, drop_all, init_schema
from .embeddings import get_embedder
from .helpdocs import ingest, load_dir

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ROOT / "data" / "help"


def demo_rows(now: datetime) -> tuple[list[tuple], list[tuple]]:
    """Orders and shipments positioned relative to `now`, so the demo always reads live."""
    tz = settings.timezone
    now = now.astimezone(tz)
    hour = lambda d: d.replace(minute=0, second=0, microsecond=0)  # noqa: E731
    morning = now.replace(hour=8, minute=0, second=0, microsecond=0)
    shipped_4821 = morning if now >= morning + timedelta(hours=1) else now - timedelta(hours=1)
    five_pm = now.replace(hour=17, minute=0, second=0, microsecond=0)
    eta_4821 = five_pm if now < five_pm - timedelta(hours=1) else five_pm + timedelta(days=1)

    def item(name, qty, price):
        return {"name": name, "qty": qty, "price_bdt": price}

    orders = [
        # order_no, name, phone, area, items, total, method, pay_status, status, courier, tracking, created
        (4821, "Nusrat Jahan", "01712345821", "Mirpur",
         [item("Jamdani saree, teal", 1, 3200), item("Cotton blouse piece", 1, 450)], 3650,
         "bkash", "paid", "shipped", "Pathao", "PTH-88214821", shipped_4821 - timedelta(hours=14)),
        (4810, "Rafiq Ahmed", "01811004810", "Dhanmondi",
         [item("Leather sandal, size 42", 1, 1850)], 1920,
         "cod", "unpaid", "shipped", "Pathao", "PTH-88214810", now - timedelta(days=1, hours=6)),
        (4807, "Tania Akter", "01911224807", "Uttara",
         [item("Nakshi kantha cushion cover", 2, 650)], 1370,
         "cod", "unpaid", "shipped", "Pathao", "PTH-88214807", now - timedelta(days=1, hours=8)),
        (4815, "Imran Hossain", "01556784815", "Chattogram",
         [item("Panjabi, white, L", 1, 2400)], 2400,
         "nagad", "paid", "delivered", "Steadfast", "SF-55124815", now - timedelta(days=5)),
        (4822, "Sadia Islam", "01633334822", "Banani",
         [item("Block-print kurti, M", 2, 1100)], 2270,
         "cod", "unpaid", "pending", None, None, now - timedelta(minutes=40)),
        (4830, "Arif Khan", "01799994830", "Sylhet",
         [item("Muslin scarf", 1, 980)], 1110,
         "bkash", "paid", "packed", "Steadfast", None, now - timedelta(hours=5)),
        (4799, "Mitu Rahman", "01300004799", "Gulshan",
         [item("Silk tie, maroon", 1, 1200)], 1270,
         "bkash", "paid", "cancelled", None, None, now - timedelta(days=3)),
    ]
    shipments = [
        ("PTH-88214821", "Pathao", "Mirpur", shipped_4821, eta_4821, "Pathao Tejgaon hub", None),
        ("PTH-88214810", "Pathao", "Dhanmondi", hour(now) - timedelta(hours=20), hour(now) + timedelta(hours=5),
         "Pathao Tejgaon hub", None),
        ("PTH-88214807", "Pathao", "Uttara", hour(now) - timedelta(hours=20), hour(now) + timedelta(hours=3),
         "Pathao Tejgaon hub", "customer_unreachable"),
        ("SF-55124815", "Steadfast", "Chattogram", now - timedelta(days=4), now - timedelta(days=2),
         "Steadfast Chattogram hub", None),
    ]
    return orders, shipments


def seed_demo(shop_id: str = "demo", now: datetime | None = None, docs: Path = DEFAULT_DOCS) -> None:
    now = now or datetime.now(settings.timezone)
    orders, shipments = demo_rows(now)
    with connection() as conn:
        conn.execute(
            """INSERT INTO shops (id, name, support_phone) VALUES (%s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
                                              support_phone = EXCLUDED.support_phone""",
            (shop_id, "Nokshi Ghor", "09611-482100"),
        )
        conn.execute("DELETE FROM orders WHERE shop_id = %s", (shop_id,))
        for o in orders:
            conn.execute(
                """INSERT INTO orders (shop_id, order_no, customer_name, customer_phone, delivery_area,
                       items, total_bdt, payment_method, payment_status, status, courier, tracking_id,
                       created_at, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (shop_id, o[0], o[1], o[2], o[3], json.dumps(o[4], ensure_ascii=False), o[5], o[6],
                 o[7], o[8], o[9], o[10], o[11], now),
            )
        for s in shipments:
            conn.execute(
                """INSERT INTO courier_shipments
                       (tracking_id, courier, destination_area, shipped_at, eta, hub, exception)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (tracking_id) DO UPDATE SET shipped_at = EXCLUDED.shipped_at,
                       eta = EXCLUDED.eta, exception = EXCLUDED.exception""",
                s,
            )
        conn.commit()
    n = ingest(shop_id, load_dir(docs))
    print(f"seeded shop {shop_id!r}: {len(orders)} orders, {len(shipments)} shipments, {n} help chunks")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="drop all tables first")
    p.add_argument("--shop", default="demo")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p.add_argument("--docs-only", action="store_true", help="only (re)index help docs for --shop")
    a = p.parse_args()
    if a.reset:
        drop_all()
    init_schema(get_embedder().dim)
    if a.docs_only:
        print(f"indexed {ingest(a.shop, load_dir(a.docs))} help chunks for {a.shop!r}")
    else:
        seed_demo(a.shop, docs=a.docs)
    close_pool()


if __name__ == "__main__":
    main()
