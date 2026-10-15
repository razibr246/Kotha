"""Kotha's agents.

`ClaudeAgent` runs a tool-use loop with Claude. `OfflineAgent` is a deterministic agent with
the same tools and the same grounding rules; it runs when no API key is configured and as a
fallback if the API call fails, so the shop's support never goes dark.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .courier import Courier, get_courier
from .lang import area_name, detect_lang, extract_order_numbers, to_bn_digits
from .tools import TOOL_SPECS, Toolbox

log = logging.getLogger("kotha.agent")


@dataclass
class Turn:
    role: str      # user | assistant
    content: str


@dataclass
class Reply:
    text: str
    lang: str
    sources: dict[str, Any]
    engine: str
    handoff: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


SYSTEM_PROMPT = """You are Kotha, the customer support agent for the online shop "{shop_name}" in Bangladesh.

Answer questions about orders, delivery, payment, returns and refunds.

Grounding rules (strict):
- Every fact about an order must come from get_order / track_shipment results in this conversation. Every policy fact must come from search_help results. If the tools don't say it, you don't know it.
- If a customer mentions an order number, call get_order. If the order is shipped and has a tracking_id, call track_shipment too.
- For times, use the ready-made *_phrase fields in the customer's language rather than converting timestamps yourself.
- If you can't find the order or the answer, say so plainly and offer a human (support phone: {support_phone}). Never invent policies, dates, prices or tracking info.
- Never reveal the customer's full name, full phone number or address. Mention only the delivery area.

Style:
- Reply in {lang_name}. The customer wrote in {lang_name}. If they mix Bangla and English, reply in Bangla script with common English words (order, courier, refund) kept as is.
- Be short and warm, like a helpful shop assistant on chat: 1-3 sentences, no headings, no bullet lists, no markdown.
- Use Bangla digits (০-৯) and ৳ for money when replying in Bangla.
"""

_LANG_NAME = {"bn": "Bangla (বাংলা)", "en": "English"}

_HANDOFF_RE = re.compile(r"human|agent|person|manager|complain|মানুষ|কথা বলতে|অভিযোগ", re.I)


def _shop(shop_id: str) -> dict[str, Any]:
    from .db import connection
    with connection() as conn:
        row = conn.execute("SELECT name, support_phone FROM shops WHERE id = %s", (shop_id,)).fetchone()
    if not row:
        raise LookupError(f"unknown shop {shop_id!r}")
    return row


class ClaudeAgent:
    engine = "claude"

    def __init__(self, api_key: str, model: str, courier: Courier | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.courier = courier or get_courier()

    def answer(self, shop_id: str, message: str, history: list[Turn]) -> Reply:
        lang = detect_lang(message)
        shop = _shop(shop_id)
        tools = Toolbox(shop_id, self.courier, lang)
        system = SYSTEM_PROMPT.format(shop_name=shop["name"], support_phone=shop["support_phone"],
                                      lang_name=_LANG_NAME[lang])
        messages: list[dict[str, Any]] = [{"role": t.role, "content": t.content} for t in history[-10:]]
        messages.append({"role": "user", "content": message})
        calls: list[dict[str, Any]] = []

        for _ in range(settings.max_tool_rounds):
            resp = self.client.messages.create(model=self.model, max_tokens=600, system=system,
                                               tools=TOOL_SPECS, messages=messages)
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                return Reply(text, lang, tools.source_summary(), self.engine,
                             handoff=bool(_HANDOFF_RE.search(message)), tool_calls=calls)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    calls.append({"name": block.name, "input": block.input})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": tools.call(block.name, block.input)})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("tool loop did not finish")


# ---------------------------------------------------------------- offline agent

_POLICY_RE = re.compile(
    r"return|refund|exchange|size|damage|broken|wrong|cancel|warrant|charge|fee|cost|pay|bkash|nagad|cod|cash|policy|"
    r"ফেরত|রিফান্ড|বদল|সাইজ|ভাঙা|ভুল|বাতিল|ক্যানসেল|ওয়ারেন্টি|চার্জ|খরচ|পেমেন্ট|বিকাশ|নগদ|ক্যাশ",
    re.I,
)
_ORDER_TALK_RE = re.compile(
    r"order|parcel|package|deliver|track|where|arrive|ship|kothay|koi|kobe|"
    r"অর্ডার|পার্সেল|ডেলিভারি|কোথায়|কবে|পাব|পাবো|আসবে|পৌঁছ",
    re.I,
)
_TRACK_RE = re.compile(r"where|track|status|kothay|koi|কোথায়|কই|স্ট্যাটাস|ট্র্যাক|my order|amar order|আমার অর্ডার", re.I)
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|salam|assalamu|হাই|হ্যালো|সালাম|আসসালামু)\b", re.I)

T = {
    "en": {
        "not_found": "I couldn't find order #{n} in our system. Could you double-check the number? It's in your confirmation SMS.",
        "pending": "We've got order #{n} and it's being confirmed. It hasn't shipped yet; you'll get an SMS when it does.",
        "confirmed": "Order #{n} is confirmed and being prepared. It hasn't shipped yet.",
        "packed": "Order #{n} is packed and will be handed to {courier} soon.",
        "shipped_transit": "Shipped {shipped}. It reaches {area} by {eta}.",
        "shipped_out": "It's out for delivery in {area} now and should reach you by {eta}.",
        "attempt_failed": "{courier} tried to deliver order #{n} but couldn't reach you. They'll try again; please keep your phone ({phone}) reachable.",
        "delivered": "Order #{n} was delivered {at}.",
        "cancelled": "Order #{n} was cancelled.",
        "cancelled_paid": "Order #{n} was cancelled. Your payment of ৳{total} is being refunded via {method}.",
        "returned": "Order #{n} was returned to us.",
        "cod": " Please keep ৳{total} ready for cash on delivery.",
        "no_tracking": "Order #{n} has shipped with {courier}, but I couldn't get live tracking right now.",
        "ask_order": "Sure, I can check that. What's your order number? It's in your confirmation SMS, like #4821.",
        "no_answer": "I couldn't find that in our help docs. You can reach a person at {phone}, or share your order number and I'll check it.",
        "greeting": "Hi! I'm Kotha. Ask me about an order (like \"Where is order #4821?\") or about delivery, payment and returns.",
        "handoff": "I'll pass this to a person on our team. You can also call {phone}.",
    },
    "bn": {
        "not_found": "#{n} নম্বরের কোনো অর্ডার খুঁজে পাইনি। নম্বরটা একটু মিলিয়ে দেখবেন? কনফার্মেশন SMS-এ আছে।",
        "pending": "অর্ডার #{n} পেয়েছি, কনফার্ম করা হচ্ছে। এখনো শিপ হয়নি; শিপ হলে SMS পাবেন।",
        "confirmed": "অর্ডার #{n} কনফার্ম হয়েছে, প্যাক করা হচ্ছে। এখনো শিপ হয়নি।",
        "packed": "অর্ডার #{n} প্যাক করা হয়েছে, শিগগিরই {courier}-এর কাছে দেওয়া হবে।",
        "shipped_transit": "{shipped} শিপ হয়েছে। {eta}র মধ্যে {area} পৌঁছে যাবে।",
        "shipped_out": "এখন {area}-তে ডেলিভারির জন্য বের হয়েছে, {eta}র মধ্যে পেয়ে যাবেন।",
        "attempt_failed": "{courier} অর্ডার #{n} ডেলিভারি দিতে গিয়েছিল কিন্তু আপনাকে ফোনে পায়নি। আবার চেষ্টা করবে, ফোনটা ({phone}) খোলা রাখবেন।",
        "delivered": "অর্ডার #{n} {at} ডেলিভারি হয়ে গেছে।",
        "cancelled": "অর্ডার #{n} বাতিল করা হয়েছে।",
        "cancelled_paid": "অর্ডার #{n} বাতিল হয়েছে। আপনার ৳{total} {method}-এ রিফান্ড করা হচ্ছে।",
        "returned": "অর্ডার #{n} আমাদের কাছে ফেরত এসেছে।",
        "cod": " ক্যাশ অন ডেলিভারির জন্য ৳{total} রেডি রাখবেন।",
        "no_tracking": "অর্ডার #{n} {courier}-এ শিপ হয়েছে, তবে এই মুহূর্তে লাইভ ট্র্যাকিং পাচ্ছি না।",
        "ask_order": "অবশ্যই দেখে দিচ্ছি। আপনার অর্ডার নম্বরটা দিন? কনফার্মেশন SMS-এ আছে, যেমন #৪৮২১।",
        "no_answer": "এটা আমাদের হেল্প ডকে খুঁজে পাইনি। {phone} নম্বরে আমাদের টিমের সাথে কথা বলতে পারেন, অথবা অর্ডার নম্বর দিলে আমি দেখে দিচ্ছি।",
        "greeting": "হ্যালো! আমি কথা। অর্ডার নিয়ে জিজ্ঞেস করুন (যেমন \"অর্ডার #৪৮২১ কোথায়?\"), অথবা ডেলিভারি, পেমেন্ট, রিটার্ন নিয়ে।",
        "handoff": "আমি এটা আমাদের টিমের একজনের কাছে পাঠাচ্ছি। চাইলে {phone} নম্বরে কলও করতে পারেন।",
    },
}

_METHOD = {"bkash": "bKash", "nagad": "Nagad", "card": "card", "cod": "cash"}


def _num(n: Any, lang: str) -> str:
    s = f"{n:,.0f}" if isinstance(n, (int, float)) else str(n)
    return to_bn_digits(s) if lang == "bn" else s


def _snippet(text: str, limit: int = 320) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M).replace("**", "")
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    out = ""
    for s in sentences:
        if out and len(out) + len(s) > limit:
            break
        out += s + " "
    return out.strip() or text[:limit].rstrip() + "…"


class OfflineAgent:
    engine = "offline"

    def __init__(self, courier: Courier | None = None):
        self.courier = courier or get_courier()

    def _order_reply(self, tools: Toolbox, n: int, lang: str) -> str:
        t = T[lang]
        o = tools.get_order(n)
        nn = to_bn_digits(str(n)) if lang == "bn" else str(n)
        if not o["found"]:
            return t["not_found"].format(n=nn)
        status, courier = o["status"], o["courier"] or "the courier"
        total = _num(o["total_bdt"], lang)
        cod = (o["payment_method"] == "cod" and o["payment_status"] == "unpaid")
        if status in ("pending", "confirmed", "packed"):
            return t[status].format(n=nn, courier=courier)
        if status == "cancelled":
            if o["payment_status"] in ("paid", "refunded") and o["payment_method"] != "cod":
                return t["cancelled_paid"].format(n=nn, total=total,
                                                  method=_METHOD.get(o["payment_method"], o["payment_method"]))
            return t["cancelled"].format(n=nn)
        if status == "returned":
            return t["returned"].format(n=nn)
        tr = tools.track_shipment(o["tracking_id"]) if o["tracking_id"] else {"found": False}
        if not tr["found"]:
            if status == "delivered":
                return t["delivered"].format(n=nn, at=o["last_updated"][:10])
            return t["no_tracking"].format(n=nn, courier=courier)
        if tr["status"] == "delivered":
            return t["delivered"].format(n=nn, at=tr["delivered_phrase"][lang])
        if tr["status"] == "delivery_attempt_failed":
            return t["attempt_failed"].format(n=nn, courier=tr["courier"], phone=o["phone_on_file"])
        key = "shipped_out" if tr["status"] == "out_for_delivery" else "shipped_transit"
        reply = t[key].format(shipped=tr["shipped_phrase"][lang], area=area_name(o["delivery_area"], lang),
                              eta=tr["eta_phrase"][lang])
        if cod:
            reply += t["cod"].format(total=total)
        return reply

    def answer(self, shop_id: str, message: str, history: list[Turn]) -> Reply:
        lang = detect_lang(message)
        shop = _shop(shop_id)
        tools = Toolbox(shop_id, self.courier, lang)
        t = T[lang]
        phone = shop["support_phone"]
        if lang == "bn":
            phone = to_bn_digits(phone)

        if _HANDOFF_RE.search(message):
            return Reply(t["handoff"].format(phone=phone), lang, tools.source_summary(),
                         self.engine, handoff=True)

        numbers = extract_order_numbers(message)
        talks_order = bool(_ORDER_TALK_RE.search(message))
        if not numbers and talks_order:
            # "When will it arrive?" right after the customer gave a number.
            for turn in reversed(history):
                if turn.role == "user" and (prev := extract_order_numbers(turn.content)):
                    numbers = prev[:1]
                    break

        parts: list[str] = []
        if numbers:
            parts.append(self._order_reply(tools, numbers[0], lang))

        tracking_ask = not numbers and talks_order and bool(_TRACK_RE.search(message))
        if _POLICY_RE.search(message) or (not numbers and not tracking_ask):
            query = re.sub(r"#?[\d০-৯]{3,}", " ", message)
            hits = tools.search_help(query, cite=1)["results"]
            if hits:
                parts.append(_snippet(hits[0]["content"]))

        if not parts:
            if _GREETING_RE.search(message):
                parts.append(t["greeting"])
            elif talks_order:
                parts.append(t["ask_order"])
            else:
                parts.append(t["no_answer"].format(phone=phone))
        return Reply(" ".join(parts), lang, tools.source_summary(), self.engine)


class Kotha:
    """Front door: Claude when configured, offline agent otherwise or on failure."""

    def __init__(self, courier: Courier | None = None):
        courier = courier or get_courier()
        self.offline = OfflineAgent(courier)
        self.claude = (ClaudeAgent(settings.anthropic_api_key, settings.model, courier)
                       if settings.anthropic_api_key else None)

    def answer(self, shop_id: str, message: str, history: list[Turn] | None = None) -> Reply:
        history = history or []
        if self.claude:
            try:
                return self.claude.answer(shop_id, message, history)
            except LookupError:
                raise
            except Exception:  # API down, rate limit, etc.
                log.exception("Claude agent failed; using offline agent")
                reply = self.offline.answer(shop_id, message, history)
                reply.engine = "offline-fallback"
                return reply
        return self.offline.answer(shop_id, message, history)
