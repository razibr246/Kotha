"""Human time phrases in shop-local time, in English and Bangla."""
from __future__ import annotations

from datetime import datetime, timedelta

from .config import settings
from .lang import to_bn_digits

def _now() -> datetime:
    """Single clock for relative phrases; tests patch this."""
    return datetime.now(settings.timezone)


_BN_WEEKDAY = ["সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার"]


def _day_word(dt: datetime, now: datetime, lang: str) -> str:
    delta = (dt.date() - now.date()).days
    if lang == "bn":
        return {0: "আজ", 1: "আগামীকাল", -1: "গতকাল"}.get(delta, _BN_WEEKDAY[dt.weekday()])
    return {0: "today", 1: "tomorrow", -1: "yesterday"}.get(delta, dt.strftime("%A"))


def _clock(dt: datetime, lang: str) -> str:
    h, m = dt.hour, dt.minute
    if lang == "bn":
        period = ("রাত" if h < 4 else "ভোর" if h < 6 else "সকাল" if h < 12 else
                  "দুপুর" if h < 15 else "বিকেল" if h < 18 else "সন্ধ্যা" if h < 20 else "রাত")
        h12 = h % 12 or 12
        t = f"{h12}টা" if m == 0 else f"{h12}:{m:02d}"
        return f"{period} {to_bn_digits(t)}"
    h12 = h % 12 or 12
    suffix = "am" if h < 12 else "pm"
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def when(iso: str, lang: str, now: datetime | None = None, *, round_up: bool = False) -> str:
    """'5 pm today' / 'আজ বিকেল ৫টা'. round_up turns an ETA into a whole-hour 'by' time."""
    tz = settings.timezone
    dt = datetime.fromisoformat(iso).astimezone(tz)
    if round_up and (dt.minute or dt.second):
        dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    now = (now or _now()).astimezone(tz)
    day, clock = _day_word(dt, now, lang), _clock(dt, lang)
    if lang == "en" and abs((dt.date() - now.date()).days) <= 1:
        return f"{clock} {day}"
    return f"{day} {clock}"


def part_of_day(iso: str, lang: str, now: datetime | None = None) -> str:
    """'this morning' / 'আজ সকালে' — used for when something happened."""
    tz = settings.timezone
    dt = datetime.fromisoformat(iso).astimezone(tz)
    now = (now or _now()).astimezone(tz)
    delta = (now.date() - dt.date()).days
    h = dt.hour
    if lang == "bn":
        slot = "সকালে" if h < 12 else "দুপুরে" if h < 15 else "বিকেলে" if h < 18 else "রাতে"
        day = {0: "আজ", 1: "গতকাল"}.get(delta, to_bn_digits(dt.strftime("%d/%m")) + " তারিখ")
        return f"{day} {slot}"
    slot = "morning" if h < 12 else "afternoon" if h < 18 else "evening"
    if delta == 0:
        return f"this {slot}"
    if delta == 1:
        return f"yesterday {slot}"
    return f"on {dt.strftime('%d %b')}"
