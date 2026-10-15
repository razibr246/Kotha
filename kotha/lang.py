"""Language helpers for mixed Bangla / English / Banglish input."""
from __future__ import annotations

import re

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_EN_TO_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_BENGALI = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")

# Romanised Bangla ("Banglish") words that are common in support chats.
_BANGLISH = {
    "amar", "amr", "order", "kothay", "koi", "kobe", "pabo", "asbe", "ase", "ache", "ki",
    "keno", "taka", "vai", "bhai", "apu", "deliver", "hobe", "korbo", "kivabe", "parcel",
    "ekhono", "akhono", "koto", "lage", "lagbe", "din", "kemne", "jabe", "paben", "kore", "pai", "nai", "naai", "fert", "ferot", "dite", "chai",
}


def normalize_digits(text: str) -> str:
    """Bangla digits → ASCII, so '#৪৮২১' and '#4821' look the same to the tools."""
    return text.translate(_BN_DIGITS)


def to_bn_digits(text: str) -> str:
    return text.translate(_EN_TO_BN_DIGITS)


def detect_lang(text: str) -> str:
    """Return 'bn' or 'en'.

    Bengali script wins if it makes up a meaningful share of the letters. Romanised
    Bangla is answered in Bangla too, because that is what the customer is thinking in.
    """
    bn = len(_BENGALI.findall(text))
    en = len(_LATIN.findall(text))
    if bn and bn >= 0.3 * (bn + en):
        return "bn"
    words = re.findall(r"[a-z]+", text.lower())
    if words:
        hits = sum(w in _BANGLISH for w in words)
        english_markers = {"where", "is", "my", "the", "when", "what", "how", "can", "i", "will"}
        if hits >= 2 and not english_markers.intersection(words):
            return "bn"
    return "en"


_ORDER_RE = re.compile(
    r"(?:#|order\s*(?:no\.?|number|id)?\s*[:#]?\s*|অর্ডার\s*(?:নং|নম্বর)?\s*[:#]?\s*)(\d{3,8})",
    re.IGNORECASE,
)
_BARE_NUM = re.compile(r"\b(\d{4,8})\b")


def extract_order_numbers(text: str) -> list[int]:
    text = normalize_digits(text)
    found = [int(m) for m in _ORDER_RE.findall(text)]
    if not found:
        # Only fall back to bare numbers that don't look like phone numbers.
        found = [int(m) for m in _BARE_NUM.findall(text) if len(m) <= 6]
    seen: list[int] = []
    for n in found:
        if n not in seen:
            seen.append(n)
    return seen


# Common delivery areas, so Bangla replies don't switch script mid-sentence.
_AREAS_BN = {
    "dhaka": "ঢাকা", "mirpur": "মিরপুর", "dhanmondi": "ধানমন্ডি", "uttara": "উত্তরা",
    "banani": "বনানী", "gulshan": "গুলশান", "mohammadpur": "মোহাম্মদপুর", "badda": "বাড্ডা",
    "motijheel": "মতিঝিল", "savar": "সাভার", "gazipur": "গাজীপুর", "narayanganj": "নারায়ণগঞ্জ",
    "chattogram": "চট্টগ্রাম", "chittagong": "চট্টগ্রাম", "sylhet": "সিলেট", "khulna": "খুলনা",
    "rajshahi": "রাজশাহী", "barishal": "বরিশাল", "rangpur": "রংপুর", "mymensingh": "ময়মনসিংহ",
    "cumilla": "কুমিল্লা", "comilla": "কুমিল্লা", "bashundhara": "বসুন্ধরা",
}


def area_name(area: str, lang: str) -> str:
    return _AREAS_BN.get(area.strip().lower(), area) if lang == "bn" else area
