"""Help-doc ingestion and retrieval over pgvector."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .db import connection
from .embeddings import get_embedder, tokenize

_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    doc: str
    section: str
    lang: str
    content: str
    pos: int = 0


@dataclass
class Hit:
    doc: str
    section: str
    lang: str
    content: str
    score: float
    pos: int = 0


def chunk_markdown(text: str, doc: str, lang: str, max_chars: int = 900) -> list[Chunk]:
    """Split on '## ' headings; long sections are split on paragraph boundaries."""
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc
    parts = _HEADING.split(text)
    chunks: list[Chunk] = []
    intro = parts[0]
    if title_match:
        intro = intro.replace(title_match.group(0), "")
    if intro.strip():
        chunks.append(Chunk(doc, title, lang, intro.strip()))
    for heading, body in zip(parts[1::2], parts[2::2]):
        body = body.strip()
        if not body:
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(Chunk(doc, heading.strip(), lang, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(Chunk(doc, heading.strip(), lang, buf.strip()))
    for i, c in enumerate(chunks):
        c.pos = i
    return chunks


def load_dir(path: Path) -> list[Chunk]:
    """Files are named `<doc>.<lang>.md`, e.g. `returns.bn.md`."""
    chunks: list[Chunk] = []
    for f in sorted(path.glob("*.md")):
        stem_parts = f.stem.split(".")
        doc, lang = (stem_parts[0], stem_parts[1]) if len(stem_parts) > 1 else (f.stem, "en")
        chunks += chunk_markdown(f.read_text(encoding="utf-8"), doc, lang)
    return chunks


def ingest(shop_id: str, chunks: list[Chunk], *, replace: bool = True) -> int:
    embedder = get_embedder()
    # The heading is embedded with the body so short sections stay findable.
    vectors = embedder.embed([f"{c.section}\n{c.content}" for c in chunks], kind="document")
    with connection() as conn:
        if replace:
            conn.execute("DELETE FROM help_chunks WHERE shop_id = %s", (shop_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO help_chunks (shop_id, doc, section, lang, pos, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::vector)""",
                [(shop_id, c.doc, c.section, c.lang, c.pos, c.content, v)
                 for c, v in zip(chunks, vectors)],
            )
        conn.commit()
    return len(chunks)


def _stems(text: str) -> list[str]:
    """Crude stemming: first 4 characters. Folds refund/refunds, ফেরত/ফেরতের."""
    return [w[:4] for w in tokenize(text) if len(w) > 1]


class _Lexicon:
    """BM25 statistics over one shop's help chunks, cached until the docs change."""

    def __init__(self, rows: list[dict]):
        self.docs = {(r["doc"], r["pos"], r["lang"]): _stems(f"{r['section']} {r['content']}")
                     for r in rows}
        self.avg = sum(map(len, self.docs.values())) / max(1, len(self.docs))
        df: dict[str, int] = {}
        for terms in self.docs.values():
            for t in set(terms):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def bm25(self, key: tuple, query: list[str], k1: float = 1.2, b: float = 0.75) -> float:
        terms = self.docs.get(key, [])
        score = 0.0
        for q in set(query):
            tf = terms.count(q)
            if tf:
                score += self.idf.get(q, 0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(terms) / self.avg))
        return score


_lexicons: dict[str, tuple[tuple, _Lexicon]] = {}


def _lexicon(shop_id: str) -> _Lexicon:
    with connection() as conn:
        stamp = conn.execute(
            "SELECT count(*) AS n, coalesce(max(id), 0) AS m FROM help_chunks WHERE shop_id = %s",
            (shop_id,),
        ).fetchone()
        key = (stamp["n"], stamp["m"])
        cached = _lexicons.get(shop_id)
        if cached and cached[0] == key:
            return cached[1]
        rows = conn.execute(
            "SELECT doc, pos, lang, section, content FROM help_chunks WHERE shop_id = %s", (shop_id,)
        ).fetchall()
    lex = _Lexicon(rows)
    _lexicons[shop_id] = (key, lex)
    return lex


def search(shop_id: str, query: str, *, lang: str | None = None, k: int = 3,
           min_score: float = 0.25) -> list[Hit]:
    """Hybrid retrieval: pgvector cosine + BM25, then the customer's language.

    The two signals are blended after scaling BM25 to 0..1 within the candidates. With the
    local hash embedder a hit must share at least one word stem with the query, which stops
    "what is the meaning of life" from matching the refund policy.
    """
    embedder = get_embedder()
    [qvec] = embedder.embed([query], kind="query")
    with connection() as conn:
        rows = conn.execute(
            """SELECT doc, section, lang, pos, content, 1 - (embedding <=> %s::vector) AS score
               FROM help_chunks WHERE shop_id = %s
               ORDER BY embedding <=> %s::vector LIMIT 40""",
            (qvec, shop_id, qvec),
        ).fetchall()
    lex = _lexicon(shop_id)
    q = _stems(query)
    scored = []
    for r in rows:
        bm = lex.bm25((r["doc"], r["pos"], r["lang"]), q)
        if embedder.name == "hash" and bm == 0:
            continue
        scored.append((r, float(r["score"]), bm))
    if not scored:
        return []
    top_bm = max(bm for _, _, bm in scored) or 1.0
    w_lex = 0.6 if embedder.name == "hash" else 0.3
    hits = []
    for r, cos, bm in scored:
        blended = (1 - w_lex) * cos + w_lex * (bm / top_bm) * min(1.0, top_bm / 3)
        hits.append(Hit(r["doc"], r["section"], r["lang"], r["content"], blended, r["pos"]))
    hits = [h for h in hits if h.score >= min_score]
    hits.sort(key=lambda h: h.score + (0.05 if lang and h.lang == lang else 0.0), reverse=True)
    if lang:
        hits = _prefer_lang(shop_id, hits, lang)
    # The same section in two languages is one answer, not two.
    seen: set[tuple[str, int]] = set()
    out: list[Hit] = []
    for h in hits:
        if (h.doc, h.pos) not in seen:
            seen.add((h.doc, h.pos))
            out.append(h)
    return out[:k]


def _prefer_lang(shop_id: str, hits: list[Hit], lang: str) -> list[Hit]:
    """Swap each hit for its translation in the customer's language, when one exists."""
    swap = [(h.doc, h.pos) for h in hits if h.lang != lang]
    if not swap:
        return hits
    with connection() as conn:
        rows = conn.execute(
            """SELECT doc, section, lang, pos, content FROM help_chunks
               WHERE shop_id = %s AND lang = %s AND (doc, pos) IN (SELECT * FROM unnest(%s::text[], %s::int[]))""",
            (shop_id, lang, [d for d, _ in swap], [p for _, p in swap]),
        ).fetchall()
    twin = {(r["doc"], r["pos"]): r for r in rows}
    out = []
    for h in hits:
        t = twin.get((h.doc, h.pos)) if h.lang != lang else None
        out.append(Hit(t["doc"], t["section"], t["lang"], t["content"], h.score, t["pos"]) if t else h)
    return out
