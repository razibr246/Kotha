"""Text embedders.

`HashEmbedder` needs no network or model download: it hashes word unigrams/bigrams and
character trigrams into a fixed vector. It handles Bengali script fine and is good for
same-language retrieval. For cross-lingual retrieval (an English question matching a
Bangla-only doc) switch to `VoyageEmbedder` with a multilingual model.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Protocol, Sequence

import httpx

from .config import settings
from .lang import normalize_digits

_WORD = re.compile(r"[\wঀ-৿]+", re.UNICODE)

# Words that carry no retrieval signal. Kept short on purpose.
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "my", "i",
    "you", "it", "order", "please", "pls", "can", "do", "does", "what", "how", "your", "we", "our", "be", "with",
    "আমি", "আমার", "কি", "কী", "এর", "ও", "এবং", "করে", "হবে", "আছে", "যে", "এই", "না", "অর্ডার",
}


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", normalize_digits(text.lower()))
    return [w for w in _WORD.findall(text) if w not in _STOP]


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = tokenize(text)
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"<{w}>"
            # Light stemming via trigrams: "refund"/"refunds", "ফেরত"/"ফেরতের".
            feats += [(f"c:{padded[i:i + 3]}", 0.35) for i in range(max(1, len(padded) - 2))]
        for feat, weight in feats:
            idx, sign = self._bucket(feat)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        return [self._one(t) for t in texts]


class VoyageEmbedder:
    name = "voyage"
    _DIMS = {"voyage-multilingual-2": 1024, "voyage-3": 1024, "voyage-3-lite": 512, "voyage-3.5": 1024}

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.dim = self._DIMS.get(model, 1024)

    def embed(self, texts: Sequence[str], *, kind: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            r = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": list(texts[i:i + 64]), "model": self.model,
                      "input_type": "query" if kind == "query" else "document"},
                timeout=30,
            )
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return out


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        if settings.embedder == "voyage":
            if not settings.voyage_api_key:
                raise RuntimeError("KOTHA_EMBEDDER=voyage needs VOYAGE_API_KEY")
            _embedder = VoyageEmbedder(settings.voyage_api_key, settings.voyage_model)
        else:
            _embedder = HashEmbedder()
    return _embedder
