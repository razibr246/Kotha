# Kotha

Support agent that answers order questions in Bangla and English, grounded in a shop's own help docs and live order data. Every answer says where it came from.

```
Customer:  Where is order #4821?
Kotha:     Shipped this morning. It reaches Mirpur by 5 pm today.
           2 sources · order DB, courier API

Customer:  অর্ডার #৪৮২১ কোথায়?
Kotha:     আজ সকালে শিপ হয়েছে। আজ বিকেল ৫টার মধ্যে মিরপুর পৌঁছে যাবে।
```

## How it works

```
message ─► detect language (Bangla script / Banglish / English), normalise ০-৯ digits
        ─► agent ──► get_order(order_no)        Postgres: orders (scoped to the shop)
                 ├─► track_shipment(id)         courier API (mock built in, or yours)
                 └─► search_help(query)         pgvector + BM25 hybrid over help docs
        ─► answer in the customer's language + list of sources used
```

- **Two engines, same tools.** With `ANTHROPIC_API_KEY` set, Claude runs a tool-use loop under strict grounding rules (facts only from tool results, no invented dates or policies, offer a human otherwise). Without a key, or if the API call fails, a deterministic offline agent answers from the same tools, so support never goes dark.
- **Times are pre-phrased.** Tool results carry ready-made local-time phrases (`5 pm today`, `আজ বিকেল ৫টা`) in Asia/Dhaka, so the model never does timezone math.
- **Privacy.** Order lookups never return the customer's full name, phone or address; only the delivery area, first name and a masked phone (`•••• 807`).
- **Bilingual docs.** Help docs live as `data/help/<doc>.<lang>.md`. Sections are parallel across languages, so a match in English is swapped for the same section in Bangla when the customer writes Bangla (and the reverse).
- **Retrieval.** pgvector (HNSW, cosine) blended with BM25. The default embedder is a local hashing embedder (no downloads, handles Bengali script). Set `KOTHA_EMBEDDER=voyage` for a multilingual model, which also gives true cross-lingual matching.

## Run it

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY to use Claude
docker compose up --build       # Postgres+pgvector, seeds a demo shop, serves on :8000
```

Open http://localhost:8000 and try `Where is order #4821?`, `অর্ডার #৪৮১০ কবে পাব?`, `How long do refunds take?`.

Without Docker (Postgres 16 with the `vector` extension):

```bash
pip install -r requirements.txt
export DATABASE_URL=postgresql://kotha:kotha@localhost:5432/kotha
python -m kotha.seed --reset
uvicorn kotha.main:app --reload
```

Demo orders (seeded relative to the current time, so they always read "live"):

| Order | State |
|---|---|
| 4821 | Shipped this morning, Mirpur, ETA 5 pm |
| 4810 | Out for delivery, cash on delivery |
| 4807 | Delivery attempt failed, customer unreachable |
| 4815 | Delivered two days ago |
| 4822 | Pending |
| 4830 | Packed |
| 4799 | Cancelled, bKash refund |

## API

`POST /api/chat`

```json
{ "message": "Where is order #4821?", "shop_id": "demo",
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }] }
```

```json
{ "answer": "Shipped this morning. It reaches Mirpur by 5 pm today.",
  "lang": "en", "engine": "claude", "handoff": false,
  "sources": { "count": 2, "labels": ["order DB", "courier API"],
               "items": [{ "kind": "order_db", "label": "order DB", "ref": "#4821" },
                         { "kind": "courier", "label": "courier API", "ref": "PTH-88214821" }] } }
```

Also: `GET /api/health`, `GET /mock-courier/track/{tracking_id}`. Every chat is written to `chat_log` with its sources and engine.

## Plugging in a real shop

- **Orders:** point `orders.get_order` at the shop's database or API; keep the returned shape.
- **Courier:** set `COURIER_API_URL` to any service with `GET /track/{id}` returning the normalised shape in `courier.py`. A Pathao / Steadfast / RedX adapter is a small mapping layer in front of their APIs.
- **Docs:** `python -m kotha.seed --docs-only --shop <id> --docs path/to/help`.

## Tests

```bash
TEST_DATABASE_URL=postgresql://kotha:kotha@localhost:5432/kotha_test pytest
```

44 tests against real Postgres + pgvector with a pinned clock: language and digit handling, time phrases, retrieval quality and off-topic rejection, every order state in both languages, shop isolation, PII masking, courier outages, the Claude tool loop (scripted client) and its fallback, and the HTTP API.

## Layout

```
kotha/
  main.py        FastAPI app, /api/chat
  agent.py       ClaudeAgent, OfflineAgent, Kotha (picks one, falls back)
  tools.py       get_order / track_shipment / search_help + source tracking
  helpdocs.py    chunking, ingest, hybrid search, language twins
  orders.py      order lookup with PII masking
  courier.py     courier clients + mock courier API
  embeddings.py  hash and Voyage embedders
  lang.py        language detection, Bangla digits, order numbers, area names
  timefmt.py     "5 pm today" / "আজ বিকেল ৫টা"
  seed.py        demo shop
  static/        chat UI
data/help/       help docs in English and Bangla
tests/
```
