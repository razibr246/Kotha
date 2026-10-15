"""FastAPI app: chat API, widget page, mock courier."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import Kotha, Turn
from .config import settings
from .courier import mock_router
from .db import close_pool, connection, init_schema
from .embeddings import get_embedder

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema(get_embedder().dim)
    app.state.kotha = Kotha()
    yield
    close_pool()


app = FastAPI(title="Kotha", version="1.0.0", lifespan=lifespan)
app.include_router(mock_router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    shop_id: str | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)


class SourceItem(BaseModel):
    kind: str
    label: str
    ref: str


class Sources(BaseModel):
    count: int
    labels: list[str]
    items: list[SourceItem]


class ChatResponse(BaseModel):
    answer: str
    lang: str
    sources: Sources
    engine: str
    handoff: bool


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    shop_id = req.shop_id or settings.default_shop
    try:
        reply = app.state.kotha.answer(shop_id, req.message.strip(),
                                       [Turn(t.role, t.content) for t in req.history])
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    with connection() as conn:
        conn.execute(
            """INSERT INTO chat_log (shop_id, lang, question, answer, sources, engine)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (shop_id, reply.lang, req.message, reply.text,
             json.dumps(reply.sources, ensure_ascii=False), reply.engine),
        )
        conn.commit()
    return ChatResponse(answer=reply.text, lang=reply.lang, sources=reply.sources,
                        engine=reply.engine, handoff=reply.handoff)


@app.get("/api/health")
def health() -> dict:
    with connection() as conn:
        chunks = conn.execute("SELECT count(*) AS n FROM help_chunks").fetchone()["n"]
    return {"ok": True, "engine": "claude" if settings.anthropic_api_key else "offline",
            "embedder": get_embedder().name, "help_chunks": chunks}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
