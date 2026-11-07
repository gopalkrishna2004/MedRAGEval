"""The demo server: ask the application a question in a browser, and read a saved evaluation run.

Retrieval and answering are two calls, so the page can show the sources while
the model is still reading them. The answer goes through `RAG.answer` over the
retrieved chunks, which builds the same prompt as `medrag ask` and shares its
cache. The evaluation view only reads a run's files. Neither the page nor this
module sends the model server's address, which stays in .env.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints

from medrag.app import App
from medrag.generate import ABSTAIN
from medrag.llm import LLMError

log = logging.getLogger("medrag.server")
PAGE = Path(__file__).parent / "web" / "index.html"
PAGE_DPI = 144
READER_CHECK_TIMEOUT = 3

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RetrieveRequest(BaseModel):
    question: Question
    mode: Literal["bm25", "dense", "hybrid"] | None = None
    rerank: bool | None = None


class AnswerRequest(BaseModel):
    question: Question
    chunk_ids: list[str] = Field(min_length=1, max_length=50)


def reader_up(base_url: str) -> bool:
    """Any HTTP reply counts: a server started with --api-key answers 401 here and is still up."""
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=READER_CHECK_TIMEOUT).close()
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False
    return True


def load_run(run_dir: Path) -> dict | None:
    """The report's results and every item, without the candidate pools (unused here) or the server's address."""
    report_path = run_dir / "eval_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    rag = report.get("rag") or {}
    with (run_dir / "responses.jsonl").open() as f:
        items = [{k: v for k, v in json.loads(line).items() if k != "candidates"} for line in f if line.strip()]
    return {"name": run_dir.name, "model": rag.get("model"), "embedding_model": rag.get("embedding_model"),
            "reranker": rag.get("reranker"), "retrieval": rag.get("retrieval"), "judge_model": report.get("judge_model"),
            "overall": report.get("overall"), "answerable": report.get("answerable"), "unanswerable": report.get("unanswerable"),
            "per_type": report.get("per_type"), "warnings": report.get("warnings"), "items": items}


def create_app(app: App, run_dir: str | Path) -> FastAPI:
    api = FastAPI(title="MedRAG demo", docs_url=None, redoc_url=None, openapi_url=None)
    run_dir = Path(run_dir)
    retrieval_lock = threading.Lock()   # the Postgres store shares one connection across request threads
    page_lock = threading.Lock()        # MuPDF is not thread-safe
    pdf = Path(app.cfg["pdf"])

    def page_count() -> int:
        import pymupdf
        with page_lock, pymupdf.open(str(pdf)) as doc:
            return doc.page_count

    @lru_cache(maxsize=64)
    def render(n: int) -> bytes:
        import pymupdf
        with page_lock, pymupdf.open(str(pdf)) as doc:
            return doc[n - 1].get_pixmap(dpi=PAGE_DPI).tobytes("png")

    n_pages = page_count()
    run = load_run(run_dir)

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html")

    @api.get("/api/status")
    def status() -> dict:
        retriever, rc = app.retriever(), app.cfg["retrieval"]
        return {"chunks": len(retriever.chunks), "pages": n_pages, "model": app.cfg["llm"]["model"],
                "reader_up": reader_up(app.cfg["llm"]["base_url"]), "embedding_model": app.cfg["embedding"]["model"],
                "reranker": rc.get("rerank_model") if rc.get("rerank") else None, "top_k": rc["top_k"], "mode": rc["mode"],
                "examples": (app.cfg.get("demo") or {}).get("examples", [])}

    @api.post("/api/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        t0 = time.time()
        retriever = app.retriever()
        with retrieval_lock:
            hits, _ = retriever.retrieve(req.question, mode=req.mode, rerank=req.rerank)
        # Numbered as the model will see them: the context budget can leave a passage out, and citations count only those shown.
        shown = {c.chunk_id: n for n, c in enumerate(app.rag().fit([h.chunk for h in hits]), start=1)}
        return {"latency_s": round(time.time() - t0, 2),
                "hits": [{**h.to_json(), "n": shown.get(h.chunk.chunk_id), "section": h.chunk.heading_path,
                          "kind": h.chunk.kind, "body": h.chunk.body} for h in hits]}

    @api.post("/api/answer")
    def answer(req: AnswerRequest) -> dict:
        by_id = app.retriever().by_id
        missing = [cid for cid in req.chunk_ids if cid not in by_id]
        if missing:
            raise HTTPException(404, f"unknown chunk ids: {', '.join(missing)}")
        try:
            res = app.rag().answer(req.question, tag=f"demo:{req.question[:60]}", passages=[by_id[cid] for cid in req.chunk_ids])
        except LLMError as e:
            log.warning("reader failed: %s", e)
            raise HTTPException(503, "The answering model did not reply. The sources are still valid; try again when its server is back.") from e
        return {"answer": res["answer"], "abstained": res["answer"].startswith(ABSTAIN), "chunk_ids": [c.chunk_id for c in res["passages"]],
                "latency_s": res["latency_s"], "cached": bool(res["usage"].get("cached"))}

    @api.get("/api/page/{n}.png")
    def page(n: int) -> Response:
        if not 1 <= n <= n_pages:
            raise HTTPException(404, f"the PDF has pages 1 to {n_pages}")
        return Response(render(n), media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @api.get("/api/run")
    def get_run() -> dict:
        if run is None:
            raise HTTPException(404, f"no evaluation run at {run_dir}")
        return run

    return api


def serve(app: App, run: str | Path | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Fail fast on a dead model server, and load the models now so the first question is not the slow one."""
    import uvicorn
    demo = app.cfg.get("demo") or {}
    app.cfg["llm"]["timeout"] = demo.get("llm_timeout", app.cfg["llm"].get("timeout"))
    app.cfg["llm"]["max_retries"] = demo.get("llm_max_retries", 0)
    print("loading models...", flush=True)
    app.retriever().retrieve("warm up")
    api = create_app(app, run or demo.get("run", "data/runs/ta1063"))
    print(f"demo ready: http://{host}:{port}", flush=True)
    uvicorn.run(api, host=host, port=port, log_level="warning")
