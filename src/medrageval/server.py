"""The browser page: ask a document a question, build a dataset from a PDF and evaluate the application on it, read a run.

Retrieval and answering are two calls, so the page can show the sources while the model is still reading them. The
answer goes through `RAG.answer` over the retrieved chunks, which builds the same prompt as `medrag ask` and shares its
cache. The Build tab starts the pipeline's stages as child processes (`workbench`) and streams their output; the
evaluation view only reads a run's files. Neither the page nor this module sends the model server's address, which
stays in .env. The page can start paid API calls, so the server listens on 127.0.0.1 unless told otherwise.
"""

import logging
import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints, ValidationError, conint

from medrag.app import App
from medrag.generate import ABSTAIN
from medrag.llm import LLMError
from medrag.retrieve import MODES
from medrageval.config import load_config as load_bench_config
from medrageval.facts.schema import SchemaError
from medrageval.generate.generate import TYPES
from medrageval.workbench import DATASET_MAX, PDF_MAX, SLUG, STAGES, DatasetError, JobBusy, StageError, Workbench, estimate_calls, key_set

log = logging.getLogger("medrageval.server")
PAGE = Path(__file__).parent / "web" / "index.html"
PAGE_DPI = 144
READER_CHECK_TIMEOUT = 3

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Slug = Annotated[str, StringConstraints(pattern=SLUG.pattern)]
TypeName = Literal[tuple(TYPES)]


class RetrieveRequest(BaseModel):
    question: Question
    mode: Literal["bm25", "dense", "hybrid"] | None = None
    rerank: bool | None = None
    document: Slug | None = None


class AnswerRequest(BaseModel):
    question: Question
    chunk_ids: list[str] = Field(min_length=1, max_length=50)
    document: Slug | None = None


class SchemaText(BaseModel):
    text: str = Field(max_length=500_000)


class SchemaChoice(BaseModel):
    path: str = Field(max_length=500)


class EstimateRequest(BaseModel):
    targets: dict[TypeName, conint(ge=1, le=200)]
    messy: bool = True


class JobRequest(BaseModel):
    stage: Literal[STAGES]
    options: dict = Field(default_factory=dict)


class ParseOptions(BaseModel):
    force: bool = False


class DraftOptions(BaseModel):
    pass


class ExtractOptions(BaseModel):
    dry_run: bool = False


class GenerateOptions(BaseModel):
    name: Slug
    targets: dict[TypeName, conint(ge=1, le=200)]
    messy: bool = True
    seed: int | None = None
    dry_run: bool = False
    replace: bool = False


class IngestOptions(BaseModel):
    force: bool = False


class EvalOptions(BaseModel):
    dataset: Slug
    name: Slug
    question: Literal["clean", "messy"] = "clean"
    mode: Literal[MODES] | None = None
    rerank: bool = True
    top_k: conint(ge=1, le=50) | None = None
    limit: conint(ge=1) | None = None
    types: list[TypeName] = Field(default_factory=list)
    judge: bool = True
    replace: bool = False


OPTIONS = {"parse": ParseOptions, "draft": DraftOptions, "extract": ExtractOptions, "generate": GenerateOptions, "ingest": IngestOptions, "eval": EvalOptions}


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
            "question_setting": report.get("question_setting"), "datasets": report.get("datasets"),
            "overall": report.get("overall"), "answerable": report.get("answerable"), "unanswerable": report.get("unanswerable"),
            "per_type": report.get("per_type"), "warnings": report.get("warnings"), "items": items}


def create_app(app: App, run_dir: str | Path | None = None, root: str | Path = "data", bench_config: str | Path | None = None) -> FastAPI:
    wb = Workbench(root, app, load_bench_config(bench_config), lock=threading.Lock())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        wb.jobs.shutdown()

    api = FastAPI(title="MedRAGEval", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    api.state.workbench = wb
    primary = Path(app.cfg["pdf"]).stem
    try:
        initial_run = Path(run_dir).resolve().relative_to((wb.root / "runs").resolve()).as_posix() if run_dir else None
    except ValueError:
        initial_run = None
    if initial_run and not (wb.run_dir(initial_run) / "eval_report.json").exists():
        initial_run = None

    def document(slug: str) -> dict:
        try:
            return wb.document(slug)
        except KeyError:
            raise HTTPException(404, f"no document named {slug}") from None

    def ingested_app(slug: str) -> App:
        document(slug)
        if not isinstance(wb.ingest_state(slug), dict):
            raise HTTPException(409, f"{slug} is not ingested yet: the Build tab's ingest step stores its chunks")
        return wb.app_for(slug)

    @lru_cache(maxsize=64)
    def render(slug: str, n: int) -> bytes:
        import pymupdf
        with wb.page_lock, pymupdf.open(str(wb.pdf_path(slug))) as doc:
            return doc[n - 1].get_pixmap(dpi=PAGE_DPI).tobytes("png")

    async def read_body(request: Request, cap: int) -> bytes:
        parts, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > cap:
                raise HTTPException(413, f"over the {cap >> 20} MB limit")
            parts.append(chunk)
        return b"".join(parts)

    def tail(job_id: str, from_line: int = 0) -> dict:
        t = wb.jobs.tail(job_id, from_line)
        if t is None:
            raise HTTPException(404, f"no job {job_id}")
        return {**t, "progress": wb.progress(wb.jobs.get(job_id))}

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html")

    @api.get("/api/status")
    def status(document_: Annotated[Slug | None, Query(alias="document")] = None) -> dict:
        slug = document_ or primary
        d = document(slug)
        ing = d["stages"]["ingested"]
        rc = app.cfg["retrieval"]
        chunks = len(wb.app_for(slug).retriever().chunks) if isinstance(ing, dict) else 0
        running = wb.jobs.running()
        return {"document": slug, "run": initial_run, "chunks": chunks, "pages": d["pages"], "ingested": ing if isinstance(ing, dict) else None,
                "postgres_up": ing != "unknown", "key_set": key_set(), "job": running.id if running else None,
                "model": app.cfg["llm"]["model"], "reader_up": reader_up(app.cfg["llm"]["base_url"]), "embedding_model": app.cfg["embedding"]["model"],
                "reranker": rc.get("rerank_model") if rc.get("rerank") else None, "top_k": rc["top_k"], "mode": rc["mode"],
                "examples": (app.cfg.get("demo") or {}).get("examples", []) if slug == primary else []}

    @api.post("/api/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        t0 = time.time()
        a = ingested_app(req.document or primary)
        retriever = a.retriever()
        with wb.lock:
            hits, _ = retriever.retrieve(req.question, mode=req.mode, rerank=req.rerank)
        # Numbered as the model will see them: the context budget can leave a passage out, and citations count only those shown.
        shown = {c.chunk_id: n for n, c in enumerate(a.rag().fit([h.chunk for h in hits]), start=1)}
        return {"latency_s": round(time.time() - t0, 2),
                "hits": [{**h.to_json(), "n": shown.get(h.chunk.chunk_id), "section": h.chunk.heading_path,
                          "kind": h.chunk.kind, "body": h.chunk.body} for h in hits]}

    @api.post("/api/answer")
    def answer(req: AnswerRequest) -> dict:
        a = ingested_app(req.document or primary)
        by_id = a.retriever().by_id
        missing = [cid for cid in req.chunk_ids if cid not in by_id]
        if missing:
            raise HTTPException(404, f"unknown chunk ids: {', '.join(missing)}")
        try:
            res = a.rag().answer(req.question, tag=f"demo:{req.question[:60]}", passages=[by_id[cid] for cid in req.chunk_ids])
        except LLMError as e:
            log.warning("reader failed: %s", e)
            raise HTTPException(503, "The answering model did not reply. The sources are still valid; try again when its server is back.") from e
        return {"answer": res["answer"], "abstained": res["answer"].startswith(ABSTAIN), "chunk_ids": [c.chunk_id for c in res["passages"]],
                "latency_s": res["latency_s"], "cached": bool(res["usage"].get("cached"))}

    @api.get("/api/documents/{slug}/page/{n}.png")
    def page(slug: Slug, n: int) -> Response:
        pages = document(slug)["pages"]
        if not 1 <= n <= pages:
            raise HTTPException(404, f"the PDF has pages 1 to {pages}")
        return Response(render(slug, n), media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    # ----- build -----

    @api.get("/api/documents")
    def documents() -> list[dict]:
        return wb.documents()

    @api.get("/api/documents/{slug}")
    def get_document(slug: Slug) -> dict:
        return document(slug)

    @api.post("/api/documents", status_code=201)
    async def upload_pdf(request: Request, name: Annotated[str, Query(min_length=1, max_length=200)], replace: bool = False) -> dict:
        body = await read_body(request, PDF_MAX)
        if not body.startswith(b"%PDF-"):
            raise HTTPException(415, "not a PDF file")
        try:
            slug = wb.save_pdf(name, body, replace)
        except FileExistsError as e:
            raise HTTPException(409, f"a document named {e} exists; tick Replace to overwrite it") from None
        render.cache_clear()
        return document(slug)

    @api.get("/api/schemas")
    def schemas() -> list[dict]:
        return wb.schemas()

    @api.get("/api/documents/{slug}/schema")
    def get_schema(slug: Slug) -> dict:
        document(slug)
        rec = wb.read_schema(slug)
        if rec is None:
            raise HTTPException(404, "no schema chosen for this document yet")
        return rec

    @api.post("/api/documents/{slug}/schema/check")
    def check_schema(slug: Slug, req: SchemaText) -> dict:
        document(slug)
        try:
            return {"ok": True, **wb.check_schema(req.text)}
        except SchemaError as e:
            raise HTTPException(422, str(e)) from None

    @api.put("/api/documents/{slug}/schema")
    def put_schema(slug: Slug, req: SchemaText) -> dict:
        document(slug)
        try:
            return wb.save_schema(slug, req.text)
        except SchemaError as e:
            raise HTTPException(422, str(e)) from None

    @api.post("/api/documents/{slug}/schema/select")
    def select_schema(slug: Slug, req: SchemaChoice) -> dict:
        document(slug)
        try:
            return wb.select_schema(slug, req.path)
        except KeyError:
            raise HTTPException(404, f"no schema at {req.path}") from None

    @api.get("/api/build/config")
    def build_config() -> dict:
        g = wb.bench["generate"]
        return {"types": list(TYPES), "targets": g["targets"], "attempts_per_item": g["attempts_per_item"], "messy_enabled": g["messy"]["enabled"],
                "judge_types": g["judge_types"], "alone_check_types": g["alone_check_types"], "unanswerable_sources": g["unanswerable"]["source_types"],
                "min_share_of_target": g["min_share_of_target"],
                "models": {"generator": g["model"], "judge": g.get("judge_model") or g["model"], "facts": wb.bench["facts"].get("model") or g["model"],
                           "eval_judge": wb.bench["rag"]["judge"].get("model") or g["model"], "reader": app.cfg["llm"]["model"]}}

    @api.post("/api/build/estimate")
    def estimate(req: EstimateRequest) -> dict:
        return estimate_calls(wb.bench["generate"], req.targets, req.messy)

    @api.post("/api/documents/{slug}/jobs", status_code=202)
    def start_job(slug: Slug, req: JobRequest) -> dict:
        document(slug)
        try:
            opts = OPTIONS[req.stage].model_validate(req.options).model_dump()
        except ValidationError as e:
            raise HTTPException(422, "; ".join(f"{'.'.join(str(x) for x in err['loc']) or 'options'}: {err['msg']}" for err in e.errors())) from None
        try:
            wb.check_stage(slug, req.stage, opts)
        except StageError as e:
            raise HTTPException(422, str(e)) from None
        except FileExistsError as e:
            raise HTTPException(409, f"{e}; tick Replace to overwrite it") from None
        argv, result = wb.argv_for(req.stage, slug, opts)
        try:
            job = wb.jobs.start(req.stage, slug, argv, result)
        except JobBusy as e:
            raise HTTPException(409, str(e)) from None
        return tail(job.id)

    @api.get("/api/jobs")
    def jobs() -> list[dict]:
        return wb.jobs.recent()

    @api.get("/api/jobs/{job_id}")
    def get_job(job_id: str, from_: Annotated[int, Query(alias="from", ge=0)] = 0) -> dict:
        return tail(job_id, from_)

    @api.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if wb.jobs.cancel(job_id) is None:
            raise HTTPException(404, f"no job {job_id}")
        return tail(job_id)

    @api.get("/api/datasets")
    def datasets() -> list[dict]:
        return wb.datasets()

    @api.post("/api/datasets", status_code=201)
    async def upload_dataset(request: Request, name: Slug, document_: Annotated[Slug | None, Query(alias="document")] = None, replace: bool = False) -> dict:
        body = await read_body(request, DATASET_MAX)
        try:
            return wb.save_dataset(name, body, document_, replace)
        except DatasetError as e:
            raise HTTPException(422, e.problems) from None
        except FileExistsError as e:
            raise HTTPException(409, f"a dataset named {e} exists; tick Replace to overwrite it") from None
        except KeyError as e:
            raise HTTPException(404, f"no document named {e.args[0]}") from None

    @api.get("/api/runs")
    def runs() -> list[dict]:
        return wb.runs()

    @api.get("/api/runs/{name:path}")
    def get_run(name: str) -> dict:
        rec = next((r for r in wb.runs() if r["name"] == name), None)
        run = load_run(wb.run_dir(name)) if rec else None
        if run is None:
            raise HTTPException(404, f"no evaluation run named {name}")
        return {**run, "name": name, "document": rec["document"]}

    return api


def serve(app: App, run: str | Path | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load the models now so the first question is not the slow one, and warm up when the document is ingested."""
    import uvicorn
    demo = app.cfg.get("demo") or {}
    app.cfg["llm"]["timeout"] = demo.get("llm_timeout", app.cfg["llm"].get("timeout"))
    app.cfg["llm"]["max_retries"] = demo.get("llm_max_retries", 0)
    print("loading models...", flush=True)
    app.embedder
    app.reranker
    try:
        app.retriever().retrieve("warm up")
    except RuntimeError as e:
        print(f"{app.cfg['pdf']}: {e}; the Build tab can ingest it", flush=True)
    api = create_app(app, run or demo.get("run"))
    print(f"ready: http://{host}:{port}", flush=True)
    uvicorn.run(api, host=host, port=port, log_level="warning")
