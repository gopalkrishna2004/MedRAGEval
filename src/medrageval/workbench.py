"""What the page's Build tab works with: the documents under data/, the state each has reached, and the jobs that move it on.

A stage (parse, schema draft, extract, generate, ingest, eval) runs as a child process of the same command a user would
type, with its output kept line by line for the page. The command already prints its progress; a crash or a `SystemExit`
in a stage ends the child, not the server; and the run is exactly what the CLI does. One job runs at a time: every stage
either loads the embedding models, uses the one API key, or runs OCR, so two at once only slow each other down.

Nothing here is read from an exit code. A document's state is what its files say: a parse report, a fact sheet, datasets
whose generation report names its parse directory, a collection in the store, runs whose collection name carries the
PDF's hash. A generation that stops short still writes its dataset, and the dataset is listed.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from medrag.app import App, collection_name
from medrag.config import use_pdf
from medrageval.facts.schema import REPO_ROOT, SchemaError, known_roles, parse_schema
from medrageval.generate.generate import TYPES, attempts_per_item
from medrageval.generate.llm import ENV_KEY

SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PDF_MAX = 200 << 20
DATASET_MAX = 20 << 20
STAGES = ("parse", "draft", "extract", "generate", "ingest", "eval")
PAID = {"draft", "extract", "generate", "eval"}
LOG_CAP = 20_000      # lines kept per job; older ones stay in the log file
PY = sys.executable


class JobBusy(RuntimeError):
    """A job is already running."""


class StageError(ValueError):
    """The stage cannot start: an earlier one has not run, or an option is wrong."""


class DatasetError(ValueError):
    """An uploaded dataset has malformed lines."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def slugify(name: str) -> str:
    """A safe directory or file name from what the user called it; a stem that is already safe is kept as is."""
    stem = Path(name).stem if Path(name).suffix.lower() in (".pdf", ".jsonl") else name
    if SLUG.match(stem):
        return stem
    s = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:60].strip("-")
    return s or "document"


def key_set() -> bool:
    load_dotenv()
    return bool(os.environ.get(ENV_KEY, "").strip())


_sha: dict[tuple, str] = {}


def pdf_sha(path: Path) -> str:
    """Cached by (path, mtime, size): the listing hashes every PDF, and a bundle is tens of megabytes."""
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _sha:
        _sha[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _sha[key]


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


# ----- jobs -----

@dataclass
class Job:
    id: str
    stage: str
    slug: str
    argv: list[str]
    log_path: Path
    started: float
    finished: float | None = None
    returncode: int | None = None
    cancelled: bool = False
    interrupted: bool = False          # a record from an earlier server process that never finished
    lines: list[str] = field(default_factory=list)
    dropped: int = 0                   # lines cut from the head of `lines`
    loaded: bool = False               # lines read back from the log file
    result: dict = field(default_factory=dict)
    proc: subprocess.Popen | None = None

    @property
    def state(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.finished is None:
            return "running"
        if self.cancelled:
            return "cancelled"
        return "done" if self.returncode == 0 else "failed"

    @property
    def command(self) -> str:
        names = {"medrageval.cli": "medrageval", "medrag.cli": "medrag"}
        return " ".join(names.get(a, a) for a in self.argv[2:])


class Jobs:
    """One child process at a time; its output is kept in memory for the page and in a log file for later."""

    def __init__(self, root: Path, cwd: Path | None = None):
        self.dir = Path(root) / "jobs"
        self.cwd = Path(cwd or Path.cwd())
        self._lock = threading.Lock()
        self._running: Job | None = None
        self._jobs: dict[str, Job] = {}
        self._callbacks: list = []
        self._load()

    def _load(self) -> None:
        for p in sorted(self.dir.glob("*.json")) if self.dir.exists() else []:
            rec = _read_json(p)
            if not rec or "id" not in rec:
                continue
            job = Job(rec["id"], rec["stage"], rec["slug"], rec.get("argv") or [], self.dir / f"{rec['id']}.log", rec["started"],
                      rec.get("finished"), rec.get("returncode"), rec.get("cancelled", False), result=rec.get("result") or {})
            job.interrupted = job.finished is None
            self._jobs[job.id] = job

    def _write(self, job: Job) -> None:
        rec = {"id": job.id, "stage": job.stage, "slug": job.slug, "argv": job.argv, "started": job.started, "finished": job.finished,
               "returncode": job.returncode, "cancelled": job.cancelled, "result": job.result}
        (self.dir / f"{job.id}.json").write_text(json.dumps(rec, indent=1))

    def on_finish(self, fn) -> None:
        self._callbacks.append(fn)

    def running(self) -> Job | None:
        return self._running

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def start(self, stage: str, slug: str, argv: list[str], result: dict | None = None) -> Job:
        with self._lock:
            if self._running is not None:
                raise JobBusy(f"{self._running.stage} on {self._running.slug} is still running")
            self.dir.mkdir(parents=True, exist_ok=True)
            base = f"{stage}-{slug}-{time.strftime('%Y%m%d-%H%M%S')}"
            job_id, n = base, 1
            while job_id in self._jobs:
                n += 1
                job_id = f"{base}-{n}"
            job = Job(job_id, stage, slug, list(argv), self.dir / f"{job_id}.log", time.time(), result=dict(result or {}))
            job.proc = subprocess.Popen(argv, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
            self._jobs[job_id] = job
            self._running = job
        self._write(job)
        threading.Thread(target=self._pump, args=(job,), daemon=True, name=f"job-{job_id}").start()
        return job

    def _pump(self, job: Job) -> None:
        try:
            with job.log_path.open("w") as log:
                for line in job.proc.stdout:
                    line = line.rstrip("\n")
                    log.write(line + "\n")
                    log.flush()
                    with self._lock:
                        job.lines.append(line)
                        if len(job.lines) > LOG_CAP:
                            cut = len(job.lines) - LOG_CAP
                            del job.lines[:cut]
                            job.dropped += cut
        finally:
            rc = job.proc.wait()
            with self._lock:
                job.returncode, job.finished = rc, time.time()
                self._running = None
            self._write(job)
            for fn in self._callbacks:
                fn(job)

    def tail(self, job_id: str, from_line: int = 0) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with self._lock:
            if not job.lines and not job.loaded and job.proc is None and job.log_path.exists():
                job.lines = job.log_path.read_text().splitlines()[-LOG_CAP:]
                job.loaded = True
            start = max(from_line, job.dropped)
            lines = job.lines[start - job.dropped:]
            return {"id": job.id, "stage": job.stage, "slug": job.slug, "command": job.command, "state": job.state, "returncode": job.returncode,
                    "started": job.started, "finished": job.finished, "lines": lines, "next": job.dropped + len(job.lines),
                    "truncated": from_line < job.dropped, "result": job.result}

    def recent(self, n: int = 50) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)[:n]
        return [{k: v for k, v in self.tail(j.id, 0).items() if k != "lines"} for j in jobs]

    def cancel(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.proc is not None and job.finished is None:
            job.cancelled = True
            job.proc.terminate()
            threading.Timer(5, lambda: job.proc.poll() is None and job.proc.kill()).start()
        return self.tail(job_id, 0)

    def shutdown(self) -> None:
        if self._running is not None:
            self.cancel(self._running.id)


# ----- dataset files -----

def validate_dataset(lines) -> list[str]:
    """The first ten problems with an uploaded dataset, or none. Only what the evaluator reads is required."""
    problems: list[str] = []
    seen: set[str] = set()
    items = 0

    def bad(n: int, msg: str) -> None:
        problems.append(f"line {n}: {msg}")

    for n, line in enumerate(lines, 1):
        if len(problems) >= 10:
            break
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except ValueError:
            bad(n, "not valid JSON")
            continue
        if not isinstance(it, dict):
            bad(n, "not a JSON object")
            continue
        items += 1
        iid = it.get("item_id")
        if not isinstance(iid, str) or not iid:
            bad(n, "item_id must be a non-empty string")
        elif iid in seen:
            bad(n, f"duplicate item_id {iid!r}")
        else:
            seen.add(iid)
        if it.get("type") not in TYPES:
            bad(n, f"type must be one of {', '.join(TYPES)}")
        if not isinstance(it.get("question"), str) or not it["question"].strip():
            bad(n, "question must be a non-empty string")
        answerable = it.get("answerable")
        if not isinstance(answerable, bool):
            bad(n, "answerable must be true or false")
        keys = it.get("answer_keys")
        if keys is not None and not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
            bad(n, "answer_keys must be a list of strings")
        if not isinstance(it.get("evidence", []), list):
            bad(n, "evidence must be a list")
        if answerable is True and not isinstance(it.get("answer"), str):
            bad(n, "an answerable item needs an answer")
        withheld = it.get("withheld")
        if withheld is not None and not (isinstance(withheld, dict) and isinstance(withheld.get("answer_keys"), list) and isinstance(withheld.get("evidence"), list)):
            bad(n, "withheld must hold the withheld answer, answer_keys and evidence")
        messy = it.get("question_messy")
        if messy is not None and not (isinstance(messy, dict) and isinstance(messy.get("text"), str)):
            bad(n, "question_messy must be a mapping with a text field")
    if not items and not problems:
        problems.append("no items")
    return problems[:10]


def estimate_calls(cfg: dict, targets: dict[str, int], messy: bool) -> dict:
    """An upper bound on the paid calls of a generation, before rewrites of rejected items; cache hits are free."""
    gen = judge = twins = 0
    for t, n in targets.items():
        if t not in TYPES or n <= 0:
            continue
        attempts = n * attempts_per_item(cfg, t)
        if t == "unanswerable":
            judge += attempts                      # one screening call per attempt
            continue
        gen += attempts
        if t in cfg.get("judge_types", []):
            judge += attempts
        if t in cfg.get("alone_check_types", []):
            judge += 2 * n
        if messy:
            twins += n
    return {"generator": gen, "judge": judge, "messy": twins, "total": gen + judge + twins}


# ----- documents and their stages -----

class Workbench:
    def __init__(self, root: str | Path, app: App, bench: dict, lock: threading.Lock | None = None, cwd: Path | None = None):
        self.root = Path(root)
        self.app, self.cfg, self.bench = app, app.cfg, bench
        self.roles = known_roles(bench["generate"])
        self.lock = lock or threading.Lock()          # the store shares one connection across threads
        self.page_lock = threading.Lock()              # MuPDF is not thread-safe
        self.jobs = Jobs(self.root, cwd)
        self.jobs.on_finish(self._after_job)
        self.apps: dict[str, App] = {}
        self._pages: dict[tuple, int] = {}

    # paths
    def pdf_path(self, slug: str) -> Path:
        p = self.root / "raw" / f"{slug}.pdf"
        if not SLUG.match(slug) or not p.is_file():
            raise KeyError(slug)
        return p

    def slug_dir(self, slug: str) -> Path:
        return self.root / "interim" / slug

    def dataset_dir(self, name: str) -> Path:
        return self.root / "datasets" / name

    def run_dir(self, name: str) -> Path:
        return self.root / "runs" / name

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(p)

    def _state(self, slug: str) -> dict:
        return _read_json(self.slug_dir(slug) / "workbench.json") or {}

    def _save_state(self, slug: str, **values) -> None:
        d = self.slug_dir(slug)
        d.mkdir(parents=True, exist_ok=True)
        (d / "workbench.json").write_text(json.dumps({**self._state(slug), **values}, indent=1))

    # documents
    def slugs(self) -> list[str]:
        raw = self.root / "raw"
        return [p.stem for p in sorted(raw.glob("*.pdf")) if SLUG.match(p.stem)] if raw.exists() else []

    def documents(self) -> list[dict]:
        return [self.document(s) for s in self.slugs()]

    def pages(self, slug: str) -> int:
        pdf = self.pdf_path(slug)
        key = (str(pdf), pdf.stat().st_mtime_ns)
        if key not in self._pages:
            import pymupdf
            with self.page_lock, pymupdf.open(str(pdf)) as doc:
                self._pages[key] = doc.page_count
        return self._pages[key]

    def document(self, slug: str) -> dict:
        pdf = self.pdf_path(slug)
        interim = self.interim_dir(slug)
        datasets = [d["name"] for d in self.datasets() if d["document"] == slug]
        runs = [r["name"] for r in self.runs() if r["document"] == slug]
        return {"slug": slug, "pdf": self._rel(pdf), "size": pdf.stat().st_size, "pages": self.pages(slug), "sha10": pdf_sha(pdf)[:10],
                "interim": self._rel(interim) if interim else None,
                "stages": {"parsed": self._parsed(interim), "schema": self.schema_record(slug), "facts": self._facts(interim),
                           "datasets": datasets, "ingested": self.ingest_state(slug), "runs": runs}}

    def interim_dir(self, slug: str) -> Path | None:
        """The parse output of this PDF: the directory named after it when its recorded hash matches, else any other that
        matches (TA1063's parse is in data/interim/ta1063 while its PDF is source_document.pdf), else none."""
        sha = pdf_sha(self.pdf_path(slug))
        own = self.slug_dir(slug)
        if self._meta_sha(own) == sha:
            return own
        root = self.root / "interim"
        for d in sorted(root.iterdir()) if root.exists() else []:
            if d.is_dir() and self._meta_sha(d) == sha:
                return d
        return None

    @staticmethod
    def _meta_sha(d: Path) -> str | None:
        meta = _read_json(d / "pages.meta.json")
        return meta.get("sha256") if meta else None

    def _owners(self) -> dict[Path, str]:
        """Resolved parse directory -> document slug, for attributing datasets by their generation report."""
        out = {}
        for s in self.slugs():
            d = self.interim_dir(s)
            if d:
                out[d.resolve()] = s
        return out

    @staticmethod
    def _parsed(interim: Path | None) -> dict | None:
        r = _read_json(interim / "parse_report.json") if interim else None
        if not r:
            return None
        return {"dir": str(interim), "pages": r.get("pages"), "documents": r.get("documents"), "sections": r.get("sections"),
                "elements": r.get("elements"), "warnings": len(r.get("warnings") or [])}

    @staticmethod
    def _facts(interim: Path | None) -> dict | None:
        r = _read_json(interim / "factsheet_report.json") if interim else None
        if not r or not (interim / "factsheet.jsonl").exists():
            return None
        llm = r.get("llm") or {}
        return {"facts": r.get("facts"), "batches": r.get("batches", sum((g.get("batches") or 0) for g in (r.get("per_group") or {}).values())),
                "model": r.get("model"), "calls": llm.get("calls"), "generated_at": _mtime(interim / "factsheet.jsonl")}

    def ingest_state(self, slug: str):
        """{collection, chunks} when the store holds the PDF, None when not, "unknown" when the store cannot be reached."""
        name = collection_name(self.cfg, self.pdf_path(slug))
        try:
            with self.lock:
                c = self.app.store.collection(name)
        except Exception:     # psycopg errors, a missing driver, a config without a store: all mean "cannot tell"
            return "unknown"
        return {"collection": name, "chunks": c["n_chunks"]} if c else None

    def app_for(self, slug: str) -> App:
        """The application over this PDF, sharing the loaded models and the store connection of the one the server started with."""
        pdf = self.pdf_path(slug)
        if slug not in self.apps:
            if Path(self.cfg["pdf"]).resolve() == pdf.resolve():
                self.apps[slug] = self.app
            else:
                a = App(use_pdf(self.cfg, pdf), self.app.use_cache)
                a._store, a._embedder, a._reranker, a._llm = self.app._store, self.app._embedder, self.app._reranker, self.app._llm
                self.apps[slug] = a
        return self.apps[slug]

    def drop_app(self, slug: str) -> None:
        """After an ingest: a retriever built on the old collection would serve stale chunk ids."""
        a = self.apps.pop(slug, None)
        if a is not None:
            a._retriever = None
        if slug in self.slugs() and Path(self.cfg["pdf"]).resolve() == self.pdf_path(slug).resolve():
            self.app._retriever = None

    def _after_job(self, job: Job) -> None:
        if job.stage == "ingest":
            self.drop_app(job.slug)

    def save_pdf(self, name: str, body: bytes, replace: bool = False) -> str:
        if not body.startswith(b"%PDF-"):
            raise ValueError("not a PDF file")
        slug = slugify(name)
        path = self.root / "raw" / f"{slug}.pdf"
        if path.exists() and not replace:
            raise FileExistsError(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        self.drop_app(slug)
        return slug

    # schemas
    def schema_info(self, path: str | Path) -> dict:
        p = Path(path) if Path(path).is_absolute() or Path(path).exists() else REPO_ROOT / path
        rec = {"path": self._rel(p), "source": "config" if (REPO_ROOT / "configs" / "schema") in p.resolve().parents else "draft",
               "name": None, "groups": 0, "slots": 0, "links": 0, "valid": False, "error": None}
        try:
            s = parse_schema(yaml.safe_load(p.read_text()), self.roles, p)
            rec.update(name=s.name, groups=len(s.groups), slots=len(s.slots), links=len(s.links), valid=True)
        except (SchemaError, yaml.YAMLError, OSError) as e:
            rec["error"] = str(e)
        return rec

    def schemas(self) -> list[dict]:
        paths = sorted((REPO_ROOT / "configs" / "schema").glob("*.yaml")) + sorted((self.root / "interim").glob("*/schema.yaml"))
        return [self.schema_info(p) for p in paths]

    def schema_record(self, slug: str) -> dict | None:
        path = self._state(slug).get("schema")
        if not path and (self.slug_dir(slug) / "schema.yaml").exists():
            path = self._rel(self.slug_dir(slug) / "schema.yaml")
        return self.schema_info(path) if path else None

    def read_schema(self, slug: str) -> dict | None:
        rec = self.schema_record(slug)
        if rec is None:
            return None
        p = Path(rec["path"]) if Path(rec["path"]).exists() else REPO_ROOT / rec["path"]
        return {**rec, "text": p.read_text() if p.exists() else ""}

    def check_schema(self, text: str) -> dict:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise SchemaError(f"not valid YAML: {e}") from e
        s = parse_schema(data, self.roles)
        return {"name": s.name, "groups": len(s.groups), "slots": len(s.slots), "links": len(s.links)}

    def save_schema(self, slug: str, text: str) -> dict:
        """Always a copy under the document's directory: the reviewed schemas under configs/ are never edited from the page."""
        self.pdf_path(slug)
        self.check_schema(text)
        d = self.slug_dir(slug)
        d.mkdir(parents=True, exist_ok=True)
        (d / "schema.yaml").write_text(text)
        self._save_state(slug, schema=self._rel(d / "schema.yaml"))
        return self.schema_record(slug)

    def select_schema(self, slug: str, path: str) -> dict:
        self.pdf_path(slug)
        if path not in {s["path"] for s in self.schemas()}:
            raise KeyError(path)
        self._save_state(slug, schema=path)
        return self.schema_record(slug)

    # datasets and runs
    def datasets(self) -> list[dict]:
        root = self.root / "datasets"
        owners = self._owners()
        out = []
        for d in sorted(root.iterdir()) if root.exists() else []:
            f = d / "dataset.jsonl"
            if not f.is_file() or not SLUG.match(d.name):
                continue
            per_type: Counter = Counter()
            messy = False
            for line in f.open():
                if line.strip():
                    try:
                        it = json.loads(line)
                    except ValueError:
                        continue
                    per_type[it.get("type")] += 1
                    messy = messy or bool(it.get("question_messy"))
            report = _read_json(d / "generate_report.json")
            state = _read_json(d / "workbench.json") or {}
            document = owners.get(Path(report["interim_dir"]).resolve()) if report and report.get("interim_dir") else state.get("document")
            out.append({"name": d.name, "items": sum(per_type.values()), "per_type": dict(per_type), "messy": messy, "document": document,
                        "source": "generated" if report else "uploaded", "model": report.get("model") if report else None,
                        "short": bool(report and any("accepted" in w or "short" in w for w in report.get("warnings") or [])),
                        "modified": _mtime(f)})
        return out

    def runs(self) -> list[dict]:
        root = self.root / "runs"
        prefixes = {s: f"{s}-{pdf_sha(self.pdf_path(s))[:10]}-" for s in self.slugs()}
        out = []
        for p in sorted(root.glob("*/eval_report.json")) + sorted(root.glob("*/*/eval_report.json")) if root.exists() else []:
            r = _read_json(p)
            if not r:
                continue
            rag = r.get("rag") or {}
            coll = rag.get("collection") or ""
            out.append({"name": p.parent.relative_to(root).as_posix(), "document": next((s for s, pre in prefixes.items() if coll.startswith(pre)), None),
                        "items": r.get("items"), "score": (r.get("overall") or {}).get("score"), "question_setting": r.get("question_setting"),
                        "datasets": r.get("datasets"), "model": rag.get("model"), "judge_model": r.get("judge_model"), "modified": _mtime(p)})
        return sorted(out, key=lambda x: x["modified"] or "", reverse=True)

    def save_dataset(self, name: str, body: bytes, document: str | None = None, replace: bool = False) -> dict:
        if not SLUG.match(name):
            raise ValueError("the name may hold letters, digits, dots, dashes and underscores")
        if document:
            self.pdf_path(document)
        d = self.dataset_dir(name)
        if (d / "dataset.jsonl").exists() and not replace:
            raise FileExistsError(name)
        document = document or (_read_json(d / "workbench.json") or {}).get("document")   # a replacement keeps its attribution
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            raise DatasetError([f"not UTF-8 text: {e}"]) from e
        problems = validate_dataset(text.splitlines())
        if problems:
            raise DatasetError(problems)
        d.mkdir(parents=True, exist_ok=True)
        (d / "dataset.jsonl").write_text("\n".join(l for l in text.splitlines() if l.strip()) + "\n")
        (d / "workbench.json").write_text(json.dumps({"document": document, "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        return next(x for x in self.datasets() if x["name"] == name)

    # jobs
    def check_stage(self, slug: str, stage: str, opts: dict) -> None:
        if stage not in STAGES:
            raise StageError(f"unknown stage {stage!r}")
        doc = self.document(slug)
        st = doc["stages"]
        dry = bool(opts.get("dry_run"))
        paid = stage == "draft" or (stage in ("extract", "generate") and not dry) or (stage == "eval" and opts.get("judge", True))
        if paid and not key_set():
            raise StageError(f"{ENV_KEY} is not set: this step makes paid API calls")
        if stage in ("draft", "extract", "generate") and not st["parsed"]:
            raise StageError("parse the document first")
        if stage == "extract" and not (st["schema"] and st["schema"]["valid"]):
            raise StageError("choose or draft a valid schema first")
        if stage == "generate":
            if not st["facts"]:
                raise StageError("extract the fact sheet first")
            targets = opts.get("targets") or {}
            if not targets:
                raise StageError("choose at least one question type")
            sources = self.bench["generate"]["unanswerable"]["source_types"]
            if "unanswerable" in targets and not any(t in targets for t in sources):
                raise StageError("unanswerable items are built from accepted items of other types: add factual, numerical, comparative, multihop or attribution")
            if not dry and (self.dataset_dir(opts["name"]) / "dataset.jsonl").exists() and not opts.get("replace"):
                raise FileExistsError(f"a dataset named {opts['name']} exists")
        if stage == "eval":
            if not (self.dataset_dir(opts["dataset"]) / "dataset.jsonl").is_file():
                raise StageError(f"no dataset named {opts['dataset']}")
            if st["ingested"] is None:
                raise StageError("ingest the document first")
            if (self.run_dir(opts["name"]) / "eval_report.json").exists() and not opts.get("replace"):
                raise FileExistsError(f"a run named {opts['name']} exists")

    def argv_for(self, stage: str, slug: str, opts: dict) -> tuple[list[str], dict]:
        """The command for a stage, and what the page needs from it when it is done. Paths come from the checked slug only."""
        pdf = str(self.pdf_path(slug))
        own = self.slug_dir(slug)
        interim = str(self.interim_dir(slug) or own)
        rag_cfg = self.cfg.get("_path")
        rag_opt = [] if not rag_cfg or Path(rag_cfg) == Path("configs/rag.yaml") else [rag_cfg]
        ev = [PY, "-m", "medrageval.cli"]
        if stage == "parse":
            return ev + ["parse", pdf, "--out", str(own)] + (["--force"] if opts.get("force") else []), {}
        if stage == "draft":
            out = str(own / "schema.yaml")
            return ev + ["schema", "draft", interim, "--out", out], {"schema": self._rel(own / "schema.yaml")}
        if stage == "extract":
            schema = (self.schema_record(slug) or {}).get("path") or self.bench["facts"]["schema"]
            return ev + ["extract", interim, "--schema", schema] + (["--dry-run"] if opts.get("dry_run") else []), {}
        if stage == "generate":
            targets = {t: int(opts["targets"][t]) for t in TYPES if t in opts["targets"]}
            argv = ev + ["generate", interim, "--out", str(self.dataset_dir(opts["name"])), "--types", ",".join(targets)]
            for t, n in targets.items():
                argv += ["--target", f"{t}={n}"]
            if opts.get("seed") is not None:
                argv += ["--seed", str(opts["seed"])]
            if not opts.get("messy", True):
                argv.append("--no-messy")
            if opts.get("dry_run"):
                argv.append("--dry-run")
            return argv, ({} if opts.get("dry_run") else {"dataset": opts["name"], "targets": targets})
        if stage == "ingest":
            return [PY, "-m", "medrag.cli"] + (["--config"] + rag_opt if rag_opt else []) + ["ingest", pdf] + (["--force"] if opts.get("force") else []), {}
        if stage == "eval":
            argv = ev + ["eval", str(self.dataset_dir(opts["dataset"])), "--pdf", pdf, "--out", str(self.run_dir(opts["name"])),
                         "--question", opts.get("question", "clean")]
            if rag_opt:
                argv += ["--rag-config"] + rag_opt
            if opts.get("mode"):
                argv += ["--mode", opts["mode"]]
            if not opts.get("rerank", True):
                argv.append("--no-rerank")
            if opts.get("top_k"):
                argv += ["--top-k", str(opts["top_k"])]
            if opts.get("limit"):
                argv += ["--limit", str(opts["limit"])]
            if opts.get("types"):
                argv += ["--types", ",".join(t for t in TYPES if t in opts["types"])]
            if not opts.get("judge", True):
                argv.append("--no-judge")
            return argv, {"run": opts["name"]}
        raise StageError(f"unknown stage {stage!r}")

    def progress(self, job: Job) -> dict:
        """What the page can show while a job runs: accepted items per type for a generation, elapsed time for the rest."""
        elapsed = round((job.finished or time.time()) - job.started, 1)
        if job.stage == "generate" and job.result.get("dataset"):
            f = self.dataset_dir(job.result["dataset"]) / "dataset.jsonl"
            counts: Counter = Counter()
            if f.exists():
                for line in f.open():
                    if line.strip():
                        try:
                            counts[json.loads(line).get("type")] += 1
                        except ValueError:
                            pass
            return {"elapsed_s": elapsed, "accepted": dict(counts), "targets": job.result.get("targets") or {}}
        return {"elapsed_s": elapsed}
