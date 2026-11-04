"""PDF -> Docling -> elements -> chunks -> embeddings -> store."""

import json
import time
from pathlib import Path

from medrag.app import collection_name
from medrag.chunk import TokenCounter, build_chunks
from medrag.convert import convert, load_document
from medrag.elements import elements_from_docling
from medrag.pdf import outline_documents, pdf_sha256


def ingest(app, pdf: str | Path | None = None, force: bool = False, embed: bool = True, log=print) -> dict:
    cfg = app.cfg
    pdf = Path(pdf or cfg["pdf"])
    name = collection_name(cfg, pdf)
    existing = app.store.collection(name)
    if existing and not force:
        log(f"collection {name} exists with {existing['n_chunks']} chunks (use --force to rebuild)")
        return {"collection": name, "chunks": existing["n_chunks"], "cached": True}
    t0 = time.time()
    json_path = convert(cfg, pdf, log=log)
    doc = load_document(json_path)
    documents = outline_documents(pdf)
    if documents:
        log(f"  {len(documents)} documents from the PDF outline: " + "; ".join(d["title"][:40] for d in documents[:6]) + (" ..." if len(documents) > 6 else ""))
    else:
        log("  no PDF outline: chunks are titled by their top-level headings")
    elements = elements_from_docling(doc, documents)
    counter = TokenCounter(cfg["chunking"].get("tokenizer"))
    chunks = build_chunks(elements, cfg["chunking"], counter)
    kinds = {k: sum(1 for c in chunks if c.kind == k) for k in ("text", "table")}
    log(f"  {len(elements)} elements -> {len(chunks)} chunks ({kinds['text']} text, {kinds['table']} table; "
        f"{sum(c.part is not None for c in chunks)} pieces of split elements; max {max((c.tokens for c in chunks), default=0)} tokens)")
    dump = Path(cfg["cache_dir"]) / "chunks" / f"{name}.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    vectors = None
    if embed:
        t1 = time.time()
        vectors = app.embedder.embed([c.text for c in chunks])
        log(f"  embedded with {cfg['embedding']['model']} in {time.time() - t1:.0f}s")
    app.store.init_schema()
    app.store.replace_collection(name, {"pdf_path": str(pdf), "pdf_sha256": pdf_sha256(pdf), "chunking": cfg["chunking"],
                                        "embedding_model": cfg["embedding"]["model"] if embed else None}, chunks, vectors)
    log(f"  stored as collection {name} ({time.time() - t0:.0f}s total); chunk dump at {dump}")
    return {"collection": name, "chunks": len(chunks), "elements": len(elements), "kinds": kinds, "documents": len(documents), "cached": False}
