import argparse
import sys

from medrag.app import App
from medrag.retrieve import MODES


def main() -> None:
    ap = argparse.ArgumentParser(prog="medrag", description="RAG application over a PDF bundle")
    ap.add_argument("--config", help="default configs/rag.yaml")
    sub = ap.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database administration")
    db.add_argument("action", choices=["init", "list"])

    ing = sub.add_parser("ingest", help="parse, chunk, embed and store a PDF")
    ing.add_argument("pdf", nargs="?", help="default: the config's pdf")
    ing.add_argument("--force", action="store_true", help="rebuild the collection even if it exists")
    ing.add_argument("--no-embed", action="store_true", help="store chunks without vectors (BM25 only)")

    se = sub.add_parser("search", help="show the chunks retrieved for a question")
    se.add_argument("question")
    se.add_argument("--mode", choices=MODES)
    se.add_argument("--k", type=int)
    se.add_argument("--no-rerank", action="store_true")

    ask = sub.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES)
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--model", help="override llm.model")
    ask.add_argument("--base-url", help="override llm.base_url")
    ask.add_argument("--no-cache", action="store_true")

    srv = sub.add_parser("serve", help="the demo: ask questions and read an evaluation run in a browser")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--run", help="evaluation run directory; default: the config's demo.run")

    a = ap.parse_args()
    app = App(a.config, use_cache=not getattr(a, "no_cache", False))
    if getattr(a, "model", None):
        app.cfg["llm"]["model"] = a.model
    if getattr(a, "base_url", None):
        app.cfg["llm"]["base_url"] = a.base_url

    if a.command == "db":
        if a.action == "init":
            app.store.init_schema()
            print("schema ready")
        else:
            for c in app.store.collections():
                print(f"{c['name']}: {c['n_chunks']} chunks, {c.get('embedding_model')}, {c.get('created_at')}")
    elif a.command == "ingest":
        from medrag.ingest import ingest
        ingest(app, a.pdf, a.force, not a.no_embed)
    elif a.command == "search":
        r = app.retriever(a.mode, False if a.no_rerank else None)
        for h in r.search(a.question, a.mode, a.k, False if a.no_rerank else None):
            head, _, body = h.chunk.text.partition("\n")
            print(f"[{h.rank}] {h.chunk.chunk_id} {h.ranks} {head}\n     {' '.join(body.split())[:200]}")
    elif a.command == "ask":
        rag = app.rag(a.mode, False if a.no_rerank else None)
        res = rag.answer(a.question, mode=a.mode)
        print(res["answer"])
        print(f"\n{len(res['passages'])} passages ({res['context_chars']} chars, {res['latency_s']}s, {rag.llm.model}):")
        for i, c in enumerate(res["passages"], start=1):
            head, _, body = c.text.partition("\n")
            print(f"  [{i}] {c.chunk_id} {head}\n      {' '.join(body.split())[:160]}")
    elif a.command == "serve":
        from medrag.server import serve
        serve(app, a.run, a.host, a.port)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
