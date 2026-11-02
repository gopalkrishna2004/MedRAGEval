import argparse
from pathlib import Path

from medrageval.config import load_config
from medrageval.facts import draft as draft_stage
from medrageval.facts import extract as extract_stage
from medrageval.generate import generate as generate_stage
from medrageval.parse.pipeline import parse_pdf, print_report
from medrageval.rag import evaluate as rag_eval


def main() -> None:
    ap = argparse.ArgumentParser(prog="medrageval")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="parse a PDF into documents, sections and elements")
    p.add_argument("pdf", type=Path)
    p.add_argument("--out", type=Path, help="output directory (default: data/interim/<pdf name>)")
    p.add_argument("--config", type=Path)
    p.add_argument("--force", action="store_true", help="re-run conversion even if cached")

    s = sub.add_parser("schema", help="fact-sheet schemas")
    ssub = s.add_subparsers(dest="schema_command", required=True)
    draft_stage.add_arguments(ssub.add_parser("draft", help="draft a schema for a new document family with an LLM, for review"))

    e = sub.add_parser("extract", help="extract the fact sheet from a parse output directory")
    extract_stage.add_arguments(e)

    g = sub.add_parser("generate", help="generate an evaluation dataset from a parse output directory and its fact sheet")
    generate_stage.add_arguments(g)

    i = sub.add_parser("index", help="build the RAG's chunks and embeddings for a parse output directory")
    rag_eval.add_index_arguments(i)

    q = sub.add_parser("ask", help="answer one question with the RAG and show the passages it used")
    rag_eval.add_ask_arguments(q)

    v = sub.add_parser("eval", help="run the RAG over evaluation datasets and score its answers")
    rag_eval.add_arguments(v)

    args = ap.parse_args()
    if args.command == "parse":
        out = args.out or Path("data/interim") / args.pdf.stem
        print_report(parse_pdf(args.pdf, out, load_config(args.config), force=args.force))
        print(f"outputs in {out}/")
    elif args.command == "schema":
        draft_stage.run_cli(args)
    elif args.command == "extract":
        extract_stage.run_cli(args)
    elif args.command == "generate":
        generate_stage.run_cli(args)
    elif args.command == "index":
        rag_eval.run_index_cli(args)
    elif args.command == "ask":
        rag_eval.run_ask_cli(args)
    elif args.command == "eval":
        rag_eval.run_cli(args)


if __name__ == "__main__":
    main()
