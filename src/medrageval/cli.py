import argparse
from pathlib import Path

from medrageval.config import load_config
from medrageval.generate import generate as generate_stage
from medrageval.parse.pipeline import parse_pdf, print_report


def main() -> None:
    ap = argparse.ArgumentParser(prog="medrageval")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="parse a PDF into documents, sections and elements")
    p.add_argument("pdf", type=Path)
    p.add_argument("--out", type=Path, help="output directory (default: data/interim/<pdf name>)")
    p.add_argument("--config", type=Path)
    p.add_argument("--force", action="store_true", help="re-run conversion even if cached")

    g = sub.add_parser("generate", help="generate an evaluation dataset from a parse output directory")
    generate_stage.add_arguments(g)

    args = ap.parse_args()
    if args.command == "parse":
        out = args.out or Path("data/interim") / args.pdf.stem
        print_report(parse_pdf(args.pdf, out, load_config(args.config), force=args.force))
        print(f"outputs in {out}/")
    elif args.command == "generate":
        generate_stage.run_cli(args)


if __name__ == "__main__":
    main()
