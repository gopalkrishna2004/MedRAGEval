"""medrag: a retrieval-augmented question-answering application over a PDF bundle.

Parsing with Docling, structure-aware chunking, Postgres + pgvector storage,
hybrid BM25 + dense retrieval with a cross-encoder reranker, and answers from
an OpenAI-compatible chat server (vLLM). Independent of the benchmark package.
"""
