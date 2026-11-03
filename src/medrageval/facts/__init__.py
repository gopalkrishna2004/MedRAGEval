"""Fact sheet: the structured, verified facts that question generation is built on.

Questions are written from a config of key facts rather than from raw
text. A real bundle has no such config, so this package builds one: a schema
says which facts matter for a document family, `extract` reads them out of
the parsed bundle with an LLM, and every fact is checked against a verbatim
quote before it is kept.
"""
