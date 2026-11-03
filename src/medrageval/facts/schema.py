"""The fact-sheet schema: which facts to extract from a bundle, and how they link.

A schema describes a document family (NICE single technology appraisals),
not one bundle. It has groups of slots, and for each group:
- the document roles it is read from;
- section-path patterns and keywords that pick the passages to read;
- a weight that sets the group's share of the dataset;
- links between slots, for multi-hop items.

The documents are real, so the schema drives extraction rather than
document generation.
`draft_schema` asks an LLM for a first version for a new document family,
which a person reviews before use.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from medrageval.config import DEFAULT_CONFIG

KINDS = {"number", "text", "stance"}
REPO_ROOT = DEFAULT_CONFIG.parents[1]
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaError(ValueError):
    """The schema file is malformed: a wrong kind, an unknown role or a link to a missing slot."""


@dataclass(frozen=True)
class Slot:
    group: str
    name: str
    kind: str
    describe: str
    qualifiers: tuple[str, ...] = ()
    per: str | None = None            # number slots: values keyed by this (arm, comparator, technology)
    parties: tuple[str, ...] = ()     # stance slots: who may hold a position
    ask: bool = True                  # False: metadata used by other items (a respondent's name), never asked about

    @property
    def id(self) -> str:
        return f"{self.group}.{self.name}"


@dataclass(frozen=True)
class Group:
    id: str
    weight: float
    describe: str
    roles: tuple[str, ...]
    sections: tuple[str, ...]
    keywords: tuple[str, ...]
    slots: dict[str, Slot] = field(default_factory=dict)


MATCHES = {"value", "qualifier"}


@dataclass(frozen=True)
class Link:
    """Multi-hop: a `source` fact identifies the thing a `target` fact is about (its `via` qualifier, or one of its
    `per` labels), and a question refers to that thing through the source instead of naming it.

    match "value": the source's value names it ("the primary endpoint of {trial}" is PFS; the target is PFS's median).
    match "qualifier": the source's `source_via` qualifier names it and the source's value describes it ("the
    comparator the company ruled out because of TA239" is fulvestrant; the target is fulvestrant's cost).
    """
    source: str
    target: str
    via: str
    describe: str     # how a question refers to the bridge: {qualifier} placeholders from the source, {value} its value
    match: str = "value"
    source_via: str | None = None


@dataclass
class Schema:
    name: str
    version: int
    groups: dict[str, Group]
    links: list[Link]
    path: Path | None = None

    @property
    def slots(self) -> dict[str, Slot]:
        return {s.id: s for g in self.groups.values() for s in g.slots.values()}

    def slot(self, slot_id: str) -> Slot:
        try:
            return self.slots[slot_id]
        except KeyError:
            raise SchemaError(f"unknown slot {slot_id!r}") from None

    def weights(self) -> dict[str, float]:
        return {g.id: g.weight for g in self.groups.values()}


def _strs(value, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SchemaError(f"{where}: expected a list of strings, got {value!r}")
    return tuple(value)


def parse_schema(data: dict, roles: Iterable[str], path: Path | None = None) -> Schema:
    """Build and check a Schema from its YAML/JSON form. `roles` are the document roles the corpus knows."""
    if not isinstance(data, dict) or not isinstance(data.get("groups"), dict) or not data["groups"]:
        raise SchemaError("a schema needs a non-empty 'groups' mapping")
    known_roles = set(roles)
    groups: dict[str, Group] = {}
    for gid, g in data["groups"].items():
        where = f"group {gid}"
        if not _ID.match(str(gid)):
            raise SchemaError(f"{where}: ids are lower_snake_case")
        if not isinstance(g, dict):
            raise SchemaError(f"{where}: expected a mapping")
        group_roles = _strs(g.get("roles"), f"{where}.roles")
        if not group_roles:
            raise SchemaError(f"{where}: no roles")
        if unknown := [r for r in group_roles if r not in known_roles]:
            raise SchemaError(f"{where}: unknown roles {unknown}; known: {sorted(known_roles)}")
        sections = _strs(g.get("sections"), f"{where}.sections")
        for pat in sections:
            try:
                re.compile(pat)
            except re.error as e:
                raise SchemaError(f"{where}: bad section pattern {pat!r}: {e}") from e
        weight = g.get("weight", 1.0)
        if isinstance(weight, str):   # "0.15" from an LLM draft
            try:
                weight = float(weight)
            except ValueError:
                pass
        if not isinstance(weight, (int, float)) or weight < 0:
            raise SchemaError(f"{where}: weight must be a number >= 0")
        slots: dict[str, Slot] = {}
        for name, s in (g.get("slots") or {}).items():
            sw = f"slot {gid}.{name}"
            if not _ID.match(str(name)) or not isinstance(s, dict):
                raise SchemaError(f"{sw}: expected a lower_snake_case id with a mapping")
            kind = s.get("kind")
            if kind not in KINDS:
                raise SchemaError(f"{sw}: kind must be one of {sorted(KINDS)}, got {kind!r}")
            if not isinstance(s.get("describe"), str) or not s["describe"].strip():
                raise SchemaError(f"{sw}: needs a 'describe' sentence")
            per = s.get("per")
            if per is not None and (kind != "number" or not isinstance(per, str)):
                raise SchemaError(f"{sw}: 'per' is only for number slots")
            parties = _strs(s.get("parties"), f"{sw}.parties")
            if kind == "stance" and not parties:
                raise SchemaError(f"{sw}: a stance slot lists its parties")
            slots[name] = Slot(gid, name, kind, s["describe"].strip(), _strs(s.get("qualifiers"), f"{sw}.qualifiers"),
                               per, parties, bool(s.get("ask", True)))
        if not slots:
            raise SchemaError(f"{where}: no slots")
        groups[gid] = Group(gid, float(weight), str(g.get("describe") or gid.replace("_", " ")), group_roles, sections,
                            _strs(g.get("keywords"), f"{where}.keywords"), slots)
    schema = Schema(str(data.get("name") or "schema"), int(data.get("version") or 1), groups, [], path)
    all_slots = schema.slots
    for i, ln in enumerate(data.get("links") or []):
        where = f"link {i + 1}"
        if not isinstance(ln, dict) or not all(isinstance(ln.get(k), str) for k in ("from", "to", "via", "describe")):
            raise SchemaError(f"{where}: needs from, to, via and describe")
        for end in ("from", "to"):
            if ln[end] not in all_slots:
                raise SchemaError(f"{where}: unknown slot {ln[end]!r}")
        target = all_slots[ln["to"]]
        if ln["via"] not in target.qualifiers and ln["via"] != target.per:
            raise SchemaError(f"{where}: {ln['to']} has no qualifier or per key {ln['via']!r}")
        match = str(ln.get("match") or "value")
        if match not in MATCHES:
            raise SchemaError(f"{where}: match must be one of {sorted(MATCHES)}, got {match!r}")
        source_via = ln.get("source_via")
        if match == "qualifier" and source_via not in all_slots[ln["from"]].qualifiers:
            raise SchemaError(f"{where}: a qualifier link names a source_via qualifier of {ln['from']}, got {source_via!r}")
        schema.links.append(Link(ln["from"], ln["to"], ln["via"], ln["describe"], match, source_via if match == "qualifier" else None))
    if sum(g.weight for g in groups.values()) <= 0:
        raise SchemaError("at least one group needs a weight above 0")
    return schema


def resolve_path(path: str | Path) -> Path:
    """A schema path as given, or relative to the repository root (where configs/ lives)."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def load_schema(path: str | Path, roles: Iterable[str]) -> Schema:
    p = resolve_path(path)
    if not p.exists():
        raise SchemaError(f"schema file not found: {p}")
    with open(p) as f:
        return parse_schema(yaml.safe_load(f), roles, p)


def known_roles(cfg: dict) -> list[str]:
    """Document roles from the generate config: the ones the corpus can assign."""
    return list(cfg["roles"]["patterns"]) + ["other"]
