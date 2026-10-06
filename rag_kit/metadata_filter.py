"""Self-querying retrieval: LLM -> JSON metadata -> Weaviate filters -> search.

Trade-off measured in the course (notes/05-metadata-filter-vs-direct-rag.md):
- Direct RAG (embed the query, search): 0 extra LLM tokens, fast, but the
  embedding can ignore hard constraints ("under $50", "for men", "blue").
- Metadata filtering: one extra LLM call (~1.5k tokens with the naive prompt),
  but hard constraints are enforced *exactly*, so results are better.

This module makes the filter path cheap enough to keep:
- compact prompt: allowed values listed once, terse rules, no prose example
- temperature 0 and a small max_tokens (JSON only)
- the model may omit keys it can't infer (no "Any" padding to generate)
- values are validated against the allowed set before building filters
- progressive relaxation if filters are too strict
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from rag_kit.llm import LLMClient, LLMResponse, user
from rag_kit.params import PRESETS, GenParams
from rag_kit.tracing import set_output, span


@dataclass
class FilterSchema:
    """What the LLM may filter on.

    categorical: property -> allowed values (taken from your data)
    numeric: properties that accept {"min": x, "max": y}
    relax_order: properties dropped first when results are too few
                 (least important first). Unlisted keys are dropped last.
    """

    categorical: dict[str, list[str]]
    numeric: list[str] = field(default_factory=list)
    relax_order: list[str] = field(default_factory=list)


def allowed_values(records: Iterable[dict], keys: list[str]) -> dict[str, list[str]]:
    values: dict[str, set] = {k: set() for k in keys}
    for r in records:
        for k in keys:
            if r.get(k) not in (None, ""):
                values[k].add(str(r[k]))
    return {k: sorted(v) for k, v in values.items()}


def parse_json_loose(text: str) -> dict | None:
    """Extract the first JSON object from LLM output (handles ```json fences, prose, inf)."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    blob = match.group(0)
    blob = re.sub(r'(?<!")\binf\b(?!")', '"inf"', blob)  # bare inf -> string
    for candidate in (blob, blob.replace("'", '"')):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def sanitize(meta: dict | None, schema: FilterSchema) -> dict[str, Any]:
    """Keep only known keys and allowed values; drop empty or 'Any'."""
    if not meta:
        return {}
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if key in schema.categorical:
            vals = value if isinstance(value, list) else [value]
            allowed = {a.lower(): a for a in schema.categorical[key]}
            kept = [allowed[str(v).lower()] for v in vals if str(v).lower() in allowed]
            if kept:
                clean[key] = kept
        elif key in schema.numeric and isinstance(value, dict):
            lo, hi = value.get("min"), value.get("max")
            lo = float(lo) if isinstance(lo, (int, float)) and lo > 0 else None
            hi = float(hi) if isinstance(hi, (int, float)) else None
            if lo is not None or hi is not None:
                clean[key] = {"min": lo, "max": hi}
    return clean


def to_weaviate_filters(meta: dict[str, Any]) -> dict[str, Any]:
    """Return {property: Filter} so relaxation can drop filters by property name."""
    from weaviate.classes.query import Filter

    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, dict):
            parts = []
            if value.get("min") is not None:
                parts.append(Filter.by_property(key).greater_or_equal(value["min"]))
            if value.get("max") is not None:
                parts.append(Filter.by_property(key).less_or_equal(value["max"]))
            out[key] = parts[0] if len(parts) == 1 else Filter.all_of(parts)
        else:
            out[key] = Filter.by_property(key).contains_any(value)
    return out


def combine(filters: dict[str, Any]) -> Any:
    from weaviate.classes.query import Filter

    parts = list(filters.values())
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else Filter.all_of(parts)


class MetadataFilterExtractor:
    def __init__(
        self,
        llm: LLMClient,
        schema: FilterSchema,
        domain: str = "items",
        params: GenParams = PRESETS["extraction"],
        model: str | None = None,
    ):
        self.llm = llm
        self.schema = schema
        self.domain = domain
        self.params = params
        self.model = model

    def build_prompt(self, query: str) -> str:
        allowed = json.dumps(self.schema.categorical, separators=(",", ":"))
        numeric = ", ".join(self.schema.numeric)
        num_rule = f' For {numeric} use {{"min":n,"max":n}} (omit unknown bounds).' if numeric else ""
        return (
            f"Extract search filters for {self.domain} from the query as JSON.\n"
            f"Allowed values: {allowed}\n"
            f"Rules: values must be lists using only allowed values; include only keys the query "
            f"implies (related values are fine, e.g. a 'look' implies several article types).{num_rule} "
            f"Output JSON only.\n"
            f"Query: {query}"
        )

    def extract(self, query: str) -> tuple[dict[str, Any], LLMResponse]:
        """Return (validated metadata, raw LLM response — use .total_tokens for cost)."""
        with span("generate_metadata", kind="chain", input={"query": query}) as s:
            resp = self.llm.complete(user(self.build_prompt(query)), self.params, self.model, "metadata_llm")
            meta = sanitize(parse_json_loose(resp.content), self.schema)
            set_output(s, meta)
            return meta, resp


def search_with_relaxation(
    store,
    query: str,
    filters: dict[str, Any],
    relax_order: list[str],
    limit: int = 20,
    min_results: int = 5,
):
    """Search with all filters; if too few hits, drop filters in relax_order and retry.

    Returns (docs, active_filter_keys).
    """
    active = dict(filters)
    order = [k for k in relax_order if k in active] + [k for k in active if k not in relax_order]
    with span("search_with_relaxation", kind="chain", input={"query": query, "filters": list(active)}) as s:
        while True:
            docs = store.search(query, limit=limit, filters=combine(active))
            if len(docs) >= min_results or not order:
                set_output(s, {"active_filters": list(active), "n": len(docs)})
                return docs, list(active)
            active.pop(order.pop(0))
