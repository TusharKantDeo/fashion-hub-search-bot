"""Turn retrieved records into compact prompt context.

- Put only the fields the answer needs; every field is paid for on every query.
- Put the most relevant item *last* (closest to the question) — models attend
  more to the end of the prompt ("lost in the middle").
- Cap the context with a character budget as a safety net.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _props(record: Any) -> Mapping[str, Any]:
    return getattr(record, "properties", record)


def format_records(
    records: Sequence[Any],
    fields: Mapping[str, str],
    sep: str = " | ",
    max_chars: int | None = None,
) -> str:
    """fields maps a display label -> property key, e.g. {"ID": "product_id", "Name": "name"}."""
    lines: list[str] = []
    used = 0
    for r in records:
        p = _props(r)
        line = sep.join(f"{label}: {p[key]}" for label, key in fields.items() if p.get(key) not in (None, ""))
        if max_chars is not None and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def order_for_prompt(records: Sequence[Any], most_relevant_last: bool = True) -> list[Any]:
    """Search results come most-relevant-first; flip them so the best is nearest the question."""
    return list(reversed(records)) if most_relevant_last else list(records)
