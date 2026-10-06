"""Generation parameters chosen by the *nature of the task*.

Rule of thumb (see notes/02-llm-params-by-task.md):
- Classification / routing: temperature 0, tiny max_tokens. The answer is
  one label, so anything longer is wasted money and a parsing risk.
- Extraction (JSON): temperature 0, max_tokens just big enough for the JSON.
- Technical answers (facts, prices, lists): low temperature, narrower top_p.
- Creative answers (styling, ideas): high temperature, wider top_p.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 500

    def as_kwargs(self) -> dict:
        return asdict(self)


PRESETS: dict[str, GenParams] = {
    "classification": GenParams(temperature=0.0, top_p=1.0, max_tokens=5),
    "extraction": GenParams(temperature=0.0, top_p=1.0, max_tokens=300),
    "technical": GenParams(temperature=0.3, top_p=0.7, max_tokens=500),
    "creative": GenParams(temperature=1.0, top_p=0.9, max_tokens=700),
    "default": GenParams(temperature=0.7, top_p=0.9, max_tokens=500),
}


def params_for(task: str | None) -> GenParams:
    """Return the preset for a task label, falling back to 'default'.

    Note: the course version used `if ...: / if ...: / else:` which silently
    overwrote the 'technical' params with the fallback. A dict lookup avoids it.
    """
    return PRESETS.get((task or "").strip().lower(), PRESETS["default"])
