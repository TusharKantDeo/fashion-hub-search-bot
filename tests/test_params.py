from rag_kit.params import PRESETS, params_for


def test_technical_is_not_overwritten_by_fallback():
    # Regression for the course's if/if/else bug.
    assert params_for("technical") == PRESETS["technical"]


def test_unknown_and_none_fall_back_to_default():
    assert params_for("weird") == PRESETS["default"]
    assert params_for(None) == PRESETS["default"]


def test_classification_is_deterministic_and_tiny():
    p = PRESETS["classification"]
    assert p.temperature == 0 and p.max_tokens <= 10
