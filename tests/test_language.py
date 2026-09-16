import numpy as np

from game.language import (parse_instruction, make_instruction, make_subtask,
                           parse_subtask)


def test_parse_full():
    r = parse_instruction("Go to the hollow circle.")
    assert r == {"shape": "circle", "fill": "hollow"}


def test_parse_shape_only():
    r = parse_instruction("Go to the triangle.")
    assert r == {"shape": "triangle", "fill": None}


def test_parse_missing_shape_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_instruction("Do something.")


def test_instruction_contains_semantics():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ins = make_instruction("square", "solid", rng)
        assert "solid" in ins and "square" in ins


def test_shape_only_instruction_no_fill_word():
    rng = np.random.default_rng(1)
    for _ in range(50):
        ins = make_instruction("circle", "solid", rng, use_shape_only=True)
        assert "solid" not in ins and "hollow" not in ins
        assert "circle" in ins


def test_subtask_parse_roundtrip():
    rng = np.random.default_rng(2)
    sub = make_subtask("triangle", "hollow", rng)
    parsed = parse_subtask(sub)
    assert parsed["shape"] == "triangle"
    assert parsed["fill"] == "hollow"


def test_template_variety():
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(200):
        seen.add(make_instruction("circle", "solid", rng))
    assert len(seen) > 1
