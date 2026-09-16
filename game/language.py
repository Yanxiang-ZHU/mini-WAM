"""Language instruction templates and parsing.

The vocabulary is deliberately tiny: shape words, fill words, and a handful of
template patterns.  Parsing maps a natural instruction back to a semantic
(shape, fill) pair so the task generator and evaluation share one ground truth.
Templates are varied so the policy cannot memorise exact sentence strings.
"""

from __future__ import annotations

import re

SHAPE_WORDS = {"circle": "circle", "triangle": "triangle", "square": "square"}
FILL_WORDS = {"solid": "solid", "hollow": "hollow"}

# shape+fill -> list of templates (the object phrase is substituted)
FULL_TEMPLATES = [
    "Go to the {obj}.",
    "Move to the {obj}.",
    "Go toward the {obj}.",
    "Reach the {obj}.",
    "Find the {obj} and go to it.",
    "Head to the {obj}.",
    "Travel to the {obj}.",
]

# shape-only templates (for Task C)
SHAPE_TEMPLATES = [
    "Go to the {shape}.",
    "Move to the {shape}.",
    "Find the {shape}.",
    "Reach the {shape}.",
    "Go toward the {shape}.",
]

SUBTASK_VOCAB = [
    "move toward the {obj}",
    "approach the {obj}",
    "reach the {obj}",
    "navigate to the {obj}",
    "go to the {obj}",
]


def _obj_phrase(shape: str, fill: str | None) -> str:
    return f"{fill} {shape}" if fill else shape


def make_instruction(shape: str, fill: str | None, rng, use_shape_only: bool = False) -> str:
    if use_shape_only or fill is None:
        tpl = SHAPE_TEMPLATES[rng.integers(len(SHAPE_TEMPLATES))]
        return tpl.format(shape=shape)
    tpl = FULL_TEMPLATES[rng.integers(len(FULL_TEMPLATES))]
    return tpl.format(obj=_obj_phrase(shape, fill))


def make_subtask(shape: str, fill: str | None, rng) -> str:
    tpl = SUBTASK_VOCAB[rng.integers(len(SUBTASK_VOCAB))]
    return tpl.format(obj=_obj_phrase(shape, fill))


def parse_instruction(text: str) -> dict:
    """Parse a natural instruction into {'shape':..., 'fill':...}.

    ``fill`` may be None for shape-only instructions.  Raises ValueError if no
    shape word is found.
    """
    low = text.lower()
    shape = None
    for w in SHAPE_WORDS:
        if w in low:
            shape = w
            break
    if shape is None:
        raise ValueError(f"no shape word in instruction: {text!r}")
    fill = None
    for w in FILL_WORDS:
        if w in low:
            fill = w
            break
    return {"shape": shape, "fill": fill}


def parse_subtask(text: str) -> dict:
    return parse_instruction(text)
