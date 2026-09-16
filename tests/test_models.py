import torch

from models.language_encoder import tokenize, VOCAB_SIZE
from models.world_model import WorldModel, unpatchify
from models.action_expert import ActionExpert
from models.simple_policy import SimplePolicy


def _cfg(**kw):
    c = dict(hidden_dim=128, layers=2, heads=4, patch=8, history=4,
             action_horizon=8, num_actions=4, denoise_steps=4,
             lang_layers=2, lang_heads=4)
    c.update(kw)
    return c


def test_tokenize_length():
    toks = tokenize("go to the hollow circle")
    assert len(toks) == 12
    assert toks[0] != 0  # CLS is not pad


def test_unpatchify_roundtrip():
    B, patch = 2, 8
    n = 36
    x = torch.randn(B, n, patch * patch)
    img = unpatchify(x, patch, 48, 48)
    assert img.shape == (B, 1, 48, 48)


def test_world_model_forward():
    m = WorldModel(_cfg())
    B = 2
    x_t = torch.randn(B, 1, 48, 48)
    t = torch.rand(B)
    history = torch.randn(B, 4, 48, 48)
    lang = torch.randint(0, VOCAB_SIZE, (B, 12))
    sub = torch.randint(0, VOCAB_SIZE, (B, 12))
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False,
             "control_mode": "keyboard"}] * B
    v = m(x_t, t, history, lang, sub, meta)
    assert v.shape == (B, 1, 48, 48)


def test_action_expert_no_subgoal():
    m = ActionExpert(_cfg(use_subgoal=False))
    B = 2
    a_t = torch.randn(B, 8, 4)
    t = torch.rand(B)
    history = torch.randn(B, 4, 48, 48)
    lang = torch.randint(0, VOCAB_SIZE, (B, 12))
    sub = torch.randint(0, VOCAB_SIZE, (B, 12))
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False,
             "control_mode": "keyboard"}] * B
    v = m(a_t, t, history, None, lang, sub, meta)
    assert v.shape == (B, 8, 4)


def test_action_expert_with_subgoal():
    m = ActionExpert(_cfg(use_subgoal=True))
    B = 2
    a_t = torch.randn(B, 8, 4)
    t = torch.rand(B)
    history = torch.randn(B, 4, 48, 48)
    subgoal = torch.randn(B, 1, 48, 48)
    lang = torch.randint(0, VOCAB_SIZE, (B, 12))
    sub = torch.randint(0, VOCAB_SIZE, (B, 12))
    meta = [{"speed": "normal", "quality": 1.0, "mistake": False,
             "control_mode": "keyboard"}] * B
    v = m(a_t, t, history, subgoal, lang, sub, meta)
    assert v.shape == (B, 8, 4)


def test_simple_policy_forward():
    m = SimplePolicy(_cfg())
    B = 2
    history = torch.randn(B, 4, 48, 48)
    lang = torch.randint(0, VOCAB_SIZE, (B, 12))
    logits = m(history, lang)
    assert logits.shape == (B, 4)
