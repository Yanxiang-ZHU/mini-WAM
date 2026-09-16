"""Small language encoder for the controlled instruction/subtask vocabulary.

The vocabulary is tiny and fully known (shape words, fill words, a handful of
template words), so a word-level tokenizer plus a small Transformer yields a
fixed-size conditioning vector.  Instruction and subtask share this encoder.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

# Full controlled vocabulary (lowercased).  Special tokens are reserved.
SPECIAL = ["<pad>", "<unk>", "<cls>"]
WORDS = [
    "circle", "triangle", "square", "solid", "hollow",
    "go", "to", "the", "move", "toward", "reach", "find", "and", "it",
    "head", "travel", "approach", "navigate",
]

VOCAB = SPECIAL + WORDS
IDX = {w: i for i, w in enumerate(VOCAB)}
PAD_ID = IDX["<pad>"]
UNK_ID = IDX["<unk>"]
CLS_ID = IDX["<cls>"]
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 12


def tokenize(text: str, max_len: int = MAX_LEN) -> list[int]:
    toks = [CLS_ID]
    for w in re.findall(r"[a-z]+", text.lower()):
        toks.append(IDX.get(w, UNK_ID))
    toks = toks[:max_len]
    if len(toks) < max_len:
        toks += [PAD_ID] * (max_len - len(toks))
    return toks


class LanguageEncoder(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 2, heads: int = 4,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L) -> (B, dim) via CLS token
        x = self.embed(ids) + self.pos[:, :ids.shape[1]]
        pad_mask = (ids == PAD_ID)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x[:, 0])  # CLS token
        return x
