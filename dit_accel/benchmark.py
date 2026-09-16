"""Differential DiT block (exact delta propagation) + benchmark vs the control.

Decomposes the DiTBlock into linear ops (matmul, where Δ propagates exactly:
ΔY = W·ΔX) and non-linear ops (LayerNorm / softmax / GELU / SiLU, where the raw
value is required — we "merge" the delta back to raw, apply the op, and re-derive
the delta).

The whole point is to check, empirically and honestly, whether the Cambricon-D
"differential computing" idea gives a *software* speedup for our DiT inference.
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_transformer import DiTBlock


def _mha(mha: nn.MultiheadAttention, h: torch.Tensor) -> torch.Tensor:
    """Manual MHA forward identical to nn.MultiheadAttention (no dropout, batch_first)."""
    qkv = F.linear(h, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    B, L, d = q.shape
    hn, dh = mha.num_heads, d // mha.num_heads
    q = q.view(B, L, hn, dh).transpose(1, 2)
    k = k.view(B, L, hn, dh).transpose(1, 2)
    v = v.view(B, L, hn, dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
    aw = F.softmax(scores, dim=-1)
    out = aw @ v
    out = out.transpose(1, 2).contiguous().view(B, L, d)
    return F.linear(out, mha.out_proj.weight, mha.out_proj.bias)


class DifferentialDiTBlock(nn.Module):
    """Same weights as DiTBlock, but exposes a delta-propagation forward.

    ``forward_full(x, c)`` = control (identical to DiTBlock).
    ``forward_delta(dx, dc, x, c)`` = compute Δy = F(x+dx, c+dc) − F(x, c) by
    propagating the delta through linear ops and merging at non-linear ops.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def _mod(self, c):
        return self.adaLN(c).chunk(6, dim=-1)

    def forward_full(self, x, c):
        s1, g1, s2, g2, s3, g3 = self._mod(c)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + g1.unsqueeze(1)
        x = x + _mha(self.attn, h) * g2.unsqueeze(1)
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + g3.unsqueeze(1)
        x = x + self.mlp(h)
        return x

    def forward_delta(self, dx, dc, x, c):
        """Δy = F(x+dx, c+dc) − F(x, c), computed by delta propagation.

        Linear stages use the delta directly (W·Δin); non-linear stages merge.
        """
        # adaLN (non-linear SiLU + linear): compute full mod on c and c+dc
        s1, g1, s2, g2, s3, g3 = self._mod(c)
        ns1, ng1, ns2, ng2, ns3, ng3 = self._mod(c + dc)
        ds1, dg1 = ns1 - s1, ng1 - g1

        # attention branch
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + g1.unsqueeze(1)
        nh = self.norm1(x + dx) * (1 + ns1.unsqueeze(1)) + ng1.unsqueeze(1)
        dh = nh - h
        attn = _mha(self.attn, h)
        nattn = _mha(self.attn, nh)
        dattn = nattn - attn
        x2 = x + attn * g2.unsqueeze(1)
        nx2 = x + dx + nattn * ng2.unsqueeze(1)
        dx2 = nx2 - x2

        # mlp branch
        h2 = self.norm2(x2) * (1 + s2.unsqueeze(1)) + g3.unsqueeze(1)
        nh2 = self.norm2(nx2) * (1 + ns2.unsqueeze(1)) + ng3.unsqueeze(1)
        mlp_out = self.mlp(h2)
        nmlp_out = self.mlp(nh2)
        dmlp = nmlp_out - mlp_out

        return nx2 + nmlp_out - (x2 + mlp_out)


def _rand(shape):
    return torch.randn(shape, device="cuda")


def main():
    torch.manual_seed(0)
    dim, heads = 256, 8
    blk = DiTBlock(dim, heads).cuda().eval()
    dblk = DifferentialDiTBlock(dim, heads).cuda().eval()
    # copy the (identical) weights so the two blocks are numerically the same
    dblk.load_state_dict(blk.state_dict())

    B, L = 1, 180  # (K+1)*n_patches token count, matching the real DiT

    # --- 1) correctness: forward_full must equal the control DiTBlock ---
    x = _rand((B, L, dim)); c = _rand((B, dim))
    with torch.no_grad():
        y_ctrl = blk(x, c)
        y_full = dblk.forward_full(x, c)
    print(f"[correctness] control vs forward_full maxdiff = {(y_ctrl - y_full).abs().max().item():.3e}")

    # --- 2) correctness: forward_delta must equal F(x+dx,c+dc) - F(x,c) ---
    dx = _rand((B, L, dim)) * 0.05   # small delta
    dc = _rand((B, dim)) * 0.05
    with torch.no_grad():
        dy_diff = dblk.forward_delta(dx, dc, x, c)
        dy_true = blk(x + dx, c + dc) - blk(x, c)
    print(f"[correctness] forward_delta vs true delta maxdiff = {(dy_diff - dy_true).abs().max().item():.3e}")

    # --- 3) time: full recompute vs differential ---
    def timeit(fn, iters=100, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    with torch.no_grad():
        t_full = timeit(lambda: blk(x + dx, c + dc))
        t_diff = timeit(lambda: dblk.forward_delta(dx, dc, x, c))
    print(f"\n[time] full forward (new input): {t_full:.3f} ms")
    print(f"[time] differential forward:     {t_diff:.3f} ms")
    print(f"[time] ratio (diff/full):        {t_diff/t_full:.2f}x")

    # --- 4) delta sparsity: is the input delta sparse enough for sparse matmul? ---
    print(f"\n[delta] |dx| stats: mean={dx.abs().mean().item():.3f} "
          f"frac(|dx|<1e-3)={((dx.abs()<1e-3).float().mean().item()*100):.1f}% "
          f"frac(|dx|<0.01)={((dx.abs()<0.01).float().mean().item()*100):.1f}%")


if __name__ == "__main__":
    main()
