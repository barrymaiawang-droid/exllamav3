"""
Ornith-1.5-122B (Qwen3_5Moe, EXL3 mul1) decode shapes against the independent trellis reference.

Shape source: the pack's own quantization_config.json (read-only), per-module trellis geometry:

    experts gate/up      k=3072  n=1024   K=4   (12288 tensors)
    experts down         k=1024  n=3072   K=4   (12288 tensors)
    shared expert          k=3072  n=1024 / k=1024 n=3072   K=5
    linear_attn in_proj_qkv  k=3072 n=12288  K=5
    linear_attn in_proj_z    k=3072 n=8192   K=5
    linear_attn out_proj     k=8192 n=3072   K=5
    self_attn q_proj         k=3072 n=16384  K=5
    self_attn k/v_proj       k=3072 n=512    K=5
    self_attn o_proj         k=8192 n=3072   K=5
    lm_head                  k=3072 n=248320 K=6

Every tensor is codebook mul1 at integer K, so the ROCm dispatch (exl3_gemm, m = 1..8) takes the RDNA
multirow GEMV for all of them. m = 1 is the autoregressive decode shape; m = 3 is the MTP verifier with
num_draft_tokens = 2 (two drafts plus the bonus position); m = 8 is the widest m the RDNA path accepts.

Reference: testlib.trellis.linear, the float64 product with the independently decoded weight. Tolerance is
the one test_exl3_gemv.py states for this kernel, unchanged: relative RMS < 2.5e-3 and max error < 4e-3 of
the output's max (a few fp16 unit roundoffs; a mis-mapped tile is O(1)).
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.exl3 import generator, rand_scale

AR_SHAPES = [
    ("experts.gate_up", 3072, 1024, 4),
    ("experts.down", 1024, 3072, 4),
    ("shared_expert.gate_up", 3072, 1024, 5),
    ("shared_expert.down", 1024, 3072, 5),
    ("linear_attn.in_proj_z", 3072, 8192, 5),
    ("linear_attn.in_proj_qkv", 3072, 12288, 5),
    ("self_attn.q_proj", 3072, 16384, 5),
    ("self_attn.k_proj", 3072, 512, 5),
    ("self_attn.o_proj", 8192, 3072, 5),
]

VERIFY_SHAPES = [
    ("experts.gate_up", 3072, 1024, 4, 3),
    ("experts.down", 1024, 3072, 4, 3),
    ("experts.gate_up.m8", 3072, 1024, 4, 8),
]

LM_HEAD = (3072, 248320, 6)
REF_CHUNK_TILES = 2048      # 32768 columns per reference chunk, a multiple of 128


def _packed(k, n, K, seed):
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    return p, rand_scale(k, g), rand_scale(n, g), g


def _ref(A, p, suh, svh, K):
    """tref.linear over column chunks; chunk edges are multiples of 128, where the Hadamards block"""
    n = p.shape[1] * 16
    out = []
    for t0 in range(0, p.shape[1], REF_CHUNK_TILES):
        t1 = min(p.shape[1], t0 + REF_CHUNK_TILES)
        out.append(tref.linear(A, p[:, t0:t1], suh, svh[t0 * 16:t1 * 16], K, "mul1"))
    return torch.cat(out, dim=-1)


def _gemm(A, trellis, suh, svh, n):
    C = torch.empty((A.shape[0], n), dtype=torch.half, device=A.device)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, True, 0)
    return C


def _check(C, ref):
    e = C.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    return rel_rms, rel_max


@pytest.mark.parametrize("name, k, n, K", AR_SHAPES)
@torch.inference_mode()
def test_ornith_ar_shape_m1(device, name, k, n, K):
    """One row: the autoregressive decode of every EXL3 linear in the pack"""
    p, suh, svh, g = _packed(k, n, K, k * 7 + n + K)
    A = (torch.randn(1, k, generator=g) * 0.5).half().to(device)
    ref = _ref(A, p, suh, svh, K)
    C = _gemm(A, torch.from_numpy(p.copy()).to(device), suh.to(device), svh.to(device), n)
    assert torch.isfinite(C).all()
    rel_rms, rel_max = _check(C, ref)
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"{name}: rel_rms {rel_rms:.2e} rel_max {rel_max:.2e}"


@pytest.mark.parametrize("name, k, n, K, m", VERIFY_SHAPES)
@torch.inference_mode()
def test_ornith_mtp_verify_shape(device, name, k, n, K, m):
    """m = num_draft_tokens + 1 rows in one forward: the MTP verifier batch on the multirow kernel"""
    p, suh, svh, g = _packed(k, n, K, k + n + K * 31 + m)
    A = (torch.randn(m, k, generator=g) * 0.5).half().to(device)
    ref = _ref(A, p, suh, svh, K)
    tr = torch.from_numpy(p.copy()).to(device)
    C = _gemm(A, tr, suh.to(device), svh.to(device), n)
    assert torch.isfinite(C).all()
    rel_rms, rel_max = _check(C, ref)
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"{name} m={m}: rel_rms {rel_rms:.2e} rel_max {rel_max:.2e}"
    C2 = _gemm(A, tr, suh.to(device), svh.to(device), n)
    assert torch.equal(C, C2), f"{name} m={m}: repeat launch differs"


@torch.inference_mode()
def test_ornith_lm_head_full_width(device):
    """The widest mul1 GEMV in the pack (n = 248320, K = 6). Full-width launch; the reference covers the
    first 32768 columns (2048 tiles), which is enough to catch a wrong wide-config mapping."""
    k, n, K = LM_HEAD
    ref_tiles = 2048
    p, suh, svh, g = _packed(k, n, K, 424242)
    A = (torch.randn(1, k, generator=g) * 0.5).half().to(device)
    ref = tref.linear(A, p[:, :ref_tiles], suh, svh[:ref_tiles * 16], K, "mul1")
    C = _gemm(A, torch.from_numpy(p.copy()).to(device), suh.to(device), svh.to(device), n)
    assert C.shape == (1, n)
    assert torch.isfinite(C).all()
    rel_rms, rel_max = _check(C[:, :ref_tiles * 16], ref)
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"lm_head: rel_rms {rel_rms:.2e} rel_max {rel_max:.2e}"
