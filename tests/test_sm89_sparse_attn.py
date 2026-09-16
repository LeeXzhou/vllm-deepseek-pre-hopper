"""sm89 sparse-MLA attention path vs an independent torch reference.

Covers (all operating on the fp8_ds_mla 576B token layout):
  1. dequantize_and_gather_k_cache_triton  (prefill KV gather)
  2. rocm_sparse_attn_prefill               (prefill attention, bf16 gathered KV)
  3. rocm_sparse_attn_decode                (decode attention, two paged caches)

Run（需已应用补丁的 vllm 环境）:
  python tests/test_sm89_sparse_attn.py
"""

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)

torch.manual_seed(11)
dev = "cuda:0"
H = 16
HEAD_DIM, NOPE, ROPE = 512, 448, 64
DATA_BYTES, SCALE_BYTES = 576, 8  # per fp8_ds_mla token


def make_token_k(nope_fp8, scales_ue8m0, rope_bf16):
    """Pack one fp8_ds_mla token: [448B fp8 | 128B bf16 rope | 8B ue8m0]."""
    return torch.cat(
        [
            nope_fp8.view(torch.uint8),
            rope_bf16.view(torch.uint8),
            scales_ue8m0,
            torch.zeros(1, dtype=torch.uint8, device=dev),
        ]
    )


def make_cache(nb, bs, k_nope_fp8, k_rope_bf16, k_scale_ue8m0):
    """[nb*bs, 448/128/8...] per-token parts -> [nb, bs, bs*584] block layout."""
    toks = []
    for i in range(nb * bs):
        toks.append(
            make_token_k(k_nope_fp8[i], k_scale_ue8m0[i], k_rope_bf16[i])
        )
    tok = torch.stack(toks)  # [nb*bs, 584]
    blk = tok.view(nb, bs, 584)
    # block layout: [bs*576 data | bs*8 scales]
    data = blk[:, :, :576].reshape(nb, bs * 576)
    scal = blk[:, :, 576:].reshape(nb, bs * 8)
    return torch.cat([data, scal], dim=1).contiguous()  # [nb, bs*584]


def dequant_tokens(cache, nb, bs, slots):
    """Dequant fp8_ds_mla tokens at global slots -> bf16 [n, 512]."""
    flat = cache.reshape(-1)
    out = torch.empty(slots.shape[0], HEAD_DIM, device=dev, dtype=torch.bfloat16)
    for i, s in enumerate(slots.tolist()):
        b, p = s // bs, s % bs
        base = b * (bs * (DATA_BYTES + SCALE_BYTES))
        tdata = base + p * DATA_BYTES
        tscale = base + bs * DATA_BYTES + p * SCALE_BYTES
        nope_u8 = flat[tdata : tdata + NOPE]
        nope = nope_u8.view(torch.float8_e4m3fn).to(torch.float32)
        sc = flat[tscale : tscale + NOPE // 64]
        sc = torch.exp2(sc.to(torch.float32) - 127.0)
        nope = (nope * sc.repeat_interleave(64)).to(torch.bfloat16)
        rope = flat[tdata + NOPE : tdata + DATA_BYTES].view(torch.bfloat16)
        out[i] = torch.cat([nope, rope])
    return out


def softmax_attn(q, k, scale, sink):
    """Reference: scores->(+sink logit)->softmax->weighted sum. bf16 math."""
    T, Hh, D = q.shape
    out = torch.empty_like(q)
    for t in range(T):
        s = (q[t].float() @ k[t].float().T) * scale  # [H, n]
        logits = torch.cat([s, sink.float().unsqueeze(1)], dim=1)  # [H, n+1]
        p = torch.softmax(logits, dim=-1)
        v = torch.cat([k[t], torch.zeros(Hh, D, device=dev, dtype=k.dtype)], dim=0)
        # kernel casts p to bf16 before the PV dot
        out[t] = (p[:, :-1].to(torch.bfloat16).float() @ v[:-1].float()).to(
            torch.bfloat16
        )
    return out


# ---------- build shared caches ----------
NB, BS = 6, 64
N_TOK = NB * BS
k_nope = (
    (torch.randn(N_TOK, NOPE, device=dev) * 0.3).to(torch.float8_e4m3fn)
)
k_rope = torch.randn(N_TOK, ROPE, device=dev, dtype=torch.bfloat16)
# ue8m0 scales in [2^-3, 2^2]
k_scale = torch.randint(124, 129, (N_TOK, NOPE // 64), device=dev, dtype=torch.uint8)
main_cache = make_cache(NB, BS, k_nope, k_rope, k_scale)

# ---------- 1) gather test ----------
reqs = 3
seq_lens = torch.tensor([130, 200, 37], dtype=torch.int32, device=dev)
gather_lens = torch.tensor([100, 40, 37], dtype=torch.int32, device=dev)
max_blocks = 4
bt_gather = torch.tensor(
    [[1, 3, 0, 0], [2, 4, 5, 0], [0, 0, 0, 0]], dtype=torch.int32, device=dev
)
out_gather = torch.zeros(reqs, 600, HEAD_DIM, device=dev, dtype=torch.bfloat16)
dequantize_and_gather_k_cache_triton(
    out_gather,
    main_cache,
    seq_lens=seq_lens,
    gather_lens=gather_lens,
    block_table=bt_gather,
    block_size=BS,
    offset=0,
)
for r in range(reqs):
    sl, gl = int(seq_lens[r]), int(gather_lens[r])
    slots = torch.tensor(
        [
            bt_gather[r, p // BS].item() * BS + p % BS
            for p in range(sl - gl, sl)
        ],
        device=dev,
    )
    ref_rows = dequant_tokens(main_cache, NB, BS, slots)
    diff = (out_gather[r, :gl].float() - ref_rows.float()).abs()
    print(f"gather req{r}: maxdiff {diff.max().item():.3e}")
    assert diff.max().item() < 3e-2, "gather mismatch"

# ---------- 2) prefill attention test ----------
T = 9
q = torch.randn(T, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
K = 40  # indices per token
# local indices into a gathered buffer of 300 rows
kv_buf = dequant_tokens(
    main_cache, NB, BS, torch.arange(300, device=dev)
).unsqueeze(1)  # [300, 1, 512]
indices = torch.randint(0, 250, (T, 1, K), device=dev, dtype=torch.int32)
lens = torch.randint(1, K, (T,), device=dev, dtype=torch.int32)
sink = torch.full((H,), -float("inf"), device=dev, dtype=torch.float32)
sink[: H // 2] = torch.linspace(-2, 0.5, H // 2, device=dev)
out_p = torch.empty_like(q)
rocm_sparse_attn_prefill(
    q=q,
    kv=kv_buf,
    indices=indices,
    topk_length=lens,
    scale=HEAD_DIM**-0.5,
    head_dim=HEAD_DIM,
    nope_head_dim=NOPE,
    rope_head_dim=ROPE,
    attn_sink=sink,
    output=out_p,
)
# reference: per token gather rows by indices[:len], softmax with sink
qf = q.float()
for t in range(T):
    n = int(lens[t])
    idx = indices[t, 0, :n]
    k_sel = kv_buf[idx, 0]  # [n, 512] bf16
    s = (qf[t] @ k_sel.float().T) * HEAD_DIM**-0.5
    logits = torch.cat([s, sink.float().unsqueeze(1)], dim=1)
    p = torch.softmax(logits, dim=-1)[:, :-1]
    ref = (p.to(torch.bfloat16).float() @ k_sel.float()).to(torch.bfloat16)
    diff = (out_p[t].float() - ref.float()).abs()
    print(f"prefill tok{t}: maxdiff {diff.max().item():.3e}")
    assert diff.max().item() < 5e-2, "prefill attention mismatch"

# ---------- 3) decode attention test ----------
TD = 5
qd = torch.randn(TD, H, HEAD_DIM, device=dev, dtype=torch.bfloat16)
W = 32
swa_idx = torch.randint(0, N_TOK, (TD, 1, W), device=dev, dtype=torch.int32)
swa_len = torch.randint(1, W, (TD,), device=dev, dtype=torch.int32)
KT = 48
topk_idx = torch.randint(0, N_TOK, (TD, 1, KT), device=dev, dtype=torch.int32)
topk_len = torch.randint(1, KT, (TD,), device=dev, dtype=torch.int32)
out_d = torch.empty_like(qd)
rocm_sparse_attn_decode(
    q=qd,
    kv_cache=main_cache.view(NB, BS, 584),
    swa_k_cache=main_cache.view(NB, BS, 584),
    swa_only=False,
    topk_indices=topk_idx,
    topk_lens=topk_len,
    swa_indices=swa_idx,
    swa_lens=swa_len,
    swa_ragged_indices=None,
    swa_ragged_indptr=None,
    topk_ragged_indices=None,
    topk_ragged_indptr=None,
    attn_sink=sink,
    scale=HEAD_DIM**-0.5,
    head_dim=HEAD_DIM,
    nope_head_dim=NOPE,
    rope_head_dim=ROPE,
    output=out_d,
)
qdf = qd.float()
for t in range(TD):
    n1 = int(swa_len[t])
    n2 = int(topk_len[t])
    slots = torch.cat([swa_idx[t, 0, :n1], topk_idx[t, 0, :n2]])
    k_sel = dequant_tokens(main_cache, NB, BS, slots)
    s = (qdf[t] @ k_sel.float().T) * HEAD_DIM**-0.5
    logits = torch.cat([s, sink.float().unsqueeze(1)], dim=1)
    p = torch.softmax(logits, dim=-1)[:, :-1]
    ref = (p.to(torch.bfloat16).float() @ k_sel.float()).to(torch.bfloat16)
    diff = (out_d[t].float() - ref.float()).abs()
    print(f"decode tok{t}: maxdiff {diff.max().item():.3e}")
    assert diff.max().item() < 5e-2, "decode attention mismatch"

print("ALL_SPARSE_ATTN_TESTS_PASSED")
