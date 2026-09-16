"""sm89 Triton MQA-logits kernels vs the shipped torch reference impls.

Run（需已应用补丁的 vllm 环境）:
  python tests/test_sm89_mqa_logits.py
"""

import torch

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits_torch,
)
from vllm.v1.attention.ops.sm89_mqa_logits import (
    fp8_mqa_logits_sm89,
    fp8_paged_mqa_logits_sm89,
)

torch.manual_seed(7)
dev = "cuda:0"
H, D = 64, 128


def rand_fp8(*shape, scale=0.3):
    return (torch.randn(*shape, device=dev) * scale).to(torch.float8_e4m3fn)


def rand_fp8_bytes(*shape):
    """Random uint8 bytes avoiding the e4m3 NaN patterns 0x7F/0xFF."""
    v = torch.randint(0, 254, shape, device=dev, dtype=torch.int32)
    v = torch.where(v >= 127, v + 1, v).to(torch.uint8)
    return v


def compare(a, b, name):
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    valid = torch.isfinite(a) & torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), torch.isfinite(b)), (
        f"{name}: -inf pattern mismatch"
    )
    diff = (a[valid] - b[valid]).abs()
    # relu() makes many logits ~0, so plain relative error is meaningless
    # there; compare against the global magnitude of the reference row set.
    gscale = b[valid].abs().max().item()
    rel = diff / gscale
    print(
        f"{name}: valid={valid.sum().item()}/{a.numel()} "
        f"maxabs={diff.max().item():.3e} scale={gscale:.3e} "
        f"maxrel_to_scale={rel.max().item():.3e}"
    )
    assert diff.max().item() < 1e-2, f"{name}: absolute error too large"
    assert rel.max().item() < 5e-3, f"{name}: scaled error too large"


# ---- contiguous (prefill) ----
M, N = 37, 500
q = rand_fp8(M, H, D)
k = rand_fp8(N, D)
k_scale = (torch.rand(N, 1, device=dev) * 0.05 + 0.01).float()
w = torch.randn(M, H, device=dev, dtype=torch.float32) * 0.2
ks = torch.zeros(M, dtype=torch.int32, device=dev)
ke = torch.clamp(
    torch.arange(1, M + 1, device=dev, dtype=torch.int32) * 13, max=N
)
ref = fp8_mqa_logits_torch(q, (k, k_scale), w, ks, ke)
out = fp8_mqa_logits_sm89(q, k, k_scale, w, ks, ke)
compare(out, ref, "contiguous")

# ---- paged (decode) ----
# The shipped torch reference's paged path mishandles the [block, 1, D+4]
# cache layout (wrong finite counts / values), so validate against an
# independent implementation of the documented semantics:
#   logit[m, j] = k_scale[j] * sum_h w[m, h] * relu(dot(q[m, h], k[j]))
B, block, n_blocks, mml = 4, 64, 16, 700
q1 = rand_fp8(B, 1, H, D)
# indexer cache per-block layout: [bs*D values | bs*4 fp32 scales] per block
cache = rand_fp8_bytes(n_blocks, block, 1, D + 4).clone()
scales = (torch.rand(n_blocks, block, device=dev) * 0.05 + 0.01).float()
cf = cache.view(n_blocks, -1)
cf[:, : block * D] = rand_fp8_bytes(n_blocks, block * D)
cf[:, block * D :] = scales.view(torch.uint8).reshape(n_blocks, block * 4)
bt = torch.stack(
    [torch.randperm(n_blocks, device=dev)[:11] for _ in range(B)]
).to(torch.int32)


def paged_ref(q, cache, weights, context_lens, block_tables, max_model_len):
    """Independent reference for the paged MQA-logits semantics."""
    B_, nn_, H_, D_ = q.shape
    rows = B_ * nn_
    flat = cache.view(n_blocks, block * (D_ + 4))
    k_val = flat[:, : block * D_].view(torch.float8_e4m3fn).float().view(
        n_blocks, block, D_
    )
    k_sc = flat[:, block * D_ :].view(torch.float32)  # [nb, bs]
    logits = torch.full((rows, max_model_len), float("-inf"), device=q.device)
    if context_lens.dim() > 1:
        context_lens = context_lens.reshape(-1)
    for r in range(rows):
        ctx = int(context_lens[r].item())
        pages = block_tables[r // nn_, : (ctx + block - 1) // block]
        vals = k_val[pages].reshape(-1, D_)[:ctx]
        sc = k_sc[pages].reshape(-1)[:ctx]
        qf = q.reshape(rows, H_, D_)[r].float()
        s = torch.relu(vals @ qf.T) * weights[r][None, :]  # [ctx, H]
        logits[r, :ctx] = s.sum(dim=1) * sc
    return logits


nn = 3
q2 = rand_fp8(B, nn, H, D)
base = torch.tensor([200, 400, 128, 600], dtype=torch.int32, device=dev)
ctx2 = torch.stack(
    [base + i + 1 for i in range(nn)], dim=1
)  # [B, next_n] causal limits
w2 = torch.randn(B * nn, H, device=dev, dtype=torch.float32) * 0.2
ref2 = paged_ref(q2, cache, w2, ctx2, bt, mml)
out2 = fp8_paged_mqa_logits_sm89(q2, cache, w2, ctx2, bt, mml)
compare(out2, ref2, "paged next_n=3")

# next_n=1 cross-check: draft step 0 of each sequence sees ctx = base+1,
# identical to a standalone next_n=1 decode row.
out_n1 = fp8_paged_mqa_logits_sm89(
    q2[:, :1], cache, w2[0::3].contiguous(), ctx2[:, :1], bt, mml
)
compare(out_n1, ref2[0::3], "paged next_n=1 (vs draft-step-0 rows)")

print("ALL_MQA_LOGITS_TESTS_PASSED")
