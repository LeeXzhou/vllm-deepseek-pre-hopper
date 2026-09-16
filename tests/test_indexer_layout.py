"""Decisive layout audit for the indexer K cache (sm89 CPU-KV-pool mode).

The cache is constructed EXACTLY the way the real writer
``_fused_kv_compress_norm_rope_insert_indexer_attn`` stores it (verified by
code reading of fused_compress_quant_cache.py):

    per block (block_size=256, head_dim=128):
      [0, bs*128)      fp8 values, 128 contiguous bytes per token
      [bs*128, bs*132) fp32 scales, 4 bytes per token

Checks:
  [A] C++ ``_C.cp_gather_indexer_k_quant_cache`` — matches the plain
      writer layout for single- AND multi-request chunks (earlier encoding
      probes that suggested multi-request garbling were an artifact of
      fp8-NaN byte patterns 0x7F/0xFF in the synthetic encoding; value-level
      comparison with clean bytes is authoritative).
  [B] ``sm89_cp_gather_indexer_k_quant_cache`` (the sm89 Triton gather used
      for BOTH the CPU pinned pool and GPU pools since 2026-09-15) must be
      byte-exact vs the plain-layout reference, for pinned-CPU and GPU, for
      single- and multi-request chunks with non-block-aligned lengths.
      HISTORY: until 2026-09-15 the CPU-pool path delegated to the ROCm
      ``cp_gather_indexer_k_quant_cache_triton``, which reads a 16x16-tile
      SHUFFLE layout for block_size != 1 — 0/1000 rows correct vs the
      writer (see /tmp/final_gather_check.py repro) — scrambling every
      prefill indexer K vector and corrupting top-512 selection.
  [C] ``top_k_per_row_prefill`` selects within [cu_ks, cu_ke) and returns
      cu_ks-relative LOCAL indices — the contract downstream
      combine_topk_swa_indices (+ M*batch_idx) expects.

Run（需已应用补丁的 vllm 环境）:
  python tests/test_indexer_layout.py
"""

import torch

from vllm._custom_ops import cp_gather_indexer_k_quant_cache
from vllm.models.deepseek_v4.nvidia.sm89_cpu_pool import (
    sm89_cp_gather_indexer_k_quant_cache,
)

torch.manual_seed(0)
dev = "cuda:0"

BLOCK = 256  # indexer cache block size
D = 128      # index_head_dim
N_BLOCKS = 40
BS = 3       # requests in one chunk, non-aligned lengths


def build_cache(device):
    """Byte at (block, tok, d) encodes (block, tok, d); scale = 1+block/8."""
    cache = torch.zeros(N_BLOCKS, BLOCK, D + 4, dtype=torch.uint8, device=device)
    flat = cache.view(N_BLOCKS, -1)
    blocks = torch.arange(N_BLOCKS, device=device)
    tok_ids = torch.arange(BLOCK, device=device)
    vals = (
        blocks[:, None, None] * 7
        + tok_ids[None, :, None] * 3
        + torch.arange(D, device=device)[None, None, :]
    ) % 251
    flat[:, : BLOCK * D] = vals.reshape(N_BLOCKS, -1).to(torch.uint8)
    scales = (1.0 + blocks / 8.0).to(torch.float32)[:, None].expand(N_BLOCKS, BLOCK)
    flat[:, BLOCK * D :] = (
        scales.reshape(-1).contiguous().view(torch.uint8).reshape(N_BLOCKS, -1)
    )
    return cache


def gather_reference(cache, block_table, lens):
    """Plain-layout ground truth in the WRITER's flat addressing:
    cache block viewed as [bs*132]: values at t*128, scale at bs*128 + t*4."""
    flat = cache.view(cache.shape[0], -1)
    rows_k, rows_s = [], []
    for seq, L in enumerate(lens):
        for tok in range(L):
            b = int(block_table[seq, tok // BLOCK].item())
            t = tok % BLOCK
            rows_k.append(flat[b, t * D : t * D + D])
            rows_s.append(
                flat[b, BLOCK * D + t * 4 : BLOCK * D + t * 4 + 4].view(torch.float32)
            )
    return torch.stack(rows_k), torch.stack(rows_s)


def check_gather(cache, block_table, cu, lens, total, tag):
    ref_k, ref_s = gather_reference(cache.cpu(), block_table.cpu(), lens)
    kq = torch.zeros(total, D, dtype=torch.float8_e4m3fn, device=dev)
    ks = torch.zeros(total, 4, dtype=torch.uint8, device=dev)
    sm89_cp_gather_indexer_k_quant_cache(cache, kq, ks, block_table, cu)
    torch.cuda.synchronize()
    ok_v = torch.equal(kq.view(torch.uint8).cpu(), ref_k)
    ok_s = torch.equal(ks.view(torch.float32).cpu(), ref_s)
    print(f"[B] {tag}: values {'MATCH' if ok_v else 'MISMATCH'}, "
          f"scales {'MATCH' if ok_s else 'MISMATCH'}")
    assert ok_v and ok_s, f"{tag}: sm89 gather is not writer-layout exact"


def main():
    lens = [700, 256, 41]  # 3 requests, lengths not multiples of BLOCK
    total = sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    # distinct physical blocks per request, shuffled order
    perms = [torch.randperm(N_BLOCKS)[: (L + BLOCK - 1) // BLOCK + 1] for L in lens]
    width = max(len(p) for p in perms) + 2
    block_table = torch.zeros(BS, width, dtype=torch.int32, device=dev)
    for i, p in enumerate(perms):
        block_table[i, : len(p)] = p.to(torch.int32).to(dev)
        block_table[i, len(p):] = int(perms[i][-1].item())  # pad with last block

    cache_gpu = build_cache(dev)
    cache_cpu = build_cache("cpu").pin_memory()

    check_gather(cache_gpu, block_table, cu, lens, total, "GPU pool, multi-request")
    check_gather(cache_cpu, block_table, cu, lens, total, "CPU pinned pool, multi-request")

    # single-request regression (must equal the C++ op's verified behavior)
    lens1 = [933]
    cu1 = torch.tensor([0, 933], dtype=torch.int32, device=dev)
    bt1 = torch.tensor([torch.randperm(N_BLOCKS)[:5].tolist()], dtype=torch.int32, device=dev)
    check_gather(cache_gpu, bt1, cu1, lens1, 933, "GPU pool, single-request")

    # ---- [A] C++ op probes (known-issue documentation) ----
    print("\n--- [A] C++ op probes ---")
    kq1 = torch.zeros(933, D, dtype=torch.float8_e4m3fn, device=dev)
    ks1 = torch.zeros(933, 4, dtype=torch.uint8, device=dev)
    cp_gather_indexer_k_quant_cache(cache_gpu, kq1, ks1, bt1, cu1)
    torch.cuda.synchronize()
    ref_k1, _ = gather_reference(cache_gpu.cpu(), bt1.cpu(), lens1)
    ok1 = torch.equal(kq1.view(torch.uint8).cpu(), ref_k1)
    print(f"[A] C++ single-request vs plain reference: {'MATCH' if ok1 else 'MISMATCH'}")
    kqm = torch.zeros(total, D, dtype=torch.float8_e4m3fn, device=dev)
    ksm = torch.zeros(total, 4, dtype=torch.uint8, device=dev)
    cp_gather_indexer_k_quant_cache(cache_gpu, kqm, ksm, block_table, cu)
    torch.cuda.synchronize()
    ref_km, _ = gather_reference(cache_gpu.cpu(), block_table.cpu(), lens)
    n_ok = sum(
        torch.equal(kqm[j].view(torch.uint8).cpu(), ref_km[j]) for j in range(total)
    )
    print(f"[A] C++ multi-request vs plain reference: {n_ok}/{total} rows correct")
    assert n_ok == total, "C++ gather regressed vs writer layout"

    # ---- [C] top_k_per_row_prefill contract ----
    print("\n--- [C] top_k_per_row_prefill contract audit ---")
    from vllm.model_executor.layers.sparse_attn_indexer import ops as idx_ops

    M, N, K = 8, 512, 16
    logits = torch.rand(M, N, dtype=torch.float32, device=dev)
    cu_ks = torch.tensor([0, 0, 0, 0, 256, 256, 256, 256], dtype=torch.int32, device=dev)
    cu_ke = torch.tensor([256] * 4 + [512] * 4, dtype=torch.int32, device=dev)
    out = torch.full((M, K), -1, dtype=torch.int32, device=dev)
    idx_ops.top_k_per_row_prefill(
        logits, cu_ks, cu_ke, out, M, logits.stride(0), logits.stride(1), K
    )
    torch.cuda.synchronize()
    all_ok = True
    for m in range(M):
        lo, hi = int(cu_ks[m]), int(cu_ke[m])
        ref_local = torch.topk(logits[m, lo:hi], K).indices  # ks-relative
        got = out[m].sort().values
        want = ref_local.sort().values
        if not torch.equal(got, want):
            all_ok = False
            print(f"  row {m} (ks={lo}): got={got[:6].tolist()} want={want[:6].tolist()}")
    print(f"[C] window + local-index contract: {'ALL ROWS CORRECT' if all_ok else 'BROKEN'}")
    assert all_ok

    # ---- [D] strided (packed-pool) view regression: root cause of the
    # 2026-09-16 long-context failure. The production cache is a strided
    # view backing.view(-1, block_stride)[:, off:off+page]; the gather must
    # use kv_cache.stride(0), NOT block_size*(D+4).----
    print("\n--- [D] strided packed-pool view (defect #2 regression) ---")
    PAGE, OFF, NB2 = 1039680, 8192, 24
    backing = torch.zeros(NB2 * PAGE, dtype=torch.uint8)
    cache_v = backing.view(-1, PAGE)[:, OFF : OFF + BLOCK * (D + 4)].view(NB2, BLOCK, D + 4)
    blocks = torch.arange(NB2, device="cpu")
    tok_ids = torch.arange(BLOCK)
    flat2 = cache_v  # strided view; python indexing honors strides
    for b in range(NB2):
        page = cache_v[b].reshape(-1)
        vals = ((blocks[b].item() * 57 + tok_ids[:, None] * 3 + torch.arange(D)) % 251).to(torch.uint8)
        page[: BLOCK * D] = vals.reshape(-1)
        page[BLOCK * D :] = 127
    lensD = [700]
    cuD = torch.tensor([0, 700], dtype=torch.int32, device=dev)
    btD = torch.tensor([torch.randperm(NB2)[:3].tolist()], dtype=torch.int32, device=dev)
    assert cache_v.stride(0) == PAGE and PAGE != BLOCK * (D + 4)
    # writer layout is FLAT within the block page: values at p*128,
    # scales at bs*128 + p*4 (NOT the (bs, 132) view's row boxing)
    refD_k = torch.stack([
        cache_v[int(btD[0, t // BLOCK].item())].reshape(-1)[(t % BLOCK) * D : (t % BLOCK) * D + D]
        for t in range(700)
    ])
    kqD = torch.zeros(700, D, dtype=torch.float8_e4m3fn, device=dev)
    ksD = torch.zeros(700, 4, dtype=torch.uint8, device=dev)
    sm89_cp_gather_indexer_k_quant_cache(cache_v.pin_memory(), kqD, ksD, btD, cuD)
    torch.cuda.synchronize()
    okD = torch.equal(kqD.view(torch.uint8).cpu(), refD_k)
    print(f"[D] strided view (stride0={PAGE}): {'MATCH' if okD else 'MISMATCH'}")
    assert okD

    print("\nALL_INDEXER_LAYOUT_TESTS_PASSED")


if __name__ == "__main__":
    main()
