# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: vllm_deepseek sm89 "CPU KV pool" support.
#
# When VLLM_DSV4_KV_CPU_GB=<gb> is set, the whole packed KV-cache pool is
# allocated in pinned host memory and every kernel touches it through CUDA
# UVA zero-copy. This file provides the Triton replacements for the two C++
# pool-writer ops that carry CUDA-only device guards:
#
#   1. sm89_qnorm_rope_kv_fp8_insert  — replaces
#      torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
#      (adapted from the XPU Triton implementation; Q per-head RMSNorm +
#      GPT-J RoPE in place, KV RoPE -> UE8M0 FP8 paged insert).
#   2. cp_gather_indexer_k_quant_cache_triton (already shipped in
#      vllm/v1/attention/ops/rocm_aiter_mla_sparse.py) is used for the
#      prefill indexer K gather.

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


@triton.jit
def _sm89_qnorm_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM] bf16, modified in place
    kv_ptr,  # [num_tokens, HEAD_DIM] bf16 (contiguous)
    kv_out_ptr,  # [num_tokens, HEAD_DIM] bf16 (RoPE-applied kv)
    position_ids_ptr,
    cos_sin_cache_ptr,  # [max_pos, ROPE_DIM] fp32 = cos | sin
    eps,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(
        tl.float32
    )
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    if head_idx < num_heads:
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM
        offs = tl.arange(0, HEAD_DIM)
        q_vals = tl.load(q_base + offs).to(tl.float32)

        sq_sum = tl.sum(q_vals * q_vals, axis=0)
        rms = tl.rsqrt(sq_sum / HEAD_DIM + eps)
        q_vals = q_vals * rms
        nope_mask = offs < NOPE_DIM
        tl.store(q_base + offs, q_vals.to(q_ptr.type.element_ty), mask=nope_mask)

        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms
        new_even = q_even * cos_val - q_odd * sin_val
        new_odd = q_even * sin_val + q_odd * cos_val
        tl.store(q_base + even_offs, new_even.to(q_ptr.type.element_ty))
        tl.store(q_base + odd_offs, new_odd.to(q_ptr.type.element_ty))
    else:
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM
        offs = tl.arange(0, HEAD_DIM)
        kv_full = tl.load(kv_base + offs)
        tl.store(kv_out_base + offs, kv_full)

        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1
        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)
        new_even = kv_even * cos_val - kv_odd * sin_val
        new_odd = kv_even * sin_val + kv_odd * cos_val
        tl.store(kv_out_base + even_offs, new_even.to(kv_out_ptr.type.element_ty))
        tl.store(kv_out_base + odd_offs, new_odd.to(kv_out_ptr.type.element_ty))


def sm89_qnorm_rope_kv_fp8_insert(
    q: torch.Tensor,  # [num_tokens, num_heads, 512] bf16, in-place
    kv: torch.Tensor,  # [num_tokens, 512] bf16
    swa_kv_cache: torch.Tensor,  # uint8 pool view (any shape, flattened 2-D)
    slot_mapping: torch.Tensor,  # [num_tokens] int
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] fp32
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """Triton (UVA-capable) replacement for the C++ fused quant-insert op."""
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    kv = kv.contiguous()
    kv_roped = torch.empty_like(kv)
    num_tokens, num_heads = q.shape[0], q.shape[1]

    _sm89_qnorm_rope_kernel[(num_tokens, num_heads + 1)](
        q,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )

    cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
    quantize_and_insert_k_cache(
        kv_roped,
        cache_2d,
        slot_mapping,
        block_size=block_size,
    )
    return q


@triton.jit
def _sm89_cp_gather_indexer_plain_kernel(
    kv_cache_ptr,  # uint8 [num_blocks, block_size, D+4], PLAIN per-block layout
    k_fp8_ptr,  # uint8 view of dst_k [total, D] fp8e4m3
    k_scale_ptr,  # uint8 [total, 4] (fp32 bytes)
    block_table_ptr,  # int32 [num_reqs, block_table_stride]
    cu_seqlen_ptr,  # int32 [num_reqs + 1]
    token_to_seq_ptr,  # int32 [total]
    block_size,
    block_table_stride,
    block_stride_bytes,  # REAL stride(0) of the cache view (packed pools!)
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Writer-side layout (verified against
    # _fused_kv_compress_norm_rope_insert_indexer_attn): per cache block,
    #   [0, bs*D):        fp8 values, D contiguous bytes per token
    #   [bs*D, bs*(D+4)): fp32 scales, 4 bytes per token
    # The block's page is contiguous, but consecutive BLOCKS are strided by
    # the packed-pool page stride (kv_cache.stride(0)), NOT bs*(D+4).
    tid = tl.program_id(0)
    seq = tl.load(token_to_seq_ptr + tid)
    start = tl.load(cu_seqlen_ptr + seq)
    end = tl.load(cu_seqlen_ptr + seq + 1)
    if (tid < start) or (tid >= end):
        return

    tok = tid - start
    block_idx = tok // block_size
    pos_in_block = tok % block_size
    page = tl.load(block_table_ptr + seq * block_table_stride + block_idx)
    # The pool can exceed 4 GiB (CPU-KV-pool mode): keep 64-bit addressing.
    page_base = page.to(tl.int64) * block_stride_bytes

    d = tl.arange(0, BLOCK_D)
    vals = tl.load(
        kv_cache_ptr + page_base + pos_in_block * HEAD_DIM + d,
        mask=d < HEAD_DIM,
        other=0,
    )
    tl.store(k_fp8_ptr + tid.to(tl.int64) * HEAD_DIM + d, vals, mask=d < HEAD_DIM)

    sb = tl.arange(0, 4)
    scale = tl.load(
        kv_cache_ptr + page_base + block_size * HEAD_DIM + pos_in_block * 4 + sb
    )
    tl.store(k_scale_ptr + tid.to(tl.int64) * 4 + sb, scale)


@triton.jit
def _sm89_cp_gather_indexer_plain_kernel_cv(
    kv_cache_ptr,
    k_fp8_ptr,
    k_scale_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    token_to_seq_ptr,
    block_size,
    block_table_stride,
    block_stride_bytes,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tid = tl.program_id(0)
    seq = tl.load(token_to_seq_ptr + tid)
    start = tl.load(cu_seqlen_ptr + seq)
    end = tl.load(cu_seqlen_ptr + seq + 1)
    if (tid < start) or (tid >= end):
        return
    tok = tid - start
    block_idx = tok // block_size
    pos_in_block = tok % block_size
    page = tl.load(block_table_ptr + seq * block_table_stride + block_idx, cache_modifier=".cv")
    page_base = page.to(tl.int64) * block_stride_bytes
    d = tl.arange(0, BLOCK_D)
    vals = tl.load(
        kv_cache_ptr + page_base + pos_in_block * HEAD_DIM + d,
        mask=d < HEAD_DIM, other=0, cache_modifier=".cv",
    )
    tl.store(k_fp8_ptr + tid.to(tl.int64) * HEAD_DIM + d, vals, mask=d < HEAD_DIM)
    sb = tl.arange(0, 4)
    scale = tl.load(
        kv_cache_ptr + page_base + block_size * HEAD_DIM + pos_in_block * 4 + sb,
        cache_modifier=".cv",
    )
    tl.store(k_scale_ptr + tid.to(tl.int64) * 4 + sb, scale)


@triton.jit
def _bt_probe_kernel(bt_ptr, out_ptr, stride):
    i = tl.arange(0, 8)
    v = tl.load(bt_ptr + i)
    tl.store(out_ptr + i, v)


def _bt_probe(block_table):
    out = torch.full((8,), -7, dtype=torch.int32, device=block_table.device)
    _bt_probe_kernel[(1,)](block_table, out, block_table.stride(0))
    return out


def sm89_cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Triton (UVA-capable) indexer K gather for the PLAIN per-block layout.

    Replaces both the C++ ``cp_gather_indexer_k_quant_cache`` (whose
    multi-request-chunk behavior could not be validated on sm89) and the
    ROCm ``cp_gather_indexer_k_quant_cache_triton`` (which reads a
    16x16-tile SHUFFLE layout that the CUDA/Triton compressor store does
    not write). Works on pinned host pools (UVA) and GPU tensors alike.
    """
    total = int(dst_k.shape[0])
    num_blocks, block_size, width = kv_cache.shape
    assert width == dst_k.shape[1] + 4, "unexpected indexer cache row width"
    device = block_table.device
    token_to_seq = (
        torch.searchsorted(
            cu_seq_lens, torch.arange(total, device=device), right=True
        )
        - 1
    ).to(torch.int32)
    dst_k_u8 = dst_k.view(torch.uint8)
    _sm89_cp_gather_indexer_plain_kernel[(total,)](
        kv_cache,
        dst_k_u8,
        dst_scale,
        block_table,
        cu_seq_lens.to(torch.int32),
        token_to_seq,
        block_size,
        block_table.stride(0),
        kv_cache.stride(0),
        HEAD_DIM=dst_k.shape[1],
        BLOCK_D=triton.next_power_of_2(dst_k.shape[1]),
        num_warps=1,
    )
    _forensic_on_gather(kv_cache, block_table, cu_seq_lens, total, dst_k, dst_scale)


# ---------------------------------------------------------------------------
# Cross-step visibility forensics (VLLM_DSV4_FORENSIC=1).
# All in-forward actions are capture-safe GPU-side clones (no host sync).
# Records are flushed to /tmp at the NEXT metadata build, when the step's
# work has drained; pool probes are read at flush time.
# ---------------------------------------------------------------------------
_FORENSIC_RECORDS: list = []
_FORENSIC_STATE: dict = {}


def _forensic_on_gather(pool, block_table, cu_seq_lens, total, dst_k, dst_scale):
    import os

    if not os.environ.get("VLLM_DSV4_FORENSIC", ""):
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
        _FORENSIC_STATE["n"] = _FORENSIC_STATE.get("n", 0) + 1
        if _FORENSIC_STATE["n"] > 400:
            return
        bt = block_table.detach().clone()
        _FORENSIC_RECORDS.append(
            dict(
                kind="gather",
                idx=_FORENSIC_STATE["n"],
                bt=bt,
                cu=cu_seq_lens.detach().clone(),
                pool=pool,
                pool_shape=tuple(pool.shape),
                pool_stride=tuple(pool.stride()),
                dst_head=dst_k.detach().view(torch.uint8)[:8].clone(),
                dst_scale_head=dst_scale.detach().view(torch.uint8)[:8].clone(),
                dst_mid=(
                    dst_k.detach().view(torch.uint8)[630:640].clone()
                    if dst_k.shape[0] > 640
                    else None
                ),
                dst_tail=(
                    dst_k.detach().view(torch.uint8)[-8:].clone()
                    if dst_k.shape[0] > 16
                    else None
                ),
                cv_head=_forensic_cv_gather(
                    pool, block_table, cu_seq_lens, total, dst_k, dst_scale
                ),
                host_probe=_forensic_host_probe(pool),
                bt_pre=_bt_probe(block_table).clone(),
                bt_post=_bt_probe(block_table).clone(),
            )
        )
    except Exception:
        pass


def _forensic_cv_gather(pool, block_table, cu_seq_lens, total, dst_k, dst_scale):
    """Relaunch the gather with volatile (.cv) loads into a scratch buffer."""
    try:
        num_blocks, block_size, width = pool.shape
        device = block_table.device
        t2s = (
            torch.searchsorted(
                cu_seq_lens, torch.arange(total, device=device), right=True
            )
            - 1
        ).to(torch.int32)
        cv_k = torch.zeros_like(dst_k)
        cv_s = torch.zeros_like(dst_scale)
        _sm89_cp_gather_indexer_plain_kernel_cv[(total,)](
            pool,
            cv_k.view(torch.uint8),
            cv_s,
            block_table,
            cu_seq_lens.to(torch.int32),
            t2s,
            block_size,
            block_table.stride(0),
            pool.stride(0),
            HEAD_DIM=dst_k.shape[1],
            BLOCK_D=triton.next_power_of_2(dst_k.shape[1]),
            num_warps=1,
        )
        return cv_k.detach().view(torch.uint8)[:8].clone()
    except Exception:
        return None


def _forensic_host_probe(pool):
    """Host-side read of raw pool block1/block2 rows 0-7 (blocks known to
    hold compressed groups 0-15 for our test workload)."""
    try:
        flat = pool.view(-1, pool.shape[1] * pool.shape[2]).view(torch.uint8)
        w = pool.shape[1] * pool.shape[2]
        return dict(
            blk1_rows0_7_nz=int((flat[w : w + 132 * 8] != 0).sum()),
            blk2_rows0_7_nz=int((flat[2 * w : 2 * w + 132 * 8] != 0).sum()),
        )
    except Exception:
        return None


def _forensic_flush(extra: dict | None = None):
    """Called from metadata build (host-side, between steps)."""
    import os

    if not os.environ.get("VLLM_DSV4_FORENSIC", ""):
        return
    try:
        if extra is not None:
            _FORENSIC_RECORDS.append(dict(kind="build", **extra))
        if not _FORENSIC_RECORDS:
            return
        n = _FORENSIC_STATE.setdefault("flush", 0) + 1
        _FORENSIC_STATE["flush"] = n
        if n > 60:
            _FORENSIC_RECORDS.clear()
            return
        out = []
        pool_probe_done = set()
        for r in _FORENSIC_RECORDS:
            o = dict(kind=r["kind"], idx=r.get("idx"))
            if r["kind"] == "gather":
                bt = r["bt"].cpu()
                o["bt_row0_head"] = bt[0, :16].tolist()
                o["bt_full_rows"] = bt[:2].tolist()
                o["dst_head_nz"] = int((r["dst_head"] != 0).sum())
                o["dst_scale_head_f32"] = (
                    r["dst_scale_head"].view(torch.float32).tolist()
                )
                o["dst_mid_nz"] = (
                    int((r["dst_mid"] != 0).sum()) if r["dst_mid"] is not None else -1
                )
                o["dst_tail_nz"] = (
                    int((r["dst_tail"] != 0).sum()) if r["dst_tail"] is not None else -1
                )
                o["cv_head_nz"] = (
                    int((r["cv_head"] != 0).sum()) if r["cv_head"] is not None else -1
                )
                o["host_probe"] = r["host_probe"]
                o["bt_pre"] = r["bt_pre"].cpu().tolist()
                o["bt_post"] = r["bt_post"].cpu().tolist()
                o["cu"] = r["cu"].cpu().tolist()
                o["pool_shape"] = r["pool_shape"]
                o["pool_stride"] = r["pool_stride"]
                pid = id(r["pool"])
                if pid not in pool_probe_done:
                    pool_probe_done.add(pid)
                    pool = r["pool"]
                    flat = pool.reshape(-1)
                    # probe the first 5 blocks referenced by bt row0 + raw block0/1
                    probes = {}
                    blocks = [b for b in bt[0, :5].tolist() if b >= 0][:5] + [0, 1]
                    for b in dict.fromkeys(blocks):
                        base = b * (r["pool_shape"][1] * r["pool_shape"][2])
                        probes[f"blk{b}"] = (
                            flat[base : base + 132].clone().tolist()
                        )
                    o["pool_probe"] = probes
            else:
                o.update(
                    {k: (v.cpu().tolist() if torch.is_tensor(v) else v) for k, v in r.items() if k != "kind"}
                )
            out.append(o)
        torch.save(out, f"/tmp/forensic_step_{n:03d}.pt")
        _FORENSIC_RECORDS.clear()
    except Exception as e:  # noqa: BLE001
        logger.warning("forensic flush failed: %s", e)


def sm89_kv_pool_in_cpu() -> bool:
    import os

    return bool(os.environ.get("VLLM_DSV4_KV_CPU_GB", ""))
