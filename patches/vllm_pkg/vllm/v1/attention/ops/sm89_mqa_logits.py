# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: sm89 (RTX 6000 Ada / Ada Generation) port of the
# DeepSeek-V4 "lightning indexer" MQA-logits kernels for vLLM.
#
# DeepGEMM (the reference CUDA backend for these ops) only supports
# SM90/SM100. On sm89 we replace the two entry points
# ``fp8_fp4_mqa_logits`` (contiguous, prefill) and
# ``fp8_fp4_paged_mqa_logits`` (paged, decode) with the Triton kernels
# below. The math mirrors the torch reference implementations shipped in
# ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py``:
#
#   logit[m, j] = k_scale[j] * sum_h weights[m, h] * relu(dot(q[m, h], k[j]))
#
# with q/k stored as float8_e4m3fn and k_scale as float32. Only the FP8
# indexer-cache path is supported (the FP4 indexer cache stays disabled via
# ``use_fp4_indexer_cache=false`` on sm89).

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _fp8_mqa_logits_sm89_kernel(
    Q_ptr,  # fp8e4m3 [M, H, D]
    KV_ptr,  # fp8e4m3 [N, D]
    kv_scales_ptr,  # fp32 [N]
    weights_ptr,  # fp32 [M, H]
    cu_start_ptr,  # int32 [M]
    cu_end_ptr,  # int32 [M]
    logits_ptr,  # fp32 [M, N] (pre-filled with -inf)
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_s: tl.int64,
    stride_q_h: tl.int64,
    stride_q_d: tl.int64,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.int64,
    stride_w_s: tl.int64,
    stride_w_h: tl.int64,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    q_ptrs = (
        Q_ptr
        + row_id * stride_q_s
        + h_inds * stride_q_h
        + d_inds[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs).to(tl.float32)
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs)

    start_ind = tl.maximum(tl.load(cu_start_ptr + row_id), 0)
    end_ind = tl.minimum(tl.load(cu_end_ptr + row_id), seq_len_kv)
    num = end_ind - start_ind

    kv_col_offsets = start_ind + tl.arange(0, BLOCK_KV)
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )
    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets
    logits_ptrs = (
        logits_ptr + row_id * stride_logits_s + kv_col_offsets * stride_logits_k
    )

    for _ in tl.range(0, num, BLOCK_KV):
        in_win = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
        kv_block = tl.load(kv_ptrs, mask=in_win[None, :], other=0.0).to(tl.float32)
        kv_scales = tl.load(kv_scales_ptrs, mask=in_win, other=0.0)

        # [NUM_HEADS, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="tf32")
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores, mask=in_win)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV


def fp8_mqa_logits_sm89(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Contiguous FP8 MQA logits (prefill path).

    Matches the DeepGEMM/aiter ``fp8_mqa_logits`` semantics: returns
    ``[M, N]`` float32 logits with ``-inf`` outside ``[cu_starts, cu_ends)``.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    kv_scales_1d = kv_scales.reshape(-1)

    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if seq_len == 0 or seq_len_kv == 0:
        return logits

    block_kv = 64
    _fp8_mqa_logits_sm89_kernel[(seq_len,)](
        q,
        k_fp8,
        kv_scales_1d,
        weights,
        cu_starts.to(torch.int32),
        cu_ends.to(torch.int32),
        logits,
        seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_s=k_fp8.stride(0),
        stride_kv_d=k_fp8.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=2,
    )
    return logits


@triton.jit
def _fp8_paged_mqa_logits_sm89_kernel(
    Q_ptr,  # fp8e4m3 [rows, H, D] (rows = B*next_n)
    kv_cache_ptr,  # uint8 [num_blocks, block_size, 1, D+4]
    weights_ptr,  # fp32 [rows, H]
    context_lens_ptr,  # int32 [rows] effective context per row
    block_tables_ptr,  # int32 [rows, max_blocks]
    logits_ptr,  # fp32 [rows, max_model_len] (pre-filled -inf)
    max_model_len,
    max_blocks,
    block_stride_bytes,  # REAL kv_cache.stride(0); packed pools are strided
    HEAD_SIZE: tl.constexpr,  # D
    NUM_HEADS: tl.constexpr,  # H
    BLOCK_SIZE: tl.constexpr,  # cache block_size
    BLOCK_KV: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
):
    row = tl.program_id(0)  # b * next_n + n
    tile = tl.program_id(1)

    ctx = tl.load(context_lens_ptr + row)
    start = tile * TILES_PER_PROG * BLOCK_KV
    if start >= ctx:
        return

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    q_stride_row = NUM_HEADS * HEAD_SIZE
    q_ptrs = (
        Q_ptr + row.to(tl.int64) * q_stride_row + h_inds * HEAD_SIZE + d_inds[None, :]
    )
    q_block = tl.load(q_ptrs).to(tl.float32)
    # [H, 1] so the per-head weight broadcasts along the KV-tile axis.
    w_block = tl.load(
        weights_ptr + row * NUM_HEADS + tl.arange(0, NUM_HEADS)[:, None]
    )

    end = tl.minimum(start + TILES_PER_PROG * BLOCK_KV, ctx)

    kv_pos = start + tl.arange(0, BLOCK_KV)
    scale_f32_ptr = kv_cache_ptr.to(tl.pointer_type(tl.float32))
    for _ in tl.range(start, end, BLOCK_KV):
        in_win = kv_pos < end
        block_idx = kv_pos // BLOCK_SIZE
        page = tl.load(
            block_tables_ptr + row.to(tl.int64) * max_blocks + block_idx,
            mask=in_win,
            other=0,
        )
        # Indexer cache block layout (matches the Triton compressor store
        # `_fused_kv_compress_norm_rope_insert_indexer_attn` and the _C
        # gather `cp_gather_indexer_k_quant_cache`):
        #   [0, bs*D):            fp8 values, D bytes per token
        #   [bs*D, bs*(D+4)):     fp32 scales, 4 bytes per token
        page_base = page.to(tl.int64) * block_stride_bytes
        pos_in_block = kv_pos % BLOCK_SIZE
        kv_block = tl.load(
            kv_cache_ptr
            + page_base[:, None]
            + pos_in_block[:, None] * HEAD_SIZE
            + d_inds[None, :],
            mask=in_win[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        kv_scales = tl.load(
            (kv_cache_ptr + page_base + BLOCK_SIZE * HEAD_SIZE + pos_in_block * 4).to(
                tl.pointer_type(tl.float32)
            ),
            mask=in_win,
            other=0.0,
        )

        scores = tl.dot(q_block, tl.trans(kv_block), input_precision="tf32")
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(
            logits_ptr + row.to(tl.int64) * max_model_len + kv_pos,
            scores,
            mask=in_win,
        )

        kv_pos += BLOCK_KV


def fp8_paged_mqa_logits_sm89(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Paged FP8 MQA logits (decode path).

    kv_cache layout matches DeepGEMM/aiter: ``[num_blocks, block_size, 1,
    D+4]`` uint8 with the trailing 4 bytes holding the float32 dequant
    scale. Returns ``[B*next_n, max_model_len]`` float32 logits, ``-inf``
    outside each row's causal context.
    """
    batch_size, next_n, num_heads, head_size = q.shape
    block_size = kv_cache.shape[1]
    rows = batch_size * next_n

    logits = torch.full(
        (rows, max_model_len),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if rows == 0:
        return logits

    if context_lens.dim() > 1:
        context_lens = context_lens.reshape(-1)
    context_lens = context_lens.to(torch.int32).contiguous()

    q_flat = q.reshape(rows, num_heads, head_size).contiguous()
    # block_tables is [B, max_blocks]; repeat each batch row next_n times so
    # the kernel can index with `row // next_n`'s table directly.
    bt = (
        block_tables.unsqueeze(1)
        .expand(batch_size, next_n, block_tables.shape[1])
        .reshape(rows, -1)
        .contiguous()
    )

    # No host sync here (must stay CUDA-graph capturable): size the grid by
    # max_model_len and let each program early-return past its runtime ctx.
    block_kv = 64
    tiles_per_prog = 4
    grid = (rows, triton.cdiv(max_model_len, block_kv * tiles_per_prog))
    _fp8_paged_mqa_logits_sm89_kernel[grid](
        q_flat,
        kv_cache,
        weights,
        context_lens,
        bt,
        logits,
        max_model_len,
        bt.shape[1],
        kv_cache.stride(0),
        HEAD_SIZE=head_size,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        BLOCK_KV=block_kv,
        TILES_PER_PROG=tiles_per_prog,
        num_warps=4,
        num_stages=2,
    )
    return logits
