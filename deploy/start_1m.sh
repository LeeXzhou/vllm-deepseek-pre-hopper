#!/bin/bash
# DeepSeek-V4-Flash-0731 · sm89 长文型启动脚本（1M 上下文 · CPU-KV-pool 架构 v3）
#
# 架构：权重全驻显存（无 PCIe 流式搬运）+ 整个 KV 池放 pinned 内存
#       （14GB/卡，内核经 CUDA UVA 零拷贝读写；选中行仅 ~15MB/步过 PCIe）。
# 实测（4×RTX 6000 Ada, TP=4, 2026-09-14）：
#   decode   单流 ~67 tok/s；16 并发聚合 ~444 tok/s（8 并发 309，旧 offload 架构为 6.3/27.5）
#   prefill  端到端 2500-3800 tok/s（10k 提示 3.9s；188k 提示 76s；旧架构 145-330）
#   容量     KV 池 14GB/卡 → 2.05×1M 同时活跃；16 路并发受理
# 用法: tmux new -s dsv4 'bash start_1m.sh'
set -e

CONDA_ENV_NAME="${VLLM_ENV:-vllm_deepseek}"
CONDA_ROOT="${CONDA_ROOT:-/home/$(id -un)/anaconda3}"

if [ -f "$CONDA_ROOT/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
fi
command -v vllm >/dev/null 2>&1 || { echo "✗ vllm 未找到（需要 conda env: $CONDA_ENV_NAME）"; exit 1; }

export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CONDA_PREFIX/bin:$PATH"

# CPU-KV-pool 模式核心开关（池大小 GB/卡；4 卡共 56GB pinned）
export VLLM_DSV4_KV_CPU_GB=14
# 权重几乎占满显存，图内存预估预留关闭（真实捕获按需进行）
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ] || [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "✗ 未设置 MODEL_DIR 或目录中无 config.json（不是有效的权重目录）"
    echo "  用法: MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731 bash deploy/start_1m.sh"
    exit 1
fi
exec vllm serve "${MODEL_DIR}" --served-model-name Deepseek \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 1048576 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enable-prefix-caching
