#!/bin/bash
# DeepSeek-V4-Flash-0731 · sm89 (RTX 6000 Ada) vLLM 启动脚本 —— 速度型（推荐）
# 端口: 8000   模型名: Deepseek   conda env: vllm_deepseek
# 4 卡张量并行；上下文 131072；CUDA graph 开启
#   单流 ~70 tok/s，8 并发聚合 ~390 tok/s（短-中上下文 decode）
# 长文型（1M 上下文 + CPU KV 层）请用 start_1m.sh
# 用法: bash start.sh   （建议在 tmux 里运行: tmux new -s dsv4 'bash start.sh'）
set -e

CONDA_ENV_NAME="${VLLM_ENV:-vllm_deepseek}"
CONDA_ROOT="${CONDA_ROOT:-/home/$(id -un)/anaconda3}"

if [ -f "$CONDA_ROOT/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"
fi
command -v vllm >/dev/null 2>&1 || { echo "✗ vllm 未找到（需要 conda env: $CONDA_ENV_NAME）"; exit 1; }

# tilelang JIT 需要 sm89-capable nvcc（系统 /usr/bin/nvcc 是 CUDA 11.5，不认识 sm_89）
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CONDA_PREFIX/bin:$PATH"

MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ] || [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "✗ 未设置 MODEL_DIR 或目录中无 config.json（不是有效的权重目录）"
    echo "  用法: MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731 bash deploy/start.sh"
    exit 1
fi
exec vllm serve "${MODEL_DIR}" --served-model-name Deepseek \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.96 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enable-prefix-caching
