#!/bin/bash
# 将 sm89 补丁重新应用到 vllm_deepseek 环境（幂等，可重复执行）。
# 用法: bash apply_sm89_patches.sh [/path/to/vllm_deepseek_env]
#       默认 env: /home/$(id -un)/anaconda3/envs/vllm_deepseek
set -e
ENV_DIR="${1:-/home/$(id -un)/anaconda3/envs/vllm_deepseek}"
SP="$ENV_DIR/lib/python3.12/site-packages"
PD="$(cd "$(dirname "$0")" && pwd)"

[ -d "$SP/vllm" ] || { echo "✗ 找不到 $SP/vllm（确认 env 路径）"; exit 1; }

echo "→ 复制 vllm 包内补丁文件（含新增的 sm89 内核）..."
cp -r "$PD/vllm_pkg/." "$SP/"

echo "→ 复制 cccl 头文件补丁（nvcc 13.3 + CTK 13.0 头混装，屏蔽版本检查）..."
cp "$PD/site_headers/nvidia_cu13_cccl/cuda_toolkit.h" \
   "$SP/nvidia/cu13/include/cccl/cuda/std/__cccl/cuda_toolkit.h"
cp "$PD/site_headers/flashinfer_cccl/cuda_toolkit.h" \
   "$SP/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h"

echo "→ 确保 env/bin 里有 nvcc（tilelang JIT 需要 sm89 编译器）..."
NVCC="$SP/nvidia/cu13/bin/nvcc"
[ -x "$NVCC" ] || { echo "✗ 缺少 $NVCC（该 env 的 torch cu13 依赖包应自带）"; exit 1; }
ln -sf "$NVCC" "$ENV_DIR/bin/nvcc"

echo "✓ 补丁应用完成。清理 tilelang JIT 缓存后重启服务即可生效："
echo "  rm -rf ~/.tilelang/cache  # 如遇缓存问题"
