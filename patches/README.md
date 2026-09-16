# sm_89 补丁清单

本目录是对 **vLLM 0.26.0**（pip 安装，py3.12）的 **overlay 补丁**：只包含新增/修改的文件，由 `apply_sm89_patches.sh` 幂等地复制进 site-packages。采用全文件覆盖而非 diff，是因为改动横跨大文件多处且需新增整个内核文件；带 `.orig` 的条目保留了 stock 版本供审计与回滚。

> ⚠️ overlay 与 vLLM 版本强绑定（基于 0.26.0）。换版本请勿直接覆盖，需手工迁移——所有改动点均带 `vllm_deepseek sm89` 或 `VLLM_DSV4` 标记，`grep -rn "sm89\|VLLM_DSV4" vllm/` 可全部定位。

## 新增文件

| 文件 | 内容 |
|---|---|
| `vllm_pkg/vllm/v1/attention/ops/sm89_mqa_logits.py` | indexer MQA-logits 两个 Triton 核：`fp8_mqa_logits_sm89`（连续，prefill）与 `fp8_paged_mqa_logits_sm89`（分页，decode；逐块布局、`kv_cache.stride(0)` 跨步寻址、网格按 `max_model_len` 静态展开以兼容 CUDA graph，无 host 同步） |
| `vllm_pkg/vllm/models/deepseek_v4/nvidia/sm89.py` | `DeepseekV4Sm89MLAAttention`：稀疏 MLA 注意力层。decode/prefill 复用 ROCm 可移植 Triton ragged 核，o_proj 用 bf16 逆 RoPE + 缓存 bf16 einsum 替代 DeepGEMM `fp8_einsum` |
| `vllm_pkg/vllm/models/deepseek_v4/nvidia/sm89_cpu_pool.py` | CPU-KV-pool 支撑：Triton SWA 融合插入核（Q-RMSNorm + RoPE + FP8 量化 + 分页插入，可写 pinned 池）、朴素布局 indexer K gather 核（int64 块基址，池可超 4 GiB）、`sm89_kv_pool_in_cpu()` 判定、调试插桩（`VLLM_DSV4_FORENSIC`，默认关闭） |

## 修改文件（附 .orig）

| 文件 | 改动 |
|---|---|
| `vllm_pkg/vllm/utils/deep_gemm.py` | `fp8_fp4_mqa_logits` 增加 pre-Hopper CUDA 路径：major<9 且 `is_deep_gemm_supported()` 为假时路由到 sm89 Triton 核（仅 FP8 indexer 缓存，q_scale 须为 None） |
| `vllm_pkg/vllm/models/deepseek_v4/sparse_mla.py` | `supports_compute_capability` 放行 major=8（元数据构建器为 Triton/CPU 实现，计算内核由 sm89 注意力类提供） |
| `vllm_pkg/vllm/models/deepseek_v4/nvidia/model.py` | ① major<9 时选择 `DeepseekV4Sm89MLAAttention`；② `hc_attn_fn_broadcast` 支持惰性计算；③ CPU offload 包装层解包 |
| `vllm_pkg/vllm/model_executor/layers/sparse_attn_indexer.py` | prefill indexer K gather 统一走朴素布局 Triton 核（CPU pinned 池与 GPU 池通用；C++ 算子带 CUDA 设备守卫不能写 pinned）。`VLLM_DSV4_GATHER_CPP=1` 可回退 C++ 算子 |

## 修改文件（全文件覆盖，无 .orig）

| 文件 | 改动 |
|---|---|
| `vllm_pkg/vllm/v1/attention/backends/mla/indexer.py` | ① decode MQA logits 架构门改用 `is_deep_gemm_supported()`（`has_deep_gemm()` 只证明可 import，内核运行期才检查架构）；② CPU-KV-pool 模式预填充缓冲 40×→2×；③ 调试 flush 挂钩（默认关） |
| `vllm_pkg/vllm/v1/core/kv_cache_utils.py` | `VLLM_DSV4_KV_CPU_GB` 模式：KV 容量按主机 RAM 而非空闲显存规划 |
| `vllm_pkg/vllm/v1/worker/gpu_model_runner.py` | `VLLM_DSV4_KV_CPU_GB` 模式：KV 池以 pinned memory 分配（内核经 UVA 零拷贝访问） |
| `vllm_pkg/vllm/models/deepseek_v4/compressor.py` | CPU-KV-pool 模式下 head-512 压缩存储从 CuTe-DSL 切到 Triton（CuTe-DSL 不能写 pinned） |
| `vllm_pkg/vllm/models/deepseek_v4/attention.py` | SWA 池写入在 CPU 池时走 Triton（UVA）替换 C++ 融合算子 |
| `vllm_pkg/vllm/model_executor/kernels/mhc/tilelang.py` | MHC broadcast 变体的 DeepGEMM GEMM 加架构门，回退 tilelang GEMM（数学等价） |
| `vllm_pkg/vllm/model_executor/kernels/linear/scaled_mm/triton.py` | UE8M0 权重尺度加载后升 fp32（该 Triton launcher 无 e8m0 指针类型；e8m0 为 2 的幂，位精确） |
| `vllm_pkg/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | ① kv / main_cache 允许为 pinned 主机内存（UVA）；② pre-Hopper CUDA 走 split path |

## 环境级补丁（site_headers + nvcc）

| 文件 | 内容 |
|---|---|
| `site_headers/flashinfer_cccl/cuda_toolkit.h`（.orig 有效） | 屏蔽 CCCL 的 CTK 版本兼容检查（pip 混装 nvcc 13.3 与 CTK 13.0 头文件会触发 `#error`，阻断 TileLang/FlashInfer JIT） |
| `site_headers/nvidia_cu13_cccl/cuda_toolkit.h` | 同上（此份 `.orig` 与补丁版相同，系补丁后抓取，无 diff 价值） |

`apply_sm89_patches.sh` 另会把 env 内 torch cu13 依赖自带的 nvcc 软链到 `env/bin/nvcc`（系统 nvcc 可能过旧，不认识 sm_89，TileLang JIT 需要）。`CUDA_HOME` 由启动脚本设置。

## 应用与回滚

```bash
# 应用（幂等）
bash apply_sm89_patches.sh ~/anaconda3/envs/vllm_deepseek

# 审计
diff vllm_pkg/vllm/utils/deep_gemm.py.orig vllm_pkg/vllm/utils/deep_gemm.py

# 回滚单个文件：用 .orig 覆盖；无 .orig 的文件重装 vllm==0.26.0 后按上表迁移
```

全部改动位于专用 conda 环境内，不影响其他 vLLM 安装。
