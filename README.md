# DeepSeek-V4-Flash · sm_89 vLLM Port（1M 上下文）

vLLM 0.26.0 的 overlay 补丁集：让 **DeepSeek-V4-Flash** 运行在 **pre-Hopper NVIDIA GPU**（实测：4 × RTX 6000 Ada，sm_89）上，并支持 **1M 上下文**。

官方 CUDA 路径依赖的 DeepGEMM / FlashMLA 仅支持 SM90 (Hopper) / SM100 (Blackwell)，在更早架构上直接断言 `Unsupported architecture`。本仓库用 **Triton / TileLang / Marlin** 替换全部 SM90+ 算子依赖，并提供 **CPU-KV-pool** 内存架构（权重驻显存 + KV 池驻 pinned 主机内存，CUDA UVA 零拷贝）以在 48 GB 卡上容纳 1M 上下文。

## 性能（4 × RTX 6000 Ada，TP=4，1M 生产配置实测）

| 场景 | 数值 |
|---|---|
| decode 单流（短上下文） | **67.4 tok/s** |
| decode 随上下文衰减（实测 2026-10-23） | 184k=50.9 / 355k=42.6 / 581k=23.3 / **1M=15.6 tok/s**（打分扫描 ~0.71KB×N 过 UVA，有效带宽 ~13GB/s；`bench_decode_longctx.py`） |
| decode 8 / 16 并发聚合 | **323 / 444 tok/s** |
| prefill 10k / 60k / 188k 提示 | **3.9 s / 14.9 s / 75.6 s**（2491 tok/s） |
| 1M prefill（外推） | ~6–7 分钟 |
| KV 容量 | **2.05 × 1M 并发活跃**（14 GB/卡 pinned），16 路受理、超额排队 |
| 大海捞针（5k–128k × 多深度 + 双针交叉） | 全绿 |
| 内核数值对齐 | gather 精确一致；GEMM/注意力在 fp8/bf16 舍入量级（见 tests/） |

## 原理

### 1. 算子替换（SM90+ → 跨架构）

| # | 官方算子 | 本移植 | 位置（相对 `patches/vllm_pkg/`） |
|---|---|---|---|
| 1 | DeepGEMM `fp8_fp4_mqa_logits`（indexer prefill 打分） | 新写 Triton 核 | `vllm/v1/attention/ops/sm89_mqa_logits.py` |
| 2 | DeepGEMM `fp8_fp4_paged_mqa_logits`（indexer decode 打分） | 新写 Triton 核（分页、逐块布局、按 `kv_cache.stride(0)` 跨步寻址、CUDA graph 兼容） | 同上 |
| 3 | FlashMLA `flash_mla_with_kvcache`（稀疏 MLA decode） | vLLM ROCm 可移植 Triton ragged 核 | `vllm/models/deepseek_v4/nvidia/sm89.py` |
| 4 | FlashMLA `flash_mla_sparse_fwd`（稀疏 MLA prefill） | 同上（gather 后走 Triton ragged 核） | 同上 |
| 5 | DeepGEMM `fp8_einsum`（o_proj 分组投影） | bf16 逆 RoPE Triton 核 + 反量化缓存的 bf16 einsum | 同上 |
| 6 | DeepGEMM MegaMoE | vLLM Marlin MXFP4 MoE（自动选择） | 启动参数 |
| 7 | 稠密 FP8 块线性层 | vLLM Triton `w8a8` block-scaled GEMM；UE8M0 尺度升 fp32（位精确） | `vllm/model_executor/kernels/linear/scaled_mm/triton.py` |
| 8 | MHC 核中 DeepGEMM broadcast 变体 | 架构门回退 tilelang GEMM（数学等价） | `vllm/model_executor/kernels/mhc/tilelang.py` |
| 9 | 架构门（`has_deep_gemm` 等） | `is_deep_gemm_supported()`（含架构检查）；注意力后端放行 major=8 | 多处，见 `patches/README.md` |

选路按**架构代次**判断（`compute_capability.major < 9` → Triton 后端），不针对具体 GPU 型号；所有内核均为 Triton / TileLang / Marlin，无 sm_89 专属指令。

### 2. CPU-KV-pool（1M 上下文内存架构）

显存账（TP=4）：每卡权重 40.8 GB、可用 43.8 GB，每 1M token 的 KV ≈ 5.2 GB/卡——KV 池无法放进显存。

关键在访问模式的不对称：

| 数据 | 每步 decode 访问量 | 合适的位置 |
|---|---|---|
| 激活权重 | ~11.5 GB 合计 ≈ 2.9 GB/卡，每步必读 | **HBM**（~800 GB/s；过 PCIe 需 ~145 ms/步，不可行） |
| KV 注意力项（top-512 × 43 层） | **~13 MB/步**（与上下文长度无关） | **pinned 主机内存 + UVA 零拷贝**（≈0.7 ms/步） |
| KV 打分项（选 top-512 须扫全量压缩候选） | **~0.71 KB × N** | 同上；1M 时 ~0.75 GB/步 → 长上下文 decode 随 N 线性衰减（实测 581k=23.3、1M=15.6 tok/s，见下） |

据此，`VLLM_DSV4_KV_CPU_GB=14` 开启的内存布局为：**权重全驻显存 + 整个 KV 池放 pinned 主机内存（14 GB/卡，4 卡共 56 GB）**，内核经 CUDA UVA 直接读写主机池（对 Triton 内核是普通指针，无 staging、无显式搬运）。实现组成：

1. 池分配（`kv_cache_utils.py` / `gpu_model_runner.py`）：容量按主机 RAM 规划、pinned 分配；
2. 两个原 C++ 池写入算子（带 CUDA 设备守卫）的 Triton/UVA 替代：SWA 融合插入核、indexer K gather 核（`sm89_cpu_pool.py`）；
3. head-512 压缩存储从 CuTe-DSL 切到 Triton（CuTe-DSL 不能写 pinned）；
4. prefill 显存瘦身：indexer logits 按查询行分块（128 MB/块）、预填充缓冲 40×→2×、注意力 gather 按单请求分块。

## 兼容性

- **实测**：sm_89（RTX 6000 Ada）；
- **设计范围**：pre-Hopper CUDA 全系（Ampere sm_80/86 理论可用，Marlin 需 sm_80+，未实测，欢迎反馈）；
- CPU-KV-pool 与 GPU 型号无关，任何支持 UVA 的 CUDA 卡可用；
- 未启用 FP4 indexer 缓存路径（`use_fp4_indexer_cache=false`，仅 FP8 路径）。

## 快速开始

```bash
# 1) 环境（python 3.12；torch 2.11.0+cu130 由 vllm 依赖带入）
conda create -n vllm_deepseek python=3.12 -y
conda activate vllm_deepseek
pip install vllm==0.26.0

# 2) 应用补丁（幂等）
bash patches/apply_sm89_patches.sh ~/anaconda3/envs/vllm_deepseek

# 3) 启动（权重目录经 MODEL_DIR 指定）
MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731 bash deploy/start_1m.sh   # 1M 上下文
MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731 bash deploy/start.sh      # 速度型 131k

# 4) 验证
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Deepseek","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":20}'
```

权重加载约 12–22 分钟；CUDA graph 捕获 + JIT 另需 2–3 分钟；就绪标志为日志出现 `Application startup complete`。服务兼容 OpenAI `/v1/chat/completions`（含 tool calls）与 Anthropic `/v1/messages` 协议。

## 目录结构

```
├── patches/                        # vLLM 0.26.0 overlay 补丁 + 幂等应用脚本
│   ├── apply_sm89_patches.sh
│   ├── vllm_pkg/                   # 新增/修改的 vllm 文件（4 个含 .orig 供审计）
│   ├── site_headers/               # CCCL 版本检查屏蔽（nvcc 13.3 + CTK 13.0 混装）
│   └── README.md                   # 逐文件清单与应用/回滚
├── deploy/
│   ├── start.sh                    # 速度型（131k 上下文）
│   ├── start_1m.sh                 # 1M 上下文（CPU-KV-pool）
│   └── stop.sh
├── tests/
│   ├── test_sm89_mqa_logits.py     # indexer MQA logits 核数值对齐（连续 + 分页）
│   ├── test_sm89_sparse_attn.py    # 稀疏注意力 gather/prefill/decode 对齐
│   ├── test_indexer_layout.py      # 缓存布局/跨步视图/top-k 契约回归
│   ├── test_uva_pinned.py          # pinned UVA 读写语义
│   ├── test_niah.py                # 大海捞针矩阵 + 双针交叉
│   └── test_1m_context.py          # 1M 级单请求 + 并发验证
└── bench/
    ├── bench_speed.py              # 单流/并发速度测试
    ├── bench_decode.py
    └── bench_decode_longctx.py
```

## 环境变量

| 变量 | 作用 |
|---|---|
| `VLLM_DSV4_KV_CPU_GB=14` | **CPU-KV-pool 总开关**：KV 池大小（GB/卡，pinned）；不设则 KV 池在显存 |
| `VLLM_DSV4_GATHER_CPP=1` | prefill indexer gather 回退 C++ 算子（仅 GPU 池） |
| `VLLM_DSV4_FORENSIC=1` | 调试：gather 时刻 GPU 侧快照，步间落盘 /tmp（捕获安全） |
| `VLLM_DSV4_DUMP_SEL=1` | 调试：导出 indexer logits/top-k 选取（需 `--no-async-scheduling`） |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | 权重占满显存时关闭图内存预估 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1M prefill 显存紧张时必需 |

## 测试

```bash
# 内核级单测（无需起服务）
python tests/test_sm89_mqa_logits.py
python tests/test_sm89_sparse_attn.py
python tests/test_indexer_layout.py
python tests/test_uva_pinned.py

# 端到端（服务启动后）
DSV4_MODEL_DIR=/path/to/weights python tests/test_niah.py \
    --lengths 5,16,32,64,128 --depths 5,25,50,75,98 --double
DSV4_MODEL_DIR=/path/to/weights python tests/test_1m_context.py --target 950000
```

数值对齐参考量级：连续版 MQA logits max|Δ|=3.9e-4（vs torch 参考）；分页版 ≈2.3e-5；KV gather 精确一致（0）；稀疏注意力 3.9e-3（bf16 舍入）。

## 已知限制

- 8 路 1M 全活跃不可行（需 41 GB/卡 KV 池）；超额请求排队（等待不占 KV）；
- prefill 期间瞬时显存余量 ~0.5 GB，依赖 `expandable_segments`；
- `--max-num-batched-tokens`：CPU-KV 池模式固定 4096（8192 时 8k-token 块的激活在权重占 ~92% 显存下 OOM）；GPU 池模式保持默认 8192（调大到 16k 会使 KV 分组分配劣化）；
- DSpark 投机采样与 1M 上下文在本机显存下互斥。

## 许可

Apache-2.0（与 vLLM 一致）。算子语义参考 vLLM 自带的 ROCm/XPU Triton 实现。欢迎在其他 pre-Hopper 卡上测试并提 issue / PR。
