"""UVA 微测：Triton 核与 C++ 算子直接读写 pinned 主机内存。"""
import torch
import vllm
from vllm.triton_utils import triton, tl


@triton.jit
def _copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(src + offs, mask=m)
    tl.store(dst + offs, x, mask=m)


def main():
    dev = "cuda:0"
    n = 1 << 20
    host_src = torch.arange(n, dtype=torch.int32).contiguous().pin_memory()
    host_dst = torch.zeros(n, dtype=torch.int32).pin_memory()

    _copy_kernel[(triton.cdiv(n, 1024),)](host_src, host_dst, n, BLOCK=1024)
    torch.cuda.synchronize()
    print("Triton 读pinned→写pinned:", "OK" if torch.equal(host_dst, host_src) else "FAIL")

    # C++ SWA 写入核（fp8_ds_mla 布局, uint8 池）
    q = torch.randn(4, 16, 512, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(4, 512, device=dev, dtype=torch.bfloat16)
    swa_cpu = torch.zeros(8, 4 * 584, dtype=torch.uint8).pin_memory()
    swa_gpu = torch.zeros(8, 4, 584, dtype=torch.uint8, device=dev)
    slots = torch.arange(4, device=dev, dtype=torch.int64)
    pos = torch.arange(4, device=dev, dtype=torch.int64)
    cos = torch.randn(1024, 64, device=dev, dtype=torch.float32)
    try:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, swa_cpu.view(8, -1), slots, pos, cos, 16, 1e-6, 4)
        torch.cuda.synchronize()
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, swa_gpu.view(8, -1), slots, pos, cos, 16, 1e-6, 4)
        torch.cuda.synchronize()
        same = torch.equal(swa_cpu.view(-1), swa_gpu.view(-1).cpu())
        print("C++ SWA 写入核 pinned:", "OK" if same else "MISMATCH")
    except Exception as e:
        print("C++ SWA 写入核 pinned: FAIL", str(e)[:200])

    # C++ indexer gather 核（读 pinned 池）
    kq = torch.zeros(4, 128, dtype=torch.uint8)
    ks = torch.zeros(4, 4, dtype=torch.uint8)
    vals = (torch.arange(2 * 64 * 132, dtype=torch.int32) % 250 + 1).to(torch.uint8)
    cache_cpu = vals.reshape(2, 64, 132).contiguous().pin_memory()
    cache_gpu = cache_cpu.cuda()
    cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=dev)
    bt = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
    from vllm._custom_ops import cp_gather_indexer_k_quant_cache
    cp_gather_indexer_k_quant_cache(cache_cpu, kq, ks, bt, cu)
    torch.cuda.synchronize()
    kq2 = torch.zeros_like(kq)
    ks2 = torch.zeros_like(ks)
    cp_gather_indexer_k_quant_cache(cache_gpu, kq2, ks2, bt, cu)
    torch.cuda.synchronize()
    ok = torch.equal(kq, kq2) and torch.equal(ks, ks2)
    print("C++ indexer gather 读 pinned:", "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
