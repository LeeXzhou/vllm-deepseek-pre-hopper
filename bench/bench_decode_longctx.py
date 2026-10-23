#!/usr/bin/env python3
"""长上下文 decode 速度实测：流式拆分 TTFT(prefill) 与 decode 段。

背景（07 §4.5）：每步 KV 访问 = 注意力 gather ~13MB（与 N 无关）
+ indexer 打分扫描 ~0.71KB × N（选 top-512 须扫全量压缩候选）。
67.4 tok/s 是短提示口径；本脚本验证 decode 速度随 N 的线性下降。

用法: python3 bench_decode_longctx.py [--lengths 20000,131072,300000,581000] [--gen 48]
依赖: 服务已启动（start_1m.sh，1M 配置）；token 精确计数用 vllm_deepseek 环境 tokenizer。
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PY = "/home/lenovo/anaconda3/envs/vllm_deepseek/bin/python"
MODEL_DIR = "/data/model/deepseek/DeepSeek-V4-Flash-0731"

FILLER_UNITS = [
    "深度学习模型的训练需要大量标注数据，数据质量往往比数量更重要。",
    "分布式系统的核心挑战在于一致性与可用性之间的权衡。",
    "光合作用将太阳能转化为化学能，是地球生态系统的能量基础。",
    "编程语言的抽象层次越高，开发者生产力通常越高，但运行效率可能降低。",
    "丝绸之路促进了东西方文明之间的贸易与技术交流。",
    "免疫系统的记忆细胞能够在再次遇到病原体时快速发起响应。",
    "城市交通规划需要综合考虑公共交通、步行与私家车的平衡。",
    "量子计算机利用叠加与纠缠原理，在特定问题上具有指数级加速潜力。",
]

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", default="20000,131072,300000,581000",
                help="逗号分隔的目标提示 token 数")
ap.add_argument("--gen", type=int, default=48, help="每请求生成 token 数")
args = ap.parse_args()


def count_tokens(text):
    code = (
        "import json,sys;"
        "from transformers import AutoTokenizer;"
        f"tok=AutoTokenizer.from_pretrained('{MODEL_DIR}',trust_remote_code=True);"
        "print(json.dumps(len(tok.encode(open(sys.argv[1]).read()))))"
    )
    with open("/tmp/_prompt_lc.txt", "w", encoding="utf-8") as f:
        f.write(text)
    out = subprocess.run([PY, "-c", code, "/tmp/_prompt_lc.txt"],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip('"'))


def build_prompt(target_tokens):
    # 先用 100 单位校准每单元实际 token 数（粗估不可靠：实测 ~17 tok/单元）
    sample = "\n".join(f"{i}. {FILLER_UNITS[(i * 7) % len(FILLER_UNITS)]}"
                       for i in range(100))
    per_unit = count_tokens(sample) / 100

    def make(n_units):
        parts = [f"{i % 97}. {FILLER_UNITS[(i * 7) % len(FILLER_UNITS)]}"
                 for i in range(n_units)]
        return ("\n".join(parts)
                + "\n\n不要思考，不要解释，直接从 1 开始连续报数，只输出数字，每行一个，数到尽可能大。")

    n_units = int(target_tokens / per_unit) + 1
    text = make(n_units)
    for _ in range(2):  # 精确补齐（最多两轮）
        n = count_tokens(text)
        if n >= target_tokens:
            break
        n_units += int((target_tokens - n) / per_unit) + 2
        text = make(n_units)
    return text


def run(text, gen):
    """流式请求：TTFT ≈ prefill；后续 chunk 间隔 = decode。返回分解结果。"""
    body = json.dumps({
        "model": "Deepseek",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": gen, "temperature": 0.0, "stream": True,
        "min_tokens": gen,  # 防止长上下文下模型提前 EOS，保证 decode 统计量
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, last, chunks, usage = None, None, 0, {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage") and d["usage"].get("completion_tokens"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0].get("delta", {})
            delta = ch.get("content") or ch.get("reasoning_content")
            if delta:
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                last = now
                chunks += 1
    total = (last or time.time()) - t0
    return {"ttft": ttft or total, "total": total, "usage": usage}


print(f"{'N_tok':>8} {'ttft(s)':>8} {'prefill':>9} {'decode(s)':>9} {'decode':>9}")
for L in [int(x) for x in args.lengths.split(",")]:
    text = build_prompt(L)
    r = run(text, args.gen)
    u = r["usage"]
    n_in = u.get("prompt_tokens", count_tokens(text))
    n_out = u.get("completion_tokens", 0)
    dec_s = max(r["total"] - r["ttft"], 1e-3)
    rate = max(n_out - 1, 1) / dec_s
    warn = "  ⚠ completion<8，decode 统计无效" if n_out < 8 else ""
    print(f"{n_in:>8} {r['ttft']:>8.1f} {n_in / r['ttft']:>8.0f}/s "
          f"{dec_s:>9.1f} {rate:>8.1f}/s (out={n_out}){warn}", flush=True)
