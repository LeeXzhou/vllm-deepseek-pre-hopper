#!/usr/bin/env python3
"""1M 上下文验证：大海捞针 + 并发能力。

用法: DSV4_MODEL_DIR=/path/to/weights python3 test_1m_context.py [--target 950000] [--concurrency 0]
依赖: 能加载该模型 tokenizer 的 Python 环境（DSV4_PY 可覆盖，默认当前解释器）
"""
import argparse
import os
import concurrent.futures as cf
import json
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PY = os.environ.get("DSV4_PY", sys.executable)
MODEL_DIR = os.environ.get("DSV4_MODEL_DIR")

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=950000, help="单请求目标 token 数")
ap.add_argument("--concurrency", type=int, default=0, help=">0: 跑 N 并发 x 100k 测试")
args = ap.parse_args()

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


def count_tokens(text):
    if not MODEL_DIR:
        sys.exit("✗ 未设置 DSV4_MODEL_DIR（权重目录，用于加载 tokenizer），如："
                 "DSV4_MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731")
    code = (
        "import json,sys;"
        "from transformers import AutoTokenizer;"
        f"tok=AutoTokenizer.from_pretrained('{MODEL_DIR}',trust_remote_code=True);"
        "print(json.dumps(len(tok.encode(open(sys.argv[1]).read()))))"
    )
    with open("/tmp/_prompt.txt", "w", encoding="utf-8") as f:
        f.write(text)
    out = subprocess.run([PY, "-c", code, "/tmp/_prompt.txt"],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().strip('"'))


def build_prompt(target_tokens, needle, seed=0):
    """构造约 target_tokens 的多样化填充文本 + 针（needle 放在 8% 处）。"""
    parts = []
    est = 0
    unit_cost = 28  # 每单位约 28 token（粗估，稍后校准）
    i = seed
    while est < target_tokens:
        parts.append(f"{i % 97}. {FILLER_UNITS[(i * 7 + seed) % len(FILLER_UNITS)]}")
        est += unit_cost
        i += 1
    text = "\n".join(parts)
    # 插入 needle 到约 8% 处
    pos = int(len(text) * 0.08)
    text = text[:pos] + f"\n【重要】{needle}\n" + text[pos:]
    # 末尾提问
    text += "\n\n请回答：文中【重要】标注的那句话的内容是什么？只复述那句话。"
    return text


def ask(text, max_tokens=48, temperature=0.0, timeout=7200):
    body = json.dumps({
        "model": "Deepseek",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    return r["choices"][0]["message"]["content"], r["usage"], dt


if args.concurrency > 0:
    n = args.concurrency
    print(f"== 并发 {n} × ~100k 大海捞针 ==")
    prompts = []
    for s in range(n):
        needle = f"验证码是{100000 + s * 137}，颜色是{['红', '蓝', '绿', '紫', '橙', '青', '金', '银'][s % 8]}色。"
        prompts.append((s, needle, build_prompt(100000, needle, seed=s)))
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        futs = {ex.submit(ask, p, 48): (s, nd) for s, nd, p in prompts}
        for fu in cf.as_completed(futs):
            s, nd = futs[fu]
            c, u, dt = fu.result()
            ok = str(100000 + s * 137) in c
            print(f"req{s}: prompt={u['prompt_tokens']} out={u['completion_tokens']} "
                  f"{dt:.0f}s needle_ok={ok} 回答: {c[:60]!r}")
    print(f"总耗时 {time.time()-t0:.0f}s")
    sys.exit(0)

print(f"== 构造 ~{args.target} token 提示（含针）==")
needle = "神秘编号是 884311，形状是正六边形。"
t0 = time.time()
text = build_prompt(args.target, needle)
n_tok = count_tokens(text)
print(f"构造完成: {len(text)} chars, {n_tok} tokens ({time.time()-t0:.0f}s)")
if n_tok > 1040000:
    print("超出 1M 上限，截断")
    sys.exit(1)

print("== 发送 1M 级请求（prefill 较慢，请等待）==")
c, u, dt = ask(text, max_tokens=64)
prefill_rate = u["prompt_tokens"] / dt
print(f"prompt={u['prompt_tokens']} completion={u['completion_tokens']} wall={dt:.0f}s")
print(f"端到端吞吐(含prefill) {prefill_rate:.0f} tok/s")
print(f"回答: {c[:200]!r}")
print(f"NEEDLE_OK: {'884311' in c and '六边形' in c}")
