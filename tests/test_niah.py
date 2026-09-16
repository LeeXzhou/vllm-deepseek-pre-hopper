#!/usr/bin/env python3
"""大海捞针矩阵测试：多上下文长度 × 多深度 + 双针交叉一致性。

用法:
  DSV4_MODEL_DIR=/path/to/weights python test_niah.py \
      [--lengths 8,16,32,64,128] [--depths 5,25,50,75,98] [--host 127.0.0.1]

每个单元格一个请求：填充文本中按深度插入唯一编号的针，末尾提问。
判定：回答中包含该编号（严格子串）。双针模式：两根针不同深度，一次问两个。
结果存 /tmp/niah_results.json 并打印矩阵。
"""
import argparse
import os
import json
import random
import sys
import time
import urllib.request

URL_BASE = "http://{host}:8000/v1/chat/completions"
MODEL_DIR = os.environ.get("DSV4_MODEL_DIR")

FILLER_UNITS = [
    "深度学习模型的训练需要大量标注数据，数据质量往往比数量更重要。",
    "分布式系统的核心挑战在于一致性与可用性之间的权衡。",
    "光合作用将太阳能转化为化学能，是地球生态系统的能量基础。",
    "编程语言的抽象层次越高，开发者生产力通常越高，但运行效率可能降低。",
    "丝绸之路促进了东西方文明之间的贸易与技术交流。",
    "免疫系统的记忆细胞能够在再次遇到病原体时快速发起响应。",
    "城市交通规划需要综合考虑公共交通、步行与私家车的平衡。",
    "量子计算机利用叠加与纠缠原理，在特定问题上具有指数级加速潜力。",
    "海洋洋流调节着全球气候，寒流与暖流的交汇处往往形成大渔场。",
    "古典音乐的和声进行遵循一定的调性逻辑，但二十世纪出现了无调性尝试。",
    "发酵食品的风味来自微生物的代谢产物，温度与时间是关键变量。",
    "行星轨道的摄动分析曾经帮助天文学家预言了海王星的存在。",
]

Attrs = ["正六边形", "暗红色", "椭圆形", "银灰色", "正十二面体", "深蓝色", "菱形", "古铜色"]


def make_tokenizer():
    from transformers import AutoTokenizer

    if not MODEL_DIR:
        sys.exit("✗ 未设置 DSV4_MODEL_DIR（权重目录，用于加载 tokenizer），如："
                 "DSV4_MODEL_DIR=/path/to/DeepSeek-V4-Flash-0731")
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def build_haystack(tok, target_tokens, needles, seed=0, plain=False):
    """needles: list of (depth_fraction, needle_text). Returns prompt text."""
    units = []
    est = 0
    while est < target_tokens + 200:
        i = len(units)
        if plain:
            units.append(FILLER_UNITS[(i * 7 + seed) % len(FILLER_UNITS)])
        else:
            units.append(f"{(i * 13 + seed) % 997}. {FILLER_UNITS[(i * 7 + seed) % len(FILLER_UNITS)]}")
        est += 24
    n_units = len(units)
    for depth_frac, needle in needles:
        pos = max(1, min(n_units - 2, int(depth_frac * n_units)))
        units.insert(pos, f"【重要】{needle}")
        n_units += 1
    text = "\n".join(units)
    return text


def ask(host, text, max_tokens=128, timeout=7200):
    body = json.dumps(
        {
            "model": "Deepseek",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{host}:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    ans = msg.get("content") or ""
    if not ans.strip():
        ans = msg.get("reasoning_content") or ""
    return ans, r["usage"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lengths", default="8,16,32,64,128", help="上下文长度(千token), 逗号分隔")
    ap.add_argument("--depths", default="5,25,50,75,98", help="深度百分比, 逗号分隔")
    ap.add_argument("--double", action="store_true", help="附加双针交叉测试")
    ap.add_argument("--out", default="/tmp/niah_results.json")
    ap.add_argument("--plain", action="store_true", help="无行号纯文本填充（降低数字混叠）")
    args = ap.parse_args()

    lengths = [int(x) * 1000 for x in args.lengths.split(",")]
    depths = [int(x) / 100.0 for x in args.depths.split(",")]

    print("加载 tokenizer ...")
    tok = make_tokenizer()

    results = []
    total = len(lengths) * len(depths)
    done = 0
    for L in lengths:
        for depth in depths:
            num = 100000 + int(L * depth) * 7 + int(depth * 100)
            attr = Attrs[int(L * depth) % len(Attrs)]
            needle = f"魔法数字是 {num}，它的形状是{attr}。"
            text = build_haystack(tok, L, [(depth, needle)], seed=int(L * depth), plain=args.plain)
            n_tok = len(tok.encode(text))
            q = "\n\n问题：上文【重要】标注的魔法数字是多少？只回答数字本身。"
            prompt = text + q
            try:
                ans, usage, dt = ask(args.host, prompt)
                ok = str(num) in ans
                prefill_tps = usage["prompt_tokens"] / dt
                row = dict(
                    ctx=L, depth=int(depth * 100), needle_num=num, ok=ok,
                    answer=ans.strip()[:80], prompt_tokens=usage["prompt_tokens"],
                    wall_s=round(dt, 1), prefill_tps=round(prefill_tps),
                )
            except Exception as e:
                row = dict(ctx=L, depth=int(depth * 100), needle_num=num, ok=False,
                           answer=f"ERROR {str(e)[:80]}", prompt_tokens=0, wall_s=0,
                           prefill_tps=0)
            results.append(row)
            done += 1
            print(
                f"[{done}/{total}] ctx={L//1000}k depth={int(depth*100)}% "
                f"{'✓' if row['ok'] else '✗'} {row['wall_s']}s "
                f"ans={row['answer'][:40]!r}"
            )

    if args.double:
        print("\n== 双针交叉一致性（浅+深两根针，一次问两个）==")
        for L in lengths:
            n1 = 200000 + L
            n2 = 600000 + L
            needles = [
                (0.10, f"魔法数字是 {n1}，它的形状是{Attrs[L % 8]}。"),
                (0.85, f"魔法数字是 {n2}，它的形状是{Attrs[(L+3) % 8]}。"),
            ]
            text = build_haystack(tok, L, needles, seed=L, plain=args.plain)
            q = (
                f"\n\n问题：上文有两句【重要】标注的话。第一句的魔法数字和第二句的魔法数字分别是多少？"
                "按顺序只回答两个数字，用逗号分隔。"
            )
            try:
                ans, usage, dt = ask(args.host, text + q, max_tokens=96)
                ok = str(n1) in ans and str(n2) in ans
                row = dict(ctx=L, depth="double", ok=ok, answer=ans.strip()[:80],
                           prompt_tokens=usage["prompt_tokens"], wall_s=round(dt, 1))
            except Exception as e:
                row = dict(ctx=L, depth="double", ok=False,
                           answer=f"ERROR {str(e)[:60]}", prompt_tokens=0, wall_s=0)
            results.append(row)
            print(f"ctx={L//1000}k double {'✓' if row['ok'] else '✗'} {row['wall_s']}s ans={row['answer'][:50]!r}")

    # summary matrix
    print("\n========= 命中矩阵 (行=长度k, 列=深度%) =========")
    header = "ctx\\depth " + "".join(f"{int(d*100):>6}" for d in depths)
    print(header)
    for L in lengths:
        cells = ""
        for depth in depths:
            r = next(x for x in results if x["ctx"] == L and x["depth"] == int(depth * 100))
            cells += f"{'  ✓  ' if r['ok'] else '  ✗  '}"
        print(f"{L//1000:>8}k {cells}")
    if args.double:
        for L in lengths:
            r = next(x for x in results if x["ctx"] == L and x["depth"] == "double")
            print(f"double {L//1000}k: {'✓' if r['ok'] else '✗'}")

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n总计: {n_ok}/{len(results)} 命中 ({100.0*n_ok/len(results):.0f}%)")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"结果已存 {args.out}")


if __name__ == "__main__":
    main()
