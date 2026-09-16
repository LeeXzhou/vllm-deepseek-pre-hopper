#!/usr/bin/env python3
"""DeepSeek-V4-Flash-0731 on sm89 — speed benchmark.

Usage: python3 bench_speed.py [--host 127.0.0.1] [--port 8000]
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--out", default=None)
args = ap.parse_args()
URL = f"http://{args.host}:{args.port}/v1/chat/completions"


def one(prompt_tokens_hint, gen_tokens, temperature=0.6, timeout=1800):
    # ~prompt length via repeated filler text
    filler = "人工智能推理引擎的性能测试段落。" * (prompt_tokens_hint // 16 + 1)
    body = json.dumps(
        {
            "model": "Deepseek",
            "messages": [{"role": "user", "content": filler + "\n请基于以上内容写一段技术总结。"}],
            "max_tokens": gen_tokens,
            "temperature": temperature,
        }
    ).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    u = r["usage"]
    return {
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "wall_s": round(dt, 2),
        "tok_per_s": round(u["completion_tokens"] / dt, 2),
    }


results = {}
print("== 短提示单流（~20 tok in, 128 out）")
one_res = one(20, 128)
print(one_res)
results["single_short"] = one_res

print("== 长提示单流（~2000 tok in, 256 out）")
one_res = one(2000, 256)
print(one_res)
results["single_long"] = one_res

for conc in (4, 8):
    print(f"== 并发 {conc}（每请求 ~500 tok in, 192 out）")
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        rs = list(ex.map(lambda _: one(500, 192), range(conc)))
    wall = time.time() - t0
    tot = sum(r["completion_tokens"] for r in rs)
    agg = {"wall_s": round(wall, 2), "total_out_tokens": tot,
           "agg_tok_per_s": round(tot / wall, 2), "per_req": rs}
    print({"wall_s": agg["wall_s"], "total_out_tokens": tot, "agg_tok_per_s": agg["agg_tok_per_s"]})
    results[f"concurrency_{conc}"] = agg

print(json.dumps(results, ensure_ascii=False, indent=2))
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)
