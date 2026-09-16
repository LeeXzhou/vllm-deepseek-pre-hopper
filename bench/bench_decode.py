#!/usr/bin/env python3
"""单流/并发解码速度测试（短提示，专注 decode 吞吐）。"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
ap.add_argument("--gen", type=int, default=256)
args = ap.parse_args()
URL = "http://127.0.0.1:8000/v1/chat/completions"


def one(max_tokens):
    body = json.dumps({
        "model": "Deepseek",
        "messages": [{"role": "user", "content": "请写一篇关于推理引擎优化的短文。"}],
        "max_tokens": max_tokens, "temperature": 0.6,
    }).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}),
        timeout=1800))
    dt = time.time() - t0
    u = r["usage"]
    # 减去约 20 token 提示的 prefill（~1-2s），以 decode 为主
    return {"prompt": u["prompt_tokens"], "out": u["completion_tokens"],
            "wall": round(dt, 2), "tok_s": round(u["completion_tokens"] / dt, 2)}


res = {}
# 预热（JIT/图捕获）
one(16)
r = one(args.gen)
print("单流:", r)
res["single"] = r
for conc in (4, 8):
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        rs = list(ex.map(one, [args.gen] * conc))
    wall = time.time() - t0
    tot = sum(x["out"] for x in rs)
    agg = {"wall": round(wall, 2), "total_out": tot, "agg_tok_s": round(tot / wall, 2)}
    print(f"并发{conc}:", agg, "per-req:", [x["tok_s"] for x in rs])
    res[f"conc{conc}"] = agg
if args.out:
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("saved ->", args.out)
