#!/usr/bin/env python3
"""Tool-call repetition bench (Xiaomi's MiMo-V2.6 metric), replaying real Hermes contexts.

Each saved Hermes request (~/.hermes/sessions/request_dump_*.json) is re-sent K
times exactly as Hermes sent it (messages, tools, its own sampling fields),
non-streaming. For every sampled assistant turn:
  N = tool calls, U = unique calls after JSON canonicalisation of
  (name, arguments); duplicate rate = (N - U) / N  (Xiaomi's within-turn metric)
  flooding = N >= 32; truncated = finish_reason "length".
Reports: response-level duplicate rate (share of turns with any duplicate),
call-level duplicate rate, flooding rate, mean calls per turn, truncation rate.

  python3 toolrep_bench.py --label rl-arvq --k 5
"""
import argparse
import glob
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://10.100.10.1:8888")
ap.add_argument("--model", default="MiMo-V2.6-Pro-ARVQ")
ap.add_argument("--label", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--out", default="/home/keyspark/mimo26-arvq-tp4/mopd/toolrep.jsonl")
a = ap.parse_args()

KEEP = {"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
        "chat_template_kwargs", "tool_choice", "parallel_tool_calls", "reasoning_effort"}


def load_requests():
    reqs = []
    for f in sorted(glob.glob(os.path.expanduser("~/.hermes/sessions/request_dump_*.json")), key=os.path.getmtime):
        d = json.load(open(f))
        body = (d.get("request") or {}).get("body") or {}
        if not body.get("messages") or not body.get("tools"):
            continue
        req = {k: v for k, v in body.items() if k in KEEP}
        req.update(model=a.model, messages=body["messages"], tools=body["tools"],
                   max_tokens=a.max_tokens, stream=False)
        reqs.append((os.path.basename(f), req))
    return reqs


def canon(call):
    fn = call.get("function") or {}
    args = fn.get("arguments") or "{}"
    try:
        args = json.dumps(json.loads(args), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        args = str(args)
    return fn.get("name", "") + "|" + args


def run(item):
    name, req, i = item
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            a.base + "/v1/chat/completions", json.dumps(req).encode(),
            {"Content-Type": "application/json"}), timeout=3600))
    except Exception as exc:
        return {"dump": name, "sample": i, "error": str(exc)[:200]}
    ch = r["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    keys = [canon(c) for c in calls]
    return {"dump": name, "sample": i, "n": len(keys), "u": len(set(keys)),
            "finish": ch.get("finish_reason"), "completion_tokens": r["usage"]["completion_tokens"],
            "secs": round(time.time() - t0, 1)}


reqs = load_requests()
items = [(n, q, i) for n, q in reqs for i in range(a.k)]
print(f"{len(reqs)} Hermes contexts x {a.k} samples = {len(items)} turns", flush=True)
out = open(a.out.replace(".jsonl", f".{a.label}.rows.jsonl"), "w")
lock = threading.Lock()
rows = []
with ThreadPoolExecutor(a.concurrency) as ex:
    for j, row in enumerate(ex.map(run, items)):
        rows.append(row)
        with lock:
            out.write(json.dumps(row) + "\n"); out.flush()
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(items)} done", flush=True)

ok = [r for r in rows if "error" not in r]
withcalls = [r for r in ok if r["n"] > 0]
tot_n = sum(r["n"] for r in ok); tot_dup = sum(r["n"] - r["u"] for r in ok)
res = {
    "label": a.label, "turns": len(ok), "errors": len(rows) - len(ok),
    "turns_with_tool_calls": len(withcalls),
    "response_dup_rate": round(sum(1 for r in withcalls if r["u"] < r["n"]) / max(1, len(withcalls)), 5),
    "call_dup_rate": round(tot_dup / max(1, tot_n), 5),
    "flood_rate_ge32": round(sum(1 for r in ok if r["n"] >= 32) / max(1, len(ok)), 5),
    "mean_calls_per_tool_turn": round(statistics.mean([r["n"] for r in withcalls]), 2) if withcalls else 0,
    "max_calls": max([r["n"] for r in ok] or [0]),
    "truncated_rate": round(sum(1 for r in ok if r["finish"] == "length") / max(1, len(ok)), 5),
}
open(a.out, "a").write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
