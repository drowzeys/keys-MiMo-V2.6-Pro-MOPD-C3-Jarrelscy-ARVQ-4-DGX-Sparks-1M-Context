#!/usr/bin/env python3
"""Quality eval independent of our RL on-policy text: GSM8K, HumanEval, MMLU-Pro.

Greedy (temperature 0), thinking off (the serving default), same fixed samples for
every candidate. HumanEval completions are executed against the official tests.
  python3 qeval.py --label c1 [--gsm 250 --mmlu 280]
"""
import argparse
import gzip
import json
import os
import random
import re
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://10.100.10.1:8888")
ap.add_argument("--model", default="MiMo-V2.6-Pro-ARVQ")
ap.add_argument("--label", required=True)
ap.add_argument("--gsm", type=int, default=250)
ap.add_argument("--mmlu", type=int, default=280)
ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--tasks", default="gsm8k,humaneval,mmlu_pro")
ap.add_argument("--out", default="/home/keyspark/mimo26-arvq-tp4/mopd/quality")
a = ap.parse_args()


def chat(text, max_tokens):
    body = {"model": a.model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                a.base + "/v1/chat/completions", json.dumps(body).encode(),
                {"Content-Type": "application/json"}), timeout=1800))
            return r["choices"][0]["message"].get("content") or "", r["choices"][0].get("finish_reason")
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
    return "", "error:" + err[:100]


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def gsm_item(x):
    txt, fin = chat(x["question"] + "\n\nSolve step by step, then give the final answer on the last line as 'Answer: <number>'.", 1536)
    m = re.search(r"Answer:\s*(.+)", txt)
    pred = num(m.group(1)) if m else num(txt)
    gold = num(x["answer"].split("####")[-1])
    return {"ok": pred is not None and abs(pred - gold) < 1e-6, "fin": fin}


def mmlu_item(x):
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(x["options"]))
    txt, fin = chat(f"{x['question']}\n\n{opts}\n\nThink briefly, then give the final answer on the last line as 'Answer: <letter>'.", 1536)
    m = re.findall(r"Answer:\s*\(?([A-J])\b", txt)
    clean = re.sub(r"[*_`#]", "", txt)
    r = re.findall(r"answer(?:\s+is)?\s*[:：]?\s*\(?([A-J])\)?(?![A-Za-z])", clean, re.I)
    return {"ok": bool(m) and m[-1] == x["answer"], "ok_robust": bool(r) and r[-1].upper() == x["answer"],
            "fin": fin, "cat": x["category"], "gold": x["answer"], "text": txt[-600:]}


def he_item(x):
    txt, fin = chat("Complete the following Python function. Return the full function in one ```python code block, no explanation.\n\n```python\n"
                    + x["prompt"] + "```", 1536)
    m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.S)
    code = m[0] if m else txt
    if f"def {x['entry_point']}" not in code:
        code = x["prompt"] + code
    prog = code + "\n\n" + x["test"] + f"\n\ncheck({x['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        ok = subprocess.run(["python3", f.name], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    os.unlink(f.name)
    return {"ok": ok, "fin": fin}


rng = random.Random(1234)
gsm = load_dataset("openai/gsm8k", "main")["test"]
gsm = [gsm[i] for i in sorted(rng.sample(range(len(gsm)), a.gsm))]
mm = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
mm = [mm[i] for i in sorted(random.Random(4321).sample(range(len(mm)), a.mmlu))]
he = [json.loads(l) for l in gzip.open("/home/keyspark/exo/.venv/lib/python3.13/site-packages/human_eval/data/HumanEval.jsonl.gz", "rt")]

res = {"label": a.label}
rows = open(f"{a.out}/rows.{a.label}.jsonl", "w")
for name, data, fn in (("gsm8k", gsm, gsm_item), ("humaneval", he, he_item), ("mmlu_pro", mm, mmlu_item)):
    if name not in a.tasks.split(","):
        continue
    with ThreadPoolExecutor(a.conc) as ex:
        out = list(ex.map(fn, data))
    for i, r in enumerate(out):
        rows.write(json.dumps({"task": name, "i": i, **r}) + "\n")
    rows.flush()
    acc = sum(r["ok"] for r in out) / len(out)
    res[name] = round(acc, 4)
    if name == "mmlu_pro":
        res["mmlu_pro_robust"] = round(sum(r["ok_robust"] for r in out) / len(out), 4)
    res[name + "_n"] = len(out)
    res[name + "_truncated"] = sum(r["fin"] == "length" for r in out)
    print(name, res[name], f"n={len(out)}", flush=True)
open(f"{a.out}/qeval.jsonl", "a").write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
