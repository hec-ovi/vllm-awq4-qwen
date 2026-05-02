"""
Streaming bench matrix for vllm-awq4-qwen.

Matrix:
  context sizes: 0, 2k, 4k, 8k, 16k, 32k tokens (in system message)
  cases:         normal | two_tools | thinking | no_thinking
  output target: ~2048 tokens of decode per request (max_tokens cap)
  streaming:     always on (Server-Sent Events)
  runs/test:     configurable (default 2)

Per request we measure:
  TTFT  = time-to-first-token (proxies prefill latency)
  prefill t/s = prompt_tokens / TTFT
  decode  t/s = decode_tokens / (total_time - TTFT)

Reports per (context, case): mean, median, min, max for prefill and decode t/s.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"


# ---------- HTTP / SSE helpers ----------

def _post_json(path: str, body: dict, stream: bool = False, timeout: float = 1800.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text: str) -> list[int]:
    body = {"model": MODEL, "prompt": text, "add_special_tokens": False}
    with _post_json("/tokenize", body) as resp:
        return json.loads(resp.read())["tokens"]


def detokenize(tokens: list[int]) -> str:
    body = {"model": MODEL, "tokens": tokens}
    with _post_json("/detokenize", body) as resp:
        d = json.loads(resp.read())
    return d.get("prompt", d.get("text", ""))


def build_context(target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    seed = (
        "The history of computing began long before the first electronic computers. "
        "Charles Babbage designed the Difference Engine in the 1820s, and Ada Lovelace "
        "wrote what is considered the first algorithm intended for a machine. The 1940s "
        "saw the first electronic computers. Alan Turing's theoretical work on computability "
        "and the Turing machine laid foundations that are still used today. Claude Shannon's "
        "information theory connected logic to electronic switches. "
    )
    blob = seed * (target_tokens // 30 + 16)
    toks = tokenize(blob)
    while len(toks) < target_tokens:
        blob += seed * 8
        toks = tokenize(blob)
    return detokenize(toks[:target_tokens])


# ---------- Prompts and tools ----------

NORMAL_QUESTION = (
    "Write a detailed essay explaining the architectural differences between Transformer-based "
    "language models and recurrent neural networks. Cover attention mechanisms, computational "
    "complexity, parallelization, training stability, gradient flow, and practical use cases. "
    "Include concrete examples (BERT, GPT, LSTM, GRU) and discuss recent hybrid architectures. "
    "Aim for approximately 2000 words of substantive technical content."
)

THINKING_QUESTION = (
    "Solve this multi-step problem and show your full reasoning, then give the answer:\n"
    "A train leaves city A at 8:00 AM traveling east at 60 mph. A second train leaves city B "
    "(180 miles east of A) at 9:30 AM traveling west at 75 mph. A third train leaves city A "
    "at 10:00 AM traveling east at 90 mph. (a) At what time and distance from A do trains 1 "
    "and 2 meet? (b) When does train 3 catch up to train 1?\n\n"
    "Then design and explain a generalized Python function `meeting_times(events)` that takes "
    "a list of (start_time_minutes, position_miles, velocity_mph) events and returns all "
    "pairwise meeting times and positions. Discuss correctness, edge cases (parallel trains, "
    "no meeting, identical positions), and complexity. Aim for approximately 2000 words of "
    "explanation plus working code with tests."
)

TWO_TOOLS_QUESTION = (
    "I'm planning a trip to Tokyo for next week. First, check the current weather there in "
    "celsius, then calculate the total cost in USD for 5 nights at $180 per night plus a "
    "$1200 round-trip flight. After getting the tool results, write a detailed travel "
    "itinerary covering neighborhoods (Shibuya, Shinjuku, Asakusa), food recommendations, "
    "transit (JR Pass strategy), day-by-day activities, and a final budget breakdown. Aim "
    "for approximately 2000 words of itinerary content."
)

TWO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Numeric expression like 5*180 + 1200",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ---------- Streaming bench ----------

@dataclass
class Result:
    prompt_tokens: int = 0
    decode_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    error: str | None = None

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.ttft_s if self.ttft_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        decode_time = self.total_s - self.ttft_s
        return self.decode_tokens / decode_time if decode_time > 0 else 0.0


def stream_chat(messages: list[dict], *, max_tokens: int, thinking: bool, tools: list | None = None) -> Result:
    body: dict = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    r = Result()
    t0 = time.perf_counter()
    first_token_t: float | None = None
    try:
        with _post_json("/v1/chat/completions", body, stream=True) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    u = chunk["usage"]
                    r.prompt_tokens = u.get("prompt_tokens") or r.prompt_tokens
                    r.decode_tokens = u.get("completion_tokens") or r.decode_tokens
                ch = chunk.get("choices") or []
                if not ch:
                    continue
                delta = ch[0].get("delta") or {}
                got_token = bool(delta.get("content")) or bool(delta.get("tool_calls")) or bool(delta.get("reasoning"))
                if got_token and first_token_t is None:
                    first_token_t = time.perf_counter()
    except urllib.error.HTTPError as e:
        r.error = f"HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}"
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"

    r.total_s = time.perf_counter() - t0
    if first_token_t is not None:
        r.ttft_s = first_token_t - t0
    return r


# ---------- Run matrix ----------

CASES = ["normal", "two_tools", "thinking", "no_thinking"]


def make_messages(case: str, context_text: str) -> tuple[list[dict], list | None, bool]:
    sys_content = "You are a helpful assistant."
    if context_text:
        sys_content += "\n\nReference document:\n" + context_text
    sys_msg = {"role": "system", "content": sys_content}
    if case == "normal":
        return [sys_msg, {"role": "user", "content": NORMAL_QUESTION}], None, False
    if case == "thinking":
        return [sys_msg, {"role": "user", "content": THINKING_QUESTION}], None, True
    if case == "no_thinking":
        return [sys_msg, {"role": "user", "content": THINKING_QUESTION}], None, False
    if case == "two_tools":
        return [sys_msg, {"role": "user", "content": TWO_TOOLS_QUESTION}], TWO_TOOLS, False
    raise ValueError(case)


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="/home/hec/workspace/vllm-awq4-qwen/test/bench_matrix_results.json")
    args = ap.parse_args()

    print(f"\nMatrix: {len(args.contexts)} contexts × {len(args.cases)} cases × {args.runs} runs "
          f"= {len(args.contexts) * len(args.cases) * args.runs} requests, decode cap {args.max_tokens}\n")

    print("Building context blobs...")
    ctx_text: dict[int, str] = {}
    for c in args.contexts:
        text = build_context(c) if c > 0 else ""
        actual = len(tokenize(text)) if text else 0
        ctx_text[c] = text
        print(f"  ctx target {c:>6} -> actual {actual:>6} tokens")

    hdr = f"\n{'ctx':>6} {'case':>12} {'run':>4} {'prompt':>7} {'decode':>7} {'TTFT':>7} {'total':>7} {'pp t/s':>8} {'dec t/s':>8}  err"
    print(hdr)
    print("-" * 100)

    rows: list[dict] = []
    for ctx in args.contexts:
        for case in args.cases:
            messages, tools, thinking = make_messages(case, ctx_text[ctx])
            for run in range(args.runs):
                res = stream_chat(messages, max_tokens=args.max_tokens, thinking=thinking, tools=tools)
                rows.append({
                    "ctx": ctx, "case": case, "run": run,
                    "prompt_tokens": res.prompt_tokens, "decode_tokens": res.decode_tokens,
                    "ttft_s": res.ttft_s, "total_s": res.total_s,
                    "prefill_tps": res.prefill_tps, "decode_tps": res.decode_tps,
                    "error": res.error,
                })
                err = res.error or ""
                print(
                    f"{ctx:>6} {case:>12} {run:>4} {res.prompt_tokens:>7} {res.decode_tokens:>7} "
                    f"{res.ttft_s:>7.1f} {res.total_s:>7.1f} {res.prefill_tps:>8.1f} {res.decode_tps:>8.2f}  {err}",
                    flush=True,
                )

    print()
    print("=" * 100)
    print("AGGREGATE STATS (per context × case; t/s)")
    print("=" * 100)
    print(f"\n{'ctx':>6} {'case':>12} {'metric':>9}  {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'n':>3}")
    print("-" * 80)
    summary: dict = {}
    for ctx in args.contexts:
        for case in args.cases:
            sel = [r for r in rows if r["ctx"] == ctx and r["case"] == case and not r["error"]]
            pp = stats([r["prefill_tps"] for r in sel if r["prefill_tps"] > 0])
            dc = stats([r["decode_tps"] for r in sel if r["decode_tps"] > 0])
            summary[f"{ctx}_{case}"] = {"prefill": pp, "decode": dc, "n": len(sel)}
            print(f"{ctx:>6} {case:>12} {'prefill':>9}  {pp['mean']:>8.1f} {pp['median']:>8.1f} {pp['min']:>8.1f} {pp['max']:>8.1f} {pp['n']:>3}")
            print(f"{ctx:>6} {case:>12} {'decode':>9}  {dc['mean']:>8.2f} {dc['median']:>8.2f} {dc['min']:>8.2f} {dc['max']:>8.2f} {dc['n']:>3}")

    out = {"raw": rows, "summary": summary, "config": vars(args)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
