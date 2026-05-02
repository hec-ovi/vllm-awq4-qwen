#!/usr/bin/env python3
"""bench_prefill_warmup.py - measure 1k-token prefill rate cold vs warm.

Hypothesis under test (from .research/prefill-burst-vs-sustained):
the first 1k prefill on a freshly-booted engine pays Triton autotune
cost on first-encountered shape. Subsequent identical-shape requests
should reuse the autotuned config and run at the kernel's actual rate.

If runs 2-5 are noticeably faster than run 1, autotune cold-cache is
a real component of the burst-vs-sustained gap. If all 5 runs come in
at roughly the same (low) rate, autotune is amortized and the gap is
elsewhere (eager dispatch, hipMemcpyWithStream, etc).

Reads vLLM /metrics before/after each request to compute prefill rate
from server-side ground truth, not wall time.

Usage:
    python3 test/bench_prefill_warmup.py
    python3 test/bench_prefill_warmup.py --host http://127.0.0.1:8000
    python3 test/bench_prefill_warmup.py --runs 5 --target-tokens 1024

Stdlib only, no deps.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:8000"
DEFAULT_MODEL = "Qwen3.6-27B-AWQ4"

DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def fetch_metrics(host: str) -> dict[str, float]:
    """Snapshot /metrics. Sums values across labels per metric name."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(host + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception as e:
        print(f"{RED}metrics scrape failed: {e}{RESET}", file=sys.stderr)
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_end = line.find("{")
        if name_end == -1:
            name_end = line.find(" ")
        if name_end <= 0:
            continue
        name = line[:name_end]
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            try:
                value = float(parts[-2])
            except ValueError:
                continue
        out[name] = out.get(name, 0.0) + value
    return out


def delta(before: dict, after: dict, key: str) -> float:
    return after.get(key, 0.0) - before.get(key, 0.0)


def build_prompt(target_tokens: int, host: str, model: str) -> tuple[str, int]:
    """Build a prompt of approximately target_tokens by concatenating a
    stable Latin passage and trimming. Confirms actual token count via
    /tokenize. Returns (prompt_text, actual_token_count).
    """
    seed = (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
        "enim ad minim veniam quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor "
        "in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
        "nulla pariatur. Excepteur sint occaecat cupidatat non proident "
        "sunt in culpa qui officia deserunt mollit anim id est laborum. "
    )
    text = seed * (target_tokens // 50 + 4)
    body = {"model": model, "prompt": text}
    req = urllib.request.Request(
        host + "/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            j = json.loads(resp.read().decode())
        tokens = j.get("tokens") or []
        n = len(tokens) if tokens else j.get("count", 0)
    except Exception:
        n = len(text) // 4
    if n <= target_tokens:
        return text, n
    ratio = target_tokens / n
    cut = int(len(text) * ratio * 0.97)
    text = text[:cut]
    body = {"model": model, "prompt": text}
    req = urllib.request.Request(
        host + "/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            j = json.loads(resp.read().decode())
        n = len(j.get("tokens") or []) or j.get("count", 0)
    except Exception:
        pass
    return text, n


def run_one(host: str, model: str, prompt: str, max_output_tokens: int = 64) -> dict:
    """Send prompt and measure both prefill and decode rates.
    max_output_tokens defaults to 64 so decode rate has a stable signal.
    Returns prefill rate (server-side from /metrics) AND decode rate, so the
    test can verify a config change does not regress decode below the 17 t/s
    floor.
    """
    body = {
        "model": model,
        "input": prompt,
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "stream": False,
    }
    m_before = fetch_metrics(host)
    t0 = time.perf_counter()
    req = urllib.request.Request(
        host + "/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode())
    t1 = time.perf_counter()
    m_after = fetch_metrics(host)

    usage = payload.get("usage", {})
    prefill_tokens = usage.get("input_tokens", 0)
    decode_tokens = usage.get("output_tokens", 0)
    server_prefill_s = delta(
        m_before, m_after, "vllm:request_prefill_time_seconds_sum"
    )
    server_decode_s = delta(
        m_before, m_after, "vllm:request_decode_time_seconds_sum"
    )
    wall_s = t1 - t0
    return {
        "wall_s": wall_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "server_prefill_s": server_prefill_s,
        "server_decode_s": server_decode_s,
        "prefill_rate_server": (
            prefill_tokens / server_prefill_s if server_prefill_s > 0 else 0.0
        ),
        "decode_rate_server": (
            decode_tokens / server_decode_s if server_decode_s > 0 else 0.0
        ),
        "prefill_rate_wall": prefill_tokens / wall_s if wall_s > 0 else 0.0,
    }


def color_for(rate: float) -> str:
    if rate >= 200:
        return GREEN
    if rate >= 80:
        return YELLOW
    return RED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # Default 4096 tokens: the size at which the prefill bottleneck reproduces
    # the user's reported 8k pain (~38 t/s) and is fast enough to iterate on.
    # Lower sizes (1k, 2k) are uninteresting because they don't reproduce the
    # bottleneck. See .research/prefill-perf-experiments/CHANGELOG.md.
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    try:
        urllib.request.urlopen(args.host + "/v1/models", timeout=5).read()
    except Exception as e:
        print(f"{RED}engine unreachable at {args.host}: {e}{RESET}", file=sys.stderr)
        sys.exit(2)

    print(f"{BOLD}prefill warmup bench{RESET} -> {CYAN}{args.host}{RESET}")
    print(f"{DIM}building ~{args.target_tokens}-token prompt via /tokenize...{RESET}")
    prompt, n_tokens = build_prompt(args.target_tokens, args.host, args.model)
    print(f"{DIM}actual prompt tokens: {n_tokens}{RESET}\n")

    print(f"{BOLD}{'run':>4} {'pp_tok':>7} {'pp_t/s':>8} {'dec_tok':>8} {'dec_t/s':>8}  {'tag'}{RESET}")
    rows = []
    # Hard floor: anything below 12 t/s decode at seq=1 gets reverted.
    # Expected band is 17-18.5 t/s; peaks above; dips above 12 acceptable.
    decode_floor = 12.0
    for i in range(1, args.runs + 1):
        r = run_one(args.host, args.model, prompt)
        rows.append(r)
        pp = r["prefill_rate_server"]
        dec = r["decode_rate_server"]
        pcol = color_for(pp)
        dcol = GREEN if dec >= decode_floor else RED
        tag = f"{DIM}cold{RESET}" if i == 1 else f"{DIM}warm{RESET}"
        if dec < decode_floor and dec > 0:
            tag += f" {RED}DECODE FLOOR VIOLATED{RESET}"
        print(
            f"{i:>4} {r['prefill_tokens']:>7} {pcol}{pp:>8.1f}{RESET} "
            f"{r['decode_tokens']:>8} {dcol}{dec:>8.1f}{RESET}  {tag}"
        )

    if len(rows) >= 2:
        cold_pp = rows[0]["prefill_rate_server"]
        warm_pp = [r["prefill_rate_server"] for r in rows[1:]]
        warm_pp_avg = sum(warm_pp) / len(warm_pp)
        gap = (warm_pp_avg / cold_pp) if cold_pp > 0 else 0
        decodes = [r["decode_rate_server"] for r in rows if r["decode_rate_server"] > 0]
        decode_avg = sum(decodes) / len(decodes) if decodes else 0
        decode_min = min(decodes) if decodes else 0
        print()
        print(f"{BOLD}prefill cold (run 1):{RESET}     {color_for(cold_pp)}{cold_pp:.1f}{RESET} t/s")
        print(f"{BOLD}prefill warm avg (2-{len(rows)}):{RESET}  {color_for(warm_pp_avg)}{warm_pp_avg:.1f}{RESET} t/s")
        print(f"{BOLD}prefill warm/cold ratio:{RESET}  {gap:.2f}x")
        print()
        dcol = GREEN if decode_min >= decode_floor else RED
        print(
            f"{BOLD}decode floor check:{RESET}  min {dcol}{decode_min:.1f}{RESET} t/s, "
            f"avg {decode_avg:.1f} t/s (floor {decode_floor})"
        )
        if decode_min < decode_floor and decode_min > 0:
            print(f"{RED}{BOLD}REVERT THIS CONFIG.{RESET} Decode dropped below floor.")
        elif decode_min == 0:
            print(f"{YELLOW}decode rate could not be measured (server-side metric was zero).{RESET}")
        else:
            print(f"{GREEN}decode floor preserved.{RESET}")
        print()
        if gap > 1.5:
            print(
                f"{YELLOW}cold-cache effect detected ({gap:.2f}x).{RESET} "
                f"Triton autotune paying real cost on first-shape encounter."
            )
        else:
            print(
                f"{GREEN}no significant cold-cache effect ({gap:.2f}x).{RESET} "
                f"Autotune is amortized; bottleneck is steady-state."
            )


if __name__ == "__main__":
    main()
