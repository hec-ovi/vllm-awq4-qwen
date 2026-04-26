#!/usr/bin/env python3
"""qwen-cli - tiny ollama-style streaming client for vllm-awq4-qwen.

Connects to vLLM (default http://127.0.0.1:8000), streams the response, and
shows precise per-request metrics by snapshotting vLLM's Prometheus /metrics
endpoint before and after the request.

Three rates are reported, each with a clear meaning:

  wall rate - completion_tokens / total wall time. The honest "how fast
                  did the answer appear" number from the user's POV.
  delivery rate - completion_tokens / streaming-window time (post-first-delta).
                  Inflated by server-side <think> buffering - useful only
                  to see the burst rate during visible streaming.
  vLLM rate - vllm:generation_tokens_total delta / vllm:request_decode_time
                  delta from /metrics. The engine's own ground-truth decode
                  speed, unaffected by HTTP buffering or thinking time.

Plus DFlash acceptance metrics from /metrics (drafted, accepted, acceptance %,
mean accepted tokens per round, position-0 acceptance).

Single file, stdlib only, no deps.

Usage:
    .tools/qwen-cli.py                       # interactive REPL
    .tools/qwen-cli.py "explain mitosis"     # one-shot
    .tools/qwen-cli.py --bench               # 5-prompt bench
    .tools/qwen-cli.py --no-thinking "..."   # hide reasoning
    .tools/qwen-cli.py --no-metrics "..."    # skip /metrics scrape
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
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
MAGENTA = "\033[35m"
ORANGE = "\033[38;5;208m"  # Aperture orange


# Aperture Science ASCII logo (community art, traceable through Portal end-credits).
# Used as a tongue-in-cheek REPL banner. Source: blog.slowb.ro/apeture-science-logo-from-portal-in-ascii/
APERTURE_LOGO = r"""
                  .,-:;//;:=,
              . :H@@@MM@M#H/.,+%;,
           ,/X+ +M@@M@MM%=,-%HMMM@X/,
         -+@MM; $M@@MH+-,;XMMMM@MMMM@+-
        ;@M@@M- XM@X;. -+XXXXXHHH@M@M#@/.
      ,%MM@@MH ,@%=             .---=-=:=,.
      =@#@@@MX.,                -%HX$$%%%:;
     =-./@M@M$                   .;@MMMM@MM:
     X@/ -$MM/                    . +MM@@@M$
    ,@M@H: :@:                    . =X#@@@@-
    ,@@@MMX, .                    /H- ;@M@M=
    .H@@@@M@+,                    %MM+..%#$.
     /MMMM@MMH/.                  XM@MH; =;
      /%+%$XHH@$=              , .H@@@@MX,
       .=--------.           -%H.,@@@@@MX,
       .%MM@@@HHHXX$$$%+- .:$MMX =M@@MM%.
         =XMMM@MM@MM#H;,-+HMM@M+ /MMMX=
           =%@M@M#@$-.=$@MM@@@M; %M%=
             ,:+$+-,/H#MMMMMMM@= =,
                  =++%%%%+/:-.
"""

GLADOS_QUOTES = [
    "I'm not even angry. I'm being so sincere right now.",
    "The cake is a lie.",
    "Speedy thing goes in, speedy thing comes out.",
    "This was a triumph. I'm making a note here: HUGE SUCCESS.",
    "I think we can put our differences behind us. For science. You monster.",
    "Goodbye, my only friend. Oh, did you think I meant you? That would be funny if it weren't so sad.",
    "Thank you for participating in this Aperture Science computer-aided enrichment activity.",
]


def _color_for_tps(tps: float) -> str:
    if tps >= 18:
        return GREEN
    if tps >= 10:
        return YELLOW
    return RED


# ----------------------------------------------------------------------- metrics

def fetch_prom_metrics(host: str) -> dict[str, float] | None:
    """Snapshot vLLM's Prometheus /metrics. Returns name -> summed value (across labels)."""
    try:
        with urllib.request.urlopen(host + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `metric_name{labels=...} value [timestamp]`
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


def metric_delta(before: dict | None, after: dict | None, key: str) -> float:
    if not before or not after:
        return 0.0
    return after.get(key, 0.0) - before.get(key, 0.0)


# ----------------------------------------------------------------------- streaming

def stream_chat(prompt, host, model, max_tokens=4096, temperature=0):
    """Stream from /v1/responses (NOT chat/completions, which has a vLLM
    v0.20.0 streaming-reasoning bug - see README's known issues).

    /v1/responses emits typed SSE events:
      response.reasoning_text.delta  → live reasoning tokens
      response.output_text.delta     → live answer tokens
      response.completed             → final, includes usage

    Yields ('reasoning'|'content'|'stats', payload).
    """
    body = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        host + "/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens_seen = 0
    t_start = time.perf_counter()
    t_first = None

    current_event = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\n").rstrip("\r")
            if not line:
                current_event = None
                continue
            if line.startswith("event: "):
                current_event = line[7:].strip()
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            ev = current_event or chunk.get("type")
            if ev == "response.reasoning_text.delta":
                delta = chunk.get("delta") or ""
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    reasoning_tokens_seen += 1
                    yield ("reasoning", delta)
            elif ev == "response.output_text.delta":
                delta = chunk.get("delta") or ""
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    yield ("content", delta)
            elif ev == "response.completed":
                resp_obj = chunk.get("response") or {}
                usage = resp_obj.get("usage") or {}
                prompt_tokens = usage.get("input_tokens") or prompt_tokens
                completion_tokens = usage.get("output_tokens") or completion_tokens

    t_end = time.perf_counter()
    decode_s = (t_end - t_first) if t_first else 0.0
    wall_s = t_end - t_start
    yield (
        "stats",
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens_seen": reasoning_tokens_seen,
            "ttft_s": (t_first - t_start) if t_first else 0.0,
            "decode_s": decode_s,
            "wall_s": wall_s,
            "delivery_tps": completion_tokens / decode_s if decode_s > 0 else 0.0,
            "wall_tps": completion_tokens / wall_s if wall_s > 0 else 0.0,
        },
    )


# ----------------------------------------------------------------------- render

def _per_pos_accept_deltas(before, after):
    """Return list of accepted-tokens-per-position deltas (position 0..N-1)."""
    if not before or not after:
        return []
    out = []
    for pos in range(20):  # generous upper bound; vLLM caps at num_speculative_tokens
        key = f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="Qwen3.6-27B-AWQ4",position="{pos}"}}'
        # Our parse_prom strips labels - that means values are summed across positions, useless.
        # We need to re-parse here including position labels.
        pass  # see fetch_per_pos below
    return out


def fetch_per_pos(host: str) -> dict[int, float]:
    """Specialized parser that keeps the position label intact."""
    try:
        with urllib.request.urlopen(host + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out: dict[int, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
            continue
        # ...{...,position="N"} VALUE
        try:
            lbl_end = line.index("}")
            labels = line[line.index("{") + 1:lbl_end]
            value = float(line[lbl_end + 1:].strip().split()[0])
            for kv in labels.split(","):
                if kv.strip().startswith("position="):
                    pos = int(kv.split("=")[1].strip().strip('"'))
                    out[pos] = out.get(pos, 0.0) + value
                    break
        except (ValueError, IndexError):
            continue
    return out


def print_stats(stats, metrics_before, metrics_after, per_pos_before, per_pos_after, client_estimate=None):
    """One-line summary. Use --verbose for the full breakdown."""
    pt = stats["prompt_tokens"]
    ct = stats["completion_tokens"]
    wall_s = stats["wall_s"]
    wall_tps = stats["wall_tps"]
    color = _color_for_tps(wall_tps)

    parts = [f"{DIM}{pt}→{ct} tok · {wall_s:.2f}s ·{RESET} {color}{BOLD}{wall_tps:.1f} t/s{RESET}"]

    if metrics_before and metrics_after:
        gen_d = metric_delta(metrics_before, metrics_after, "vllm:generation_tokens_total")
        decode_d = metric_delta(metrics_before, metrics_after, "vllm:request_decode_time_seconds_sum")
        accepted_d = metric_delta(metrics_before, metrics_after, "vllm:spec_decode_num_accepted_tokens_total")
        drafted_d = metric_delta(metrics_before, metrics_after, "vllm:spec_decode_num_draft_tokens_total")
        rounds_d = metric_delta(metrics_before, metrics_after, "vllm:spec_decode_num_drafts_total")

        if decode_d > 0 and gen_d > 0:
            vllm_tps = gen_d / decode_d
            parts.append(f"{DIM}vLLM{RESET} {_color_for_tps(vllm_tps)}{vllm_tps:.1f}{RESET}")
        if drafted_d > 0 and rounds_d > 0:
            n_inferred = int(round(drafted_d / rounds_d))
            acc_pct = (accepted_d / drafted_d) * 100
            acc_color = GREEN if acc_pct >= 60 else (YELLOW if acc_pct >= 40 else RED)
            parts.append(f"{DIM}DFlash N={n_inferred} acc{RESET} {acc_color}{acc_pct:.0f}%{RESET}")

    print(f"\n{DIM}---{RESET} " + f" {DIM}·{RESET} ".join(parts))


def render_one(prompt, host, model, show_thinking=True, scrape_metrics=True):
    # Engine reachability probe - give a clear error rather than a stack trace
    try:
        urllib.request.urlopen(host + "/v1/models", timeout=3).read()
    except urllib.error.URLError as e:
        print(f"{RED}cannot reach {host}: {e}{RESET}", file=sys.stderr)
        print(f"{DIM}is the engine up + finished booting? `docker logs -f vllm-awq4-qwen`{RESET}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"{RED}engine probe failed: {e}{RESET}", file=sys.stderr)
        return False

    print(f"{DIM}{BOLD}>{RESET}{DIM} {prompt}{RESET}\n")
    in_thinking = False
    streamed_chunks = 0
    t_request_start = time.perf_counter()
    client_estimate = None

    metrics_before = fetch_prom_metrics(host) if scrape_metrics else None
    per_pos_before = fetch_per_pos(host) if scrape_metrics else None

    try:
        for kind, payload in stream_chat(prompt, host, model):
            if kind == "reasoning":
                if not in_thinking and show_thinking:
                    print(f"{MAGENTA}<thinking>{RESET}\n", end="", flush=True)
                    in_thinking = True
                if show_thinking:
                    print(f"{DIM}{payload}{RESET}", end="", flush=True)
                streamed_chunks += 1
            elif kind == "content":
                if in_thinking:
                    print(f"\n{MAGENTA}</thinking>{RESET}\n", end="", flush=True)
                    in_thinking = False
                print(payload, end="", flush=True)
                streamed_chunks += 1
            elif kind == "stats":
                t_end = time.perf_counter()
                elapsed = t_end - t_request_start
                rate = streamed_chunks / elapsed if elapsed > 0 else 0
                client_estimate = (streamed_chunks, elapsed, rate)
                if in_thinking:
                    print(f"\n{MAGENTA}</thinking>{RESET}")
                    in_thinking = False
                metrics_after = fetch_prom_metrics(host) if scrape_metrics else None
                per_pos_after = fetch_per_pos(host) if scrape_metrics else None
                print_stats(payload, metrics_before, metrics_after, per_pos_before, per_pos_after, client_estimate)
    except KeyboardInterrupt:
        print(f"\n{DIM}(aborted by user){RESET}")
    except urllib.error.URLError as e:
        print(f"\n{RED}Error: {e}{RESET}", file=sys.stderr)
        return False
    return True


def _slow_print(text, color="", per_char_delay=0.001, per_line_delay=0.015):
    """Print with a typewriter effect. Faster per-char, slower between lines."""
    for line in text.splitlines():
        if color:
            sys.stdout.write(color)
        for ch in line:
            sys.stdout.write(ch)
            sys.stdout.flush()
            if ch != " ":
                time.sleep(per_char_delay)
        if color:
            sys.stdout.write(RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()
        time.sleep(per_line_delay)


def repl(host, model, show_thinking, scrape_metrics):
    import random
    _slow_print(APERTURE_LOGO, color=ORANGE, per_char_delay=0.0008, per_line_delay=0.02)
    _slow_print("  APERTURE SCIENCE COMPUTER-AIDED ENRICHMENT CENTER", color=BOLD + ORANGE, per_char_delay=0.005, per_line_delay=0.05)
    _slow_print(f"  Welcome to GLaDOS, powered by Qwen 3.6-27B (AWQ-INT4) + DFlash on AMD Strix Halo.", color=DIM + ORANGE, per_char_delay=0.003, per_line_delay=0.05)
    _slow_print(f"  {random.choice(GLADOS_QUOTES)}", color=DIM, per_char_delay=0.005, per_line_delay=0.05)
    print()
    print(f"  {BOLD}qwen-cli{RESET} -> {CYAN}{host}{RESET} ({DIM}{model}{RESET})")
    try:
        urllib.request.urlopen(host + "/v1/models", timeout=3).read()
    except Exception as e:
        print(f"  {RED}cannot reach {host}: {e}{RESET}")
        print(f"  {DIM}is the engine up? `docker compose up -d` from the repo root{RESET}")
        return
    if scrape_metrics:
        m = fetch_prom_metrics(host)
        if m:
            print(f"  {DIM}/metrics OK - vLLM internal stats will be reported{RESET}")
        else:
            print(f"  {DIM}/metrics unreachable - wall+delivery rates only{RESET}")
    print(f"  {DIM}Ctrl-D / 'exit' / 'quit' to leave. Empty line skipped.{RESET}\n")
    while True:
        try:
            prompt = input(f"{BOLD}{GREEN}>{RESET} ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if prompt.strip() in ("exit", "quit", ":q"):
            break
        if not prompt.strip():
            continue
        render_one(prompt, host, model, show_thinking=show_thinking, scrape_metrics=scrape_metrics)
        print()


def quick_bench(host, model, scrape_metrics):
    prompts = [
        ("haiku",   "Write a haiku about programming."),
        ("math",    "What is 47 * 89? Just the number."),
        ("explain", "Explain photosynthesis in two sentences."),
        ("code",    "Write a Python function that returns the n-th Fibonacci number."),
        ("reason",  "If a train leaves NYC at 60mph and another at 75mph from the same station to Boston (200mi), which arrives first and by how many minutes?"),
    ]
    print(f"{BOLD}qwen-cli quick bench{RESET} -> {CYAN}{host}{RESET}\n")
    rows = []
    for name, p in prompts:
        print(f"{DIM}[{name}]{RESET} {p}")
        text_buf = []
        last = None
        m_before = fetch_prom_metrics(host) if scrape_metrics else None
        try:
            for kind, payload in stream_chat(p, host, model, max_tokens=512):
                if kind == "content":
                    text_buf.append(payload)
                elif kind == "stats":
                    last = payload
        except KeyboardInterrupt:
            print(f"{DIM}(aborted){RESET}")
            break
        m_after = fetch_prom_metrics(host) if scrape_metrics else None
        out = "".join(text_buf).strip()
        snippet = out[:160] + ("..." if len(out) > 160 else "")
        print(f"  {DIM}{snippet}{RESET}")
        if last:
            wall = last["wall_tps"]
            color = _color_for_tps(wall)
            line = f"  {color}{wall:.2f} t/s{RESET} (wall)  {DIM}|{RESET}  {last['delivery_tps']:.1f} t/s (delivery)"
            if m_before and m_after:
                gen_d = metric_delta(m_before, m_after, "vllm:generation_tokens_total")
                dec_d = metric_delta(m_before, m_after, "vllm:request_decode_time_seconds_sum")
                drf_d = metric_delta(m_before, m_after, "vllm:spec_decode_num_draft_tokens_total")
                acc_d = metric_delta(m_before, m_after, "vllm:spec_decode_num_accepted_tokens_total")
                if dec_d > 0 and gen_d > 0:
                    vllm_tps = gen_d / dec_d
                    line += f"  {DIM}|{RESET}  {_color_for_tps(vllm_tps)}{vllm_tps:.1f} t/s{RESET} (vLLM)"
                if drf_d > 0:
                    line += f"  {DIM}|{RESET}  {(acc_d / drf_d * 100):.0f}% acc"
            line += f"  {DIM}({last['completion_tokens']} tok in {last['decode_s']:.1f}s, ttft {last['ttft_s']:.2f}s){RESET}"
            print(line)
            rows.append((name, last["wall_tps"], last["delivery_tps"]))
        print()
    if rows:
        walls = [r[1] for r in rows]
        sw = sorted(walls)
        median = sw[len(sw) // 2]
        mean = sum(walls) / len(walls)
        color = _color_for_tps(median)
        print(f"{BOLD}Wall-rate summary:{RESET} median {color}{median:.2f} t/s{RESET} · mean {mean:.2f} · range {min(walls):.1f}-{max(walls):.1f}")


def main():
    ap = argparse.ArgumentParser(description="Tiny ollama-style CLI for vllm-awq4-qwen")
    ap.add_argument("prompt", nargs="?", help="One-shot prompt (omit for interactive REPL)")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"vLLM URL (default: {DEFAULT_HOST})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--bench", action="store_true", help="Run 5-prompt bench and exit")
    ap.add_argument("--no-thinking", action="store_true", help="Hide reasoning_content from output")
    ap.add_argument("--no-metrics", action="store_true", help="Skip /metrics scrape (faster, less precise)")
    args = ap.parse_args()

    show_thinking = not args.no_thinking
    scrape_metrics = not args.no_metrics

    if args.bench:
        quick_bench(args.host, args.model, scrape_metrics)
    elif args.prompt:
        ok = render_one(args.prompt, args.host, args.model, show_thinking=show_thinking, scrape_metrics=scrape_metrics)
        sys.exit(0 if ok else 1)
    else:
        repl(args.host, args.model, show_thinking, scrape_metrics)


if __name__ == "__main__":
    main()
