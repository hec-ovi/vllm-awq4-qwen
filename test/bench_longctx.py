"""Long-context (~25K token) DFlash test using REAL .research findings as
source material. Asks the model a hard synthesis question that requires
pulling specific facts from at least 3 different findings files. Tests
both throughput at long context AND answer quality.

Run: python3 test/bench_longctx.py
Output: test/bench_longctx_result.json
"""

from __future__ import annotations
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"
TIMEOUT = 1800

RESEARCH = Path("/home/hec/workspace/vllm-awq4-qwen/.research")
OUT = Path(__file__).parent / "bench_longctx_result.json"


# Curated set of FINDINGS files most relevant to the synthesis question.
# Total ~21K tokens of input + the question + reasoning -> closer to 25K total.
SOURCES = [
    "vllm-dflash-prs/FINDINGS.md",
    "vllm-attention-api/FINDINGS.md",
    "dflash-paper-math/FINDINGS.md",
    "rdna35-isa-triton/FINDINGS.md",
    "dflash-ddtree-spark/FINDINGS.md",
]


SYNTHESIS_QUESTION = """
Based ONLY on the research findings provided above, answer ALL three questions below with specific citations. For each fact, name the SOURCE FILE you got it from (e.g. "per vllm-dflash-prs/FINDINGS.md"). Do not invent facts not present in the findings. If a finding is silent on a question, say so explicitly.

Q1. PR #40176 was merged to vllm-project/vllm:main on 2026-04-22 (merge commit 6d09769700) but was NOT included in the v0.20.0 release tag (101584af0). Explain (a) the most likely mechanism by which this happened (release branch logistics), (b) the four files that the PR modified, and (c) for each file, name the specific change required to enable DFlash non-causal attention on gfx1151.

Q2. The DFlash drafter z-lab/Qwen3.6-27B-DFlash has interleaved sliding-window attention layers. Identify TWO distinct correctness bugs that would manifest if vLLM v0.20.0 (without PR #40898) attempted to load this drafter. For each bug, cite the specific code path / function affected.

Q3. On AMD Strix Halo / gfx1151 (RDNA 3.5), explain the relationship between (a) Triton's tl.dot lowering, (b) the WMMA hardware instructions available to gfx1151, and (c) why the ROCM_ATTN backend works without architecture gating but the ROCM_AITER_FA backend does NOT. Cite specific instruction names where applicable.

Be concise. Do not pad.
""".strip()


def post(path, body, timeout=TIMEOUT):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), time.perf_counter() - t0


def build_context() -> str:
    parts = ["# Research findings for synthesis question\n"]
    for src in SOURCES:
        path = RESEARCH / src
        parts.append(f"\n\n========================================\n## SOURCE FILE: {src}\n========================================\n")
        parts.append(path.read_text())
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    context = build_context()
    prompt = f"{context}\n\n========================================\n## QUESTION\n========================================\n\n{SYNTHESIS_QUESTION}"
    chars = len(prompt)
    est_tokens = chars // 4
    print(f"Context: {chars:,} chars, ~{est_tokens:,} tokens (4 chars/token rough estimate)")
    print(f"Sources: {len(SOURCES)} findings files")
    print(f"Asking 3 synthesis questions, max_tokens={args.max_tokens}")
    print()

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }

    t0 = time.perf_counter()
    data, wall = post("/v1/chat/completions", body, timeout=TIMEOUT)
    t1 = time.perf_counter()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = data.get("usage", {})
    pt = u.get("prompt_tokens") or 0
    ct = u.get("completion_tokens") or 0
    finish = data["choices"][0].get("finish_reason")

    decode_tps = ct / wall if wall else 0
    # Effective end-to-end (includes prefill of 25K tokens):
    e2e_tps = ct / (t1 - t0) if (t1 - t0) else 0

    print(f"=== Long-context result ===")
    print(f"  prompt_tokens (actual): {pt}")
    print(f"  completion_tokens:     {ct}")
    print(f"  reasoning_chars:       {len(reasoning)}")
    print(f"  wall:                  {wall:.2f}s")
    print(f"  finish_reason:         {finish}")
    print(f"  decode t/s:            {decode_tps:.2f}")
    print(f"  e2e t/s:               {e2e_tps:.2f}")
    print()
    print("=== Reasoning (first 1000 chars) ===")
    print(reasoning[:1000])
    print()
    print("=== Answer (first 2000 chars) ===")
    print(content[:2000])

    OUT.write_text(json.dumps({
        "prompt_chars": chars,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_chars": len(reasoning),
        "wall_seconds": round(wall, 2),
        "decode_tps": round(decode_tps, 3),
        "e2e_tps": round(e2e_tps, 3),
        "finish_reason": finish,
        "reasoning": reasoning,
        "content": content,
        "sources_used": SOURCES,
        "question": SYNTHESIS_QUESTION,
    }, indent=2))
    print(f"\n=== Saved to {OUT} ===")


if __name__ == "__main__":
    main()
