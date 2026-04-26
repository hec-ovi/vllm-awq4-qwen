"""Comprehensive DFlash bench  -  exercises all 5 endpoints + tool calling on
/v1/responses + a real Three.js codegen task. Generic, not user-specific.

Captures wall-clock t/s per run, computes mean / median / p95 per test, saves
full responses as JSON + the codegen output as a runnable .html file.

Run: python3 test/bench_full.py
Output: test/bench_full_results.json + test/bench_full_threejs.html
"""

from __future__ import annotations
import argparse
import base64
import json
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"
BACKEND_LABEL = "ROCM_ATTN + DFlash"  # filled in below from /v1/models meta
TIMEOUT = 1800

IMG_DIR = Path("/home/hec/Pictures")
IMAGE_A = IMG_DIR / "frost_1.png"
IMAGE_B = IMG_DIR / "splash.png"

OUT_JSON = Path(__file__).parent / "bench_full_results.json"
OUT_THREEJS = Path(__file__).parent / "bench_full_threejs.html"


def post(path: str, body: dict, timeout: int = TIMEOUT):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    t1 = time.perf_counter()
    return json.loads(raw), t1 - t0


def post_safe(path, body, timeout=TIMEOUT):
    try:
        return post(path, body, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:500], "status": e.code}, 0
    except Exception as e:
        return {"error": str(e)[:500]}, 0


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "max": round(max(values), 3),
        "p95": round(sorted(values)[max(0, int(len(values) * 0.95) - 1)], 3) if len(values) > 1 else round(values[0], 3),
    }


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def run_test(name: str, runs: int, fn) -> dict:
    print(f"\n=== {name} ({runs} run{'s' if runs != 1 else ''}) ===")
    rs = []
    for i in range(runs):
        rec = fn()
        rs.append(rec)
        tps = rec.get("tps", 0)
        ct = rec.get("completion_tokens", 0)
        wall = rec.get("wall", 0)
        err = rec.get("error")
        if err:
            print(f"  [run {i+1}] ERROR: {err[:200]}")
        else:
            print(f"  [run {i+1}] prompt={rec.get('prompt_tokens',0)} completion={ct} wall={wall:.2f}s -> {tps:.2f} t/s")
    tps_vals = [r["tps"] for r in rs if "tps" in r and not r.get("error")]
    ct_vals = [r["completion_tokens"] for r in rs if not r.get("error")]
    return {
        "name": name,
        "runs": rs,
        "tps_stats": stats(tps_vals),
        "completion_token_stats": stats(ct_vals) if ct_vals else {},
    }


# ---------- test definitions ----------

def test_completions_short():
    body = {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
    data, wall = post_safe("/v1/completions", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    text = (data["choices"][0].get("text") or "").strip()
    u = data.get("usage", {})
    pt = u.get("prompt_tokens") or 0
    ct = u.get("completion_tokens") or 0
    return {"output": text, "wall": wall, "tps": ct / wall if wall else 0, "completion_tokens": ct, "prompt_tokens": pt}


def test_chat_factual():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the speed of light in m/s? Answer with just the number, no commentary."}],
        "temperature": 0,
    }
    data, wall = post_safe("/v1/chat/completions", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = data.get("usage", {})
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return {"output": content[:800], "reasoning": reasoning[:400], "wall": wall, "tps": ct / wall if wall else 0,
            "completion_tokens": ct, "prompt_tokens": pt}


def test_chat_explainer():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain quantum entanglement to a 12-year-old in three sentences."}],
        "temperature": 0,
    }
    data, wall = post_safe("/v1/chat/completions", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = data.get("usage", {})
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return {"output": content[:1200], "reasoning": reasoning[:600], "wall": wall, "tps": ct / wall if wall else 0,
            "completion_tokens": ct, "prompt_tokens": pt}


def test_responses_reasoning():
    body = {
        "model": MODEL,
        "input": "If three trains leave New York simultaneously heading to Boston 200 miles away at 60 mph, 75 mph, and 90 mph, in what order do they arrive? Answer with just an ordered list.",
        "temperature": 0,
    }
    data, wall = post_safe("/v1/responses", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    out_text = []
    for item in data.get("output", []):
        if isinstance(item, dict):
            for c in item.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    t = c.get("text") or ""
                    if t:
                        out_text.append(t)
    u = data.get("usage", {})
    pt, ct = u.get("input_tokens") or 0, u.get("output_tokens") or 0
    return {"output": "\n".join(out_text)[:1200], "wall": wall, "tps": ct / wall if wall else 0,
            "completion_tokens": ct, "prompt_tokens": pt}


def test_vision(img_path: Path, prompt: str, name_suffix: str):
    if not img_path.exists():
        return {"error": f"image not found: {img_path}", "wall": 0, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    b64 = img_b64(img_path)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0,
    }
    data, wall = post_safe("/v1/chat/completions", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0,
                "image": str(img_path), "prompt": prompt}
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    u = data.get("usage", {})
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return {"image": str(img_path), "prompt": prompt, "output": content[:1500],
            "wall": wall, "tps": ct / wall if wall else 0, "completion_tokens": ct, "prompt_tokens": pt}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["c", "f"], "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
}


def test_tool_chat():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the current weather in Tokyo? Use the get_weather tool with units=c."}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "stream": False,
    }
    data, wall = post_safe("/v1/chat/completions", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    msg = data["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    u = data.get("usage", {})
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    return {
        "tool_calls": [{"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls],
        "tool_call_count": len(tool_calls),
        "content": content[:300],
        "wall": wall, "tps": ct / wall if wall else 0, "completion_tokens": ct, "prompt_tokens": pt,
    }


def test_tool_responses():
    body = {
        "model": MODEL,
        "input": "What is the current weather in Paris? Use the get_weather tool with units=c.",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": WEATHER_TOOL["function"]["description"],
            "parameters": WEATHER_TOOL["function"]["parameters"],
        }],
        "tool_choice": "auto",
        "temperature": 0,
    }
    data, wall = post_safe("/v1/responses", body)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    output_items = data.get("output", []) or []
    function_calls = []
    text_chunks = []
    for item in output_items:
        if isinstance(item, dict):
            t = item.get("type")
            if t in ("function_call", "tool_call"):
                function_calls.append({
                    "name": item.get("name") or item.get("function", {}).get("name"),
                    "arguments": item.get("arguments") or item.get("function", {}).get("arguments"),
                })
            elif t == "message":
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        text_chunks.append(c.get("text") or "")
    u = data.get("usage", {})
    pt = u.get("input_tokens") or 0
    ct = u.get("output_tokens") or 0
    return {
        "function_calls": function_calls,
        "function_call_count": len(function_calls),
        "text": "\n".join(text_chunks)[:300],
        "wall": wall, "tps": ct / wall if wall else 0, "completion_tokens": ct, "prompt_tokens": pt,
    }


THREEJS_PROMPT = """Write a complete, single-file HTML document that uses Three.js (loaded from a CDN) to render a Minecraft-style voxel world with the following features:

1. A 16x16 patch of voxel terrain with simple procedural height variation (use a sine/cosine combination for the height function  -  no external noise libraries).
2. Three different block colors based on height (e.g. stone, dirt, grass).
3. First-person camera with WASD movement and mouse-look using Pointer Lock.
4. Basic ambient + directional lighting and a sky-blue background.
5. The page should run standalone when saved as an .html file and opened in a browser.

Output ONLY the complete HTML document, starting with <!DOCTYPE html> and ending with </html>. No prose, no markdown fences, no explanation."""


def test_codegen_threejs():
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": THREEJS_PROMPT}],
        "temperature": 0,
        "max_tokens": 4096,
    }
    data, wall = post_safe("/v1/chat/completions", body, timeout=1800)
    if "error" in data:
        return {"error": data["error"], "wall": wall, "tps": 0, "completion_tokens": 0, "prompt_tokens": 0}
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = data.get("usage", {})
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    # Attempt to extract HTML doc out of model output
    html = content
    if "<!DOCTYPE" in content:
        i = content.find("<!DOCTYPE")
        j = content.rfind("</html>")
        if j != -1:
            html = content[i:j + len("</html>")]
        else:
            html = content[i:]
    OUT_THREEJS.write_text(html)
    return {
        "output_chars": len(content),
        "reasoning_chars": len(reasoning),
        "wall": wall,
        "tps": ct / wall if wall else 0,
        "completion_tokens": ct,
        "prompt_tokens": pt,
        "starts_with_doctype": content.lstrip().startswith("<!DOCTYPE"),
        "contains_threejs_import": "three" in content.lower() and ("cdn" in content.lower() or "module" in content.lower()),
        "ends_with_html": content.rstrip().endswith("</html>"),
        "saved_to": str(OUT_THREEJS),
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="runs per non-codegen test")
    args = ap.parse_args()

    print(f"=== bench_full.py  -  {BACKEND_LABEL}  -  runs/test={args.runs} ===")
    print(f"Host: {HOST}  Model: {MODEL}")
    t_start = time.time()

    results = {
        "meta": {
            "host": HOST,
            "model": MODEL,
            "backend": BACKEND_LABEL,
            "runs_per_test": args.runs,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tests": {},
    }

    results["tests"]["completions_short"] = run_test("completions short factual", args.runs, test_completions_short)
    results["tests"]["chat_factual"] = run_test("chat factual (speed of light)", args.runs, test_chat_factual)
    results["tests"]["chat_explainer"] = run_test("chat explainer (entanglement)", args.runs, test_chat_explainer)
    results["tests"]["responses_reasoning"] = run_test("responses reasoning (trains)", args.runs, test_responses_reasoning)
    results["tests"]["vision_frost"] = run_test(
        "vision (frost_1.png  -  1280x720)", args.runs,
        lambda: test_vision(IMAGE_A, "Describe this scene in two sentences. What is the dominant color and the apparent setting?", "frost"),
    )
    results["tests"]["vision_splash"] = run_test(
        "vision (splash.png  -  1024x1024)", args.runs,
        lambda: test_vision(IMAGE_B, "What objects can you identify in this image? Reply as a bulleted list.", "splash"),
    )
    results["tests"]["tool_chat"] = run_test("tool calling /v1/chat/completions", args.runs, test_tool_chat)
    results["tests"]["tool_responses"] = run_test("tool calling /v1/responses", args.runs, test_tool_responses)
    # codegen single run only  -  long output, no statistical value in repeats
    print("\n=== codegen Three.js minecraft-style (1 run, may take 1-3 min) ===")
    cg = test_codegen_threejs()
    print(f"  chars={cg.get('output_chars')} thinking={cg.get('reasoning_chars')} "
          f"completion_tokens={cg.get('completion_tokens')} wall={cg.get('wall'):.2f}s -> "
          f"{cg.get('tps', 0):.2f} t/s  doctype={cg.get('starts_with_doctype')} "
          f"</html>={cg.get('ends_with_html')}")
    results["tests"]["codegen_threejs"] = {"name": "codegen Three.js minecraft", "runs": [cg],
                                           "tps_stats": stats([cg["tps"]] if cg.get("tps") else [])}

    results["meta"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["meta"]["wall_seconds"] = round(time.time() - t_start, 2)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n=== DONE in {results['meta']['wall_seconds']:.1f}s  -  saved to {OUT_JSON} ===")
    print(f"=== Three.js HTML saved to {OUT_THREEJS} ===")
    return results


if __name__ == "__main__":
    main()
