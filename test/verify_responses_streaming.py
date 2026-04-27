"""
Streaming verification for /v1/responses with Patch 15 (enable_thinking=false).

Goal: prove that on the streaming path, with reasoning disabled, output lands on
the correct SSE channel and tool calls are parsed into function_call items.

Tests:
  T1  tiny prompt, no tools, think_off       -> expect 0 reasoning, content in output_text
  T2  tiny prompt, with tools, think_off     -> expect 0 reasoning, function_call item fires
  T3  ~2K context, no tools, think_off       -> expect 0 reasoning, content in output_text
  T4  ~2K context, with tools, think_off     -> expect 0 reasoning, function_call item fires
  T5  CONTROL: tiny, no tools, think_ON      -> expect nonzero reasoning, content in output_text

Run:
  python test/verify_responses_streaming.py
"""
import json
import time
import urllib.request

HOST = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-27B-AWQ4"

# /v1/responses uses flat FunctionTool schema (no "function" wrapper).
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
            "units": {"type": "string", "enum": ["c", "f"], "default": "c"},
        },
        "required": ["city"],
    },
}

LONG_CONTEXT = """The Strix Halo platform (also known as Ryzen AI Max+ 395) is AMD's
mobile workstation APU combining 16 Zen 5 CPU cores with the Radeon 8060S iGPU
based on the RDNA 3.5 architecture (gfx1151 ISA), backed by up to 128 GB of
unified LPDDR5X-8000 memory accessible to both the CPU and GPU.

The unified memory architecture means the iGPU can address the same physical
DRAM as the CPU, eliminating the host-to-device transfer overhead that
discrete GPUs incur. For LLM inference, this enables loading models that
exceed traditional GPU VRAM budgets - a 27B parameter model in 4-bit AWQ
quantization (~16 GB weights + ~24 GB KV cache at 128K context) fits easily
within the platform's memory budget.

Software stack on Linux: TheRock ROCm 7.13 nightly tarballs include gfx1151
support that is not yet present in the official ROCm 6.x releases. The vLLM
0.20.0 source tree requires custom patches (collected in patch_strix.py) to
enable proper attention backend selection (ROCM_ATTN), Triton autotuning
configurations specific to gfx1151, and DFlash speculative decoding paths.

Performance characteristics observed: prefill bandwidth peaks at 400 tokens/sec
on short contexts and degrades to 33-38 tokens/sec on multi-thousand-token
prompts. Decode throughput is approximately 14-18 tokens/sec single-stream and
13.5 tokens/sec/stream under three concurrent streams. Cold boot of the vLLM
container averages 9 minutes due to model load (95s), profile run / autotuning
(6-7 min), and final server startup (5s); only ~30s is reclaimed by the Triton
JIT cache on rewarm. The MIOpen perf cache is not host-mounted, so its state
does not persist across container recreations.

For the Qwen3.6-27B-AWQ-INT4 model specifically, DFlash speculative decoding
with the z-lab/Qwen3.6-27B-DFlash drafter at 8 speculative tokens provides a
modest decode-rate uplift on accepted continuations, particularly for the kind
of repetitive structured output common in tool-calling agent workloads. The
acceptance rate varies considerably by prompt class: long literal-passage
recall sits near the upper bound, free-form prose generation sits in the
middle, and complex reasoning chains sit lower as the drafter struggles to
predict deliberative token paths.

KV cache memory accounting at 128K context with max_num_seqs=3: the available
pool is 23.61 GiB after model load, which divides into roughly 7.87 GiB per
sequence. This is sufficient headroom to allow occasional bursts to the full
128K window on a single sequence without forcing eviction, provided the other
two sequences remain at moderate context lengths. With max_num_seqs=1 the
single sequence has access to the full 23.61 GiB and can comfortably handle
the full 131072-token window with significant slack.

The chat template for Qwen3.6 uses a token-prefill mechanism for disabling
reasoning: when enable_thinking is false, the template emits a closed
<think></think> block as part of the assistant prompt prefix, before the model
begins generation. This means the model literally cannot emit reasoning tokens
because the context already contains the closure - generation begins past the
reasoning region. The reasoning parser in vLLM (Qwen3ReasoningParser) detects
this state through the prompt_is_reasoning_end flag on chat_completion's code
path, but the corresponding wiring on the responses API path was missing
prior to local Patch 15."""


def post_stream(path, body, timeout=300):
    """POST and stream SSE events. Yields (event_name, data_dict) tuples."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    event_name = None
    data_buf = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        if line == "":
            if event_name or data_buf:
                payload = "\n".join(data_buf)
                if payload == "[DONE]":
                    return
                try:
                    parsed = json.loads(payload) if payload else {}
                except Exception:
                    parsed = {"_raw": payload}
                yield event_name or parsed.get("type"), parsed
            event_name = None
            data_buf = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buf.append(line[len("data:"):].lstrip())


def run_test(label, body, expect_reasoning=False, expect_tool=False):
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    print(f"  endpoint: /v1/responses  stream=True  "
          f"enable_thinking={body.get('chat_template_kwargs', {}).get('enable_thinking', '(default)')}")
    print(f"  prompt   : {body['input'][:80]!r}{'...' if len(body['input']) > 80 else ''}")
    print(f"  tools    : {len(body.get('tools', []))} tool(s)")

    event_counts = {}
    reasoning_chars = 0
    output_text_chars = 0
    output_text_buf = []
    reasoning_buf = []
    function_calls = []
    output_items = []

    t0 = time.time()
    first_event_t = None
    try:
        for ev_name, data in post_stream("/v1/responses", body, timeout=300):
            if first_event_t is None:
                first_event_t = time.time()
            ev = ev_name or data.get("type", "<unknown>")
            event_counts[ev] = event_counts.get(ev, 0) + 1

            if ev == "response.reasoning_text.delta":
                d = data.get("delta", "")
                reasoning_chars += len(d)
                reasoning_buf.append(d)
            elif ev == "response.output_text.delta":
                d = data.get("delta", "")
                output_text_chars += len(d)
                output_text_buf.append(d)
            elif ev == "response.output_item.added":
                item = data.get("item", {})
                output_items.append(("added", item.get("type"), item))
            elif ev == "response.output_item.done":
                item = data.get("item", {})
                output_items.append(("done", item.get("type"), item))
                if item.get("type") == "function_call":
                    function_calls.append({
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    })
            elif ev == "response.completed":
                pass
    except Exception as e:
        print(f"  !! ERROR streaming: {type(e).__name__}: {e}")
        return

    wall = time.time() - t0
    ttft = (first_event_t - t0) if first_event_t else None

    print(f"\n  wall: {wall:.2f}s  TTFT: {ttft:.2f}s" if ttft else f"\n  wall: {wall:.2f}s")
    print(f"  reasoning channel : {reasoning_chars:>5} chars")
    print(f"  output_text       : {output_text_chars:>5} chars")
    print(f"  function_calls    : {len(function_calls)}")
    if function_calls:
        for fc in function_calls:
            print(f"     -> {fc['name']}({fc['arguments']})")

    item_types_seen = sorted(set(t for _, t, _ in output_items if t))
    print(f"  item types seen   : {item_types_seen}")
    print(f"  event types seen  : {sorted(event_counts.keys())}")

    if reasoning_buf:
        preview = "".join(reasoning_buf)[:200].replace("\n", "\\n")
        print(f"  reasoning preview : {preview!r}")
    if output_text_buf:
        preview = "".join(output_text_buf)[:200].replace("\n", "\\n")
        print(f"  output_text view  : {preview!r}")

    verdict = "PASS"
    notes = []
    if expect_reasoning and reasoning_chars == 0:
        verdict = "FAIL"; notes.append("expected reasoning, got zero")
    if (not expect_reasoning) and reasoning_chars > 0:
        verdict = "FAIL"; notes.append(f"expected zero reasoning, got {reasoning_chars}")
    if expect_tool and not function_calls:
        verdict = "FAIL"; notes.append("expected function_call item, none parsed")
    if (not expect_tool) and output_text_chars == 0:
        verdict = "WARN"; notes.append("no output_text content emitted")
    if (not expect_tool) and "<tool_call>" in "".join(output_text_buf + reasoning_buf):
        verdict = "FAIL"; notes.append("raw <tool_call> XML leaked into a text channel")

    print(f"\n  VERDICT: {verdict}{'  -- ' + '; '.join(notes) if notes else ''}")


def main():
    THINK_OFF = {"enable_thinking": False}

    # T1 - tiny, no tools, think_off
    run_test(
        "T1  tiny | no tools | think_off",
        {
            "model": MODEL,
            "input": "What is 19 + 23? Answer with just the number.",
            "stream": True,
            "chat_template_kwargs": THINK_OFF,
        },
        expect_reasoning=False, expect_tool=False,
    )

    # T2 - tiny, with tools, think_off
    run_test(
        "T2  tiny | with tools | think_off",
        {
            "model": MODEL,
            "input": "What's the weather in Tokyo right now? Use the tool.",
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "stream": True,
            "chat_template_kwargs": THINK_OFF,
        },
        expect_reasoning=False, expect_tool=True,
    )

    # T3 - 2K context, no tools, think_off
    run_test(
        "T3  ~2K context | no tools | think_off",
        {
            "model": MODEL,
            "input": (
                LONG_CONTEXT
                + "\n\nBased on the passage above, answer in one short sentence: "
                  "what is the cold-boot time of the vLLM container, and what dominates it?"
            ),
            "stream": True,
            "chat_template_kwargs": THINK_OFF,
        },
        expect_reasoning=False, expect_tool=False,
    )

    # T4 - 2K context, with tools, think_off
    run_test(
        "T4  ~2K context | with tools | think_off",
        {
            "model": MODEL,
            "input": (
                LONG_CONTEXT
                + "\n\nThe user wants to know the current weather in the city where AMD's "
                  "headquarters is located (Santa Clara). Use the tool to look it up."
            ),
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "stream": True,
            "chat_template_kwargs": THINK_OFF,
        },
        expect_reasoning=False, expect_tool=True,
    )

    # T5 - CONTROL: tiny, no tools, think_ON (default)
    run_test(
        "T5  CONTROL: tiny | no tools | think_ON",
        {
            "model": MODEL,
            "input": "What is 19 + 23? Answer with just the number.",
            "stream": True,
        },
        expect_reasoning=True, expect_tool=False,
    )


if __name__ == "__main__":
    main()
