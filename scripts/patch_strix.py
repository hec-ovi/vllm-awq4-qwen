"""
Strix Halo (gfx1151) patch bundle for vLLM source builds.

This is the same patch script the sibling `vllm-qwen` repo ships, kept
in sync verbatim. Two reasons it lives here unchanged:

  1. Every patch is gfx1151-driven, not quant-driven. The AWQ-INT4
     model exercises the same RDNA 3.5 code paths as the BF16 model
     (same custom_ops registration, same AITER overrides, same APU
     VRAM clamp), so the patch surface is identical.

  2. AWQ INT4 dispatches through vLLM's compressed-tensors kernel,
     which is itself unaffected by these patches. The DeltaNet linear-
     attention layers are mixed-precision (FP16/BF16 weights kept
     unquantized in the AWQ checkpoint, see README "DeltaNet under
     AWQ" note), so they take the standard linear-attn path the
     compressed-tensors loader already handles.

If a future tool-call / reasoning-parser PR is needed before merging
upstream, add it here as Patch 13/14/15 (cherry-pick of vllm#40783,
#40785, #40787) and rebuild  -  the existing patch numbers stay stable
so cross-repo references don't drift.
"""
import sys
import re
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 1: vllm/platforms/__init__.py (amdsmi monkey patch  -  PROVEN working for 5 months)
    # Comment out real amdsmi imports and replace with pass stubs.
    # The actual amdsmi library doesn't work on Strix Halo APUs in containers.
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (amdsmi disabled, is_rocm forced True)")

    # Patch 1.5: vllm/platforms/rocm.py (MagicMock amdsmi + force gfx1151)
    # Prepend MagicMock so any remaining amdsmi references in rocm.py silently succeed.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add MagicMock header if not already present
        if 'sys.modules["amdsmi"] = MagicMock()' not in txt:
            header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (MagicMock amdsmi + forced gfx1151)")

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()

        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform",
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend is_aiter_found_and_supported
        if "or on_gfx1x()" not in txt:
            txt = txt.replace("import on_mi3xx", "import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")

        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n',
                r'\1 False\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER RMSNorm on gfx1x (CUDA Graph hang)
        if "is_rmsnorm_enabled" in txt:
            txt = re.sub(
                r'(def is_rmsnorm_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._RMSNORM_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support, FP8 linear empty, MoE disabled)")

    # Patch 3: rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms.rocm import on_mi3xx", "from vllm.platforms.rocm import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            p_fa.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

    # Patch 3.5: unquantized.py (Hard-block AITER MoE forced override on gfx1x)
    p_unquant = Path('vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    if p_unquant.exists():
        txt = p_unquant.read_text()
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                'if envs.is_set("VLLM_ROCM_USE_AITER")',
                'from vllm.platforms.rocm import on_gfx1x\n    if envs.is_set("VLLM_ROCM_USE_AITER")'
            )
            txt = txt.replace(
                'if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:',
                'if getattr(on_gfx1x, "__call__", lambda: False)() or not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:'
            )
            p_unquant.write_text(txt)
            print(" -> Patched unquantized.py (Blocked AITER MoE override on gfx1x)")


    # Patch 5: custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()

        # Legacy vLLM < 0.19 fallback
        if "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")" in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not getattr(self, 'on_gfx1x', lambda: False)():\n            custom_ops.append(\"+rms_norm\")"
            )

        # Modern vLLM 0.19+ struct (compilation_config.custom_ops)
        elif "compilation_config.custom_ops.append(\"+rms_norm\")" in txt:
            if "if not getattr(self, \"on_gfx1x\", lambda: False)():" not in txt:
                txt = re.sub(
                    r'(\s+)compilation_config\.custom_ops\.append\("\+rms_norm"\)',
                    r'\1if not getattr(self, "on_gfx1x", lambda: False)():\n\1    compilation_config.custom_ops.append("+rms_norm")',
                    txt
                )

        # Modern vLLM 0.19.2rc1+ IrOpPriorityConfig bypass
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )

        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops & IrOpPriorityConfig rms_norm bypassed on gfx1x)")

    # Patch 6: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (duplicate pattern bypass)
    p_fusion = Path('vllm/compilation/passes/fusion/rocm_aiter_fusion.py')
    if p_fusion.exists():
        txt = p_fusion.read_text()
        if "skip_duplicates=True" not in txt:
            txt = re.sub(
                r"(pm\.register_replacement\s*\((?:(?!\bpm\.register_replacement\b).)*?)pm_pass(\s*[\),])",
                r"\1pm_pass, skip_duplicates=True\2",
                txt, flags=re.DOTALL
            )
            p_fusion.write_text(txt)
            print(" -> Patched vllm/compilation/passes/fusion/rocm_aiter_fusion.py (skip_duplicates)")

    # Patch 7: Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):",
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

    # Patch 7: aiter JIT path fix  -  aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 8: flash_attn_interface.py  -  make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    hard_import_bare = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"

    def _patch_flash_interface(fa_iface):
        txt = fa_iface.read_text()
        if hard_import_bare not in txt or "except (ImportError" in txt:
            return False
        # Detect indentation of the original import line
        m = re.search(r'^( *)' + re.escape(hard_import_bare), txt, re.MULTILINE)
        if not m:
            return False
        indent = m.group(1)
        original_line = indent + hard_import_bare
        soft_import = (
            f"{indent}try:\n"
            f"{indent}    {hard_import_bare}\n"
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError):\n"
            f"{indent}    flash_attn_gpu = None"
        )
        txt = txt.replace(original_line, soft_import)
        fa_iface.write_text(txt)
        print(f" -> Patched {fa_iface} (aiter import made resilient)")
        return True

    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                _patch_flash_interface(fa_iface)
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            _patch_flash_interface(fa_iface)

    # Patch 9: Allow Triton MoE kernels on gfx11xx (Strix Halo)
    # vLLM recently capped MXFP4 Triton MoE kernels to < (11, 0) which excludes RDNA3.5 (11.x)
    for p_triton in [
        Path('vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py'),
        Path('vllm/model_executor/layers/fused_moe/oracle/mxfp4.py')
    ]:
        if p_triton.exists():
            txt = p_triton.read_text()
            if "cap.minor) < (11, 0)" in txt:
                txt = txt.replace("cap.minor) < (11, 0)", "cap.minor) < (12, 0)")
            if "capability() < (11, 0)" in txt:
                txt = txt.replace("capability() < (11, 0)", "capability() < (12, 0)")
            p_triton.write_text(txt)
            print(f" -> Patched {p_triton} (Triton MoE on gfx11xx)")

    # Patch 10: ROCM-21812 APU VRAM Dynamic Margin Patch
    # Explanation: ROCm nightly builds introduced a 50% APU VRAM clamp to prevent
    # OOM kernel panics on headless hosts. This broke vLLM large model loading.
    # This patch intercepts PyTorch memory bounds and dynamically proxies the
    # real amdgpu hardware GTT limits, minus a strict 8GB OS safety margin.
    # By symmetrically carving the OS margin from the top of the GTT ceiling,
    # vLLM's memory profiler allocates flawlessly while guaranteeing the OS stays alive,
    # regardless of the specific GTT allocation size on the host.
    # Ref: https://github.com/ROCm/rocm-systems/pull/5113
    # TODO: Remove this patch block entirely once PR #5113 merges and is
    # incorporated into the ROCm nightly tarballs used by this toolbox.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if "_patched_mem_info" not in txt:
            mem_patch = '''
# --- ROCM-21812 VRAM DYNAMIC PATCH ---
import torch
import glob
import os

try:
    _orig_mem_info = torch.cuda.mem_get_info
    _orig_get_dev_prop = torch.cuda.get_device_properties

    class MockCudaDeviceProperties:
        def __init__(self, prop, override_total):
            self._prop = prop
            self.total_memory = override_total
        def __getattr__(self, name):
            return getattr(self._prop, name)
        def __dir__(self):
            return dir(self._prop)

    def _patched_mem_info(device=None):
        free, total = _orig_mem_info(device)
        try:
            # On APUs, ROCm clamps total to 50% limit. We need the real GTT limits.
            if total < 70 * 1024**3:
                drm_cards = glob.glob('/sys/class/drm/card*/device/mem_info_gtt_total')
                if drm_cards:
                    card_dir = os.path.dirname(drm_cards[0])
                    with open(os.path.join(card_dir, 'mem_info_gtt_total'), 'r') as f:
                        gtt_total = int(f.read().strip())
                    with open(os.path.join(card_dir, 'mem_info_gtt_used'), 'r') as f:
                        gtt_used = int(f.read().strip())

                    # Symmetrically carve 8GB off the TOP of the device perfectly.
                    safe_ceiling = gtt_total - (8 * 1024**3)

                    real_total = safe_ceiling
                    real_free = max(0, safe_ceiling - gtt_used)

                    total = max(total, real_total)
                    free = real_free
        except Exception as e:
            pass
        return int(free), int(total)

    def _patched_get_dev_prop(device=None):
        prop = _orig_get_dev_prop(device)
        free, total = _patched_mem_info(device)
        if hasattr(prop, 'total_memory') and prop.total_memory < total:
            return MockCudaDeviceProperties(prop, total)
        return prop

    torch.cuda.mem_get_info = _patched_mem_info
    torch.cuda.get_device_properties = _patched_get_dev_prop
except Exception:
    pass
# ---------------------------
'''
            txt = mem_patch + txt
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (ROCM-21812 APU VRAM Dynamic Margin)")

    # Patch 11 (local addition): silence hipCtx* deprecation warnings in
    # csrc/cumem_allocator_compat.h. vLLM still uses hipCtxGetCurrent /
    # hipCtxSetCurrent / hipDevicePrimaryCtxRetain for CUDA-compat context
    # management; HIP marked these deprecated but there is no clean
    # replacement for the use case, and upstream vLLM hasn't migrated yet.
    # Suppressing the warning class for that file keeps our build clean.
    p_cumem = Path('csrc/cumem_allocator_compat.h')
    if p_cumem.exists():
        txt = p_cumem.read_text()
        marker = '#pragma clang diagnostic ignored "-Wdeprecated-declarations"'
        if marker not in txt:
            txt = marker + "\n" + txt
            p_cumem.write_text(txt)
            print(" -> Patched csrc/cumem_allocator_compat.h (suppress hipCtx* deprecations)")

    # Patch 12 (local): allow transformers' GGUF parser to accept Qwen 3.5/3.6
    # arch tag. Upstream transformers registers "qwen2", "qwen3", "qwen3_moe"
    # but not "qwen35" (the arch tag Unsloth's Qwen 3.6 GGUFs declare). vLLM
    # has a Qwen3_5ForConditionalGeneration class downstream that handles the
    # actual model correctly; we just need transformers' parser to route the
    # GGUF through as if it were qwen3 so the config loads. Harmless for the
    # AWQ path (no GGUF involved) but kept for parity with the BF16 sibling.
    import site as _site
    for _sp in _site.getsitepackages():
        gguf_utils = Path(_sp) / "transformers/modeling_gguf_pytorch_utils.py"
        if gguf_utils.exists():
            _txt = gguf_utils.read_text()
            _marker = 'elif "minimax-m2" in architecture:'
            _inject = (
                'elif "qwen35" in architecture or "qwen3_5" in architecture:\n'
                '        updated_architecture = "qwen3"\n'
                '    '
            )
            if 'qwen35' not in _txt and _marker in _txt:
                _txt = _txt.replace(_marker, _inject + _marker, 1)
                gguf_utils.write_text(_txt)
                print(f" -> Patched {gguf_utils} (qwen35 -> qwen3 alias for GGUF parser)")

    # Patch 13 (local cherry-pick of vLLM PR #40176): "[ROCm] Support non-causal
    # attention in ROCM_ATTN". Merged to vllm-project/vllm:main on
    # 2026-04-22T03:57Z (merge commit 6d09769700) but NOT cherry-picked into
    # the v0.20.0 release tag (101584af0, cut 2026-04-23). This patch unblocks
    # DFlash speculative decoding on gfx1151 by:
    #   - Adding `RocmAttentionMetadata.causal: bool = True` field
    #   - Threading `causal=common_attn_metadata.causal` through builder.build()
    #   - Declaring `RocmAttentionBackend.supports_non_causal() -> True`
    #   - Adding `causal: bool = True` parameter to chunked_prefill_paged_decode
    #   - Threading the flag into prefix_prefill.context_attention_fwd
    #   - Splitting the Triton _fwd_kernel inner loop with `CAUSAL: tl.constexpr`
    #     to skip the causal mask + extend the K-range to the full padded query
    #     length when CAUSAL=False
    #   - Tightening rocm_aiter_unified_attn (does NOT support non-causal)
    # Diff is 41+ / 13- across 4 files; reference patch saved at
    # .research/vllm-dflash-prs/raw/PR-40176.patch.

    # 13a: rocm_attn.py  -  backend flag + metadata field + builder propagation +
    #      type annotation cleanup + forward() flag pass-through
    p_rocm_attn = Path('vllm/v1/attention/backends/rocm_attn.py')
    if p_rocm_attn.exists():
        txt = p_rocm_attn.read_text()
        applied = False

        # 1. Drop unused FlashAttentionMetadata import
        old_import = "from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata\n"
        if old_import in txt:
            txt = txt.replace(old_import, "")
            applied = True

        # 2. Add causal field to RocmAttentionMetadata (after prefix_scheduler_metadata)
        if "causal: bool = True" not in txt:
            old_field_block = (
                "    scheduler_metadata: torch.Tensor | None = None\n"
                "    prefix_scheduler_metadata: torch.Tensor | None = None\n"
            )
            new_field_block = (
                "    scheduler_metadata: torch.Tensor | None = None\n"
                "    prefix_scheduler_metadata: torch.Tensor | None = None\n"
                "\n"
                "    # DFlash drafting sets this to False via CommonAttentionMetadata.\n"
                "    causal: bool = True\n"
            )
            if old_field_block in txt:
                txt = txt.replace(old_field_block, new_field_block, 1)
                applied = True

        # 3. Builder.build()  -  propagate common_attn_metadata.causal into the dataclass
        if "causal=common_attn_metadata.causal" not in txt:
            old_build_tail = "            prefix_scheduler_metadata=prefix_scheduler_metadata,\n        )\n        return attn_metadata\n"
            new_build_tail = (
                "            prefix_scheduler_metadata=prefix_scheduler_metadata,\n"
                "            causal=common_attn_metadata.causal,\n"
                "        )\n        return attn_metadata\n"
            )
            if old_build_tail in txt:
                txt = txt.replace(old_build_tail, new_build_tail, 1)
                applied = True

        # 4. Backend.supports_non_causal() classmethod returns True (gates DFlash)
        if "def supports_non_causal" not in txt:
            old_sink_block = (
                "        # kernel, which is less efficient than the proper triton backends.\n"
                "        return False\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            new_sink_block = (
                "        # kernel, which is less efficient than the proper triton backends.\n"
                "        return False\n\n"
                "    @classmethod\n"
                "    def supports_non_causal(cls) -> bool:\n"
                "        return True\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            if old_sink_block in txt:
                txt = txt.replace(old_sink_block, new_sink_block, 1)
                applied = True

        # 5. Type-annotation fixups: FlashAttentionMetadata -> RocmAttentionMetadata
        if "attn_metadata: FlashAttentionMetadata" in txt:
            txt = txt.replace(
                "attn_metadata: FlashAttentionMetadata",
                "attn_metadata: RocmAttentionMetadata",
            )
            applied = True

        # 6. forward()  -  pass causal=attn_metadata.causal into chunked_prefill_paged_decode
        if "causal=attn_metadata.causal" not in txt:
            old_call_tail = (
                "            sm_scale=self.scale,\n"
                "            output_scale=output_scale,\n"
                "            sinks=self.sinks,\n"
                "        )\n"
            )
            new_call_tail = (
                "            sm_scale=self.scale,\n"
                "            output_scale=output_scale,\n"
                "            sinks=self.sinks,\n"
                "            causal=attn_metadata.causal,\n"
                "        )\n"
            )
            if old_call_tail in txt:
                txt = txt.replace(old_call_tail, new_call_tail, 1)
                applied = True

        if applied:
            p_rocm_attn.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_attn.py (PR #40176: non-causal support)")

    # 13b: rocm_aiter_unified_attn.py  -  must explicitly opt OUT of non-causal
    p_rocm_aiter_uni = Path('vllm/v1/attention/backends/rocm_aiter_unified_attn.py')
    if p_rocm_aiter_uni.exists():
        txt = p_rocm_aiter_uni.read_text()
        applied = False

        # Switch the metadata import to RocmAttentionMetadata
        old_imports = (
            "from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata\n"
            "from vllm.v1.attention.backends.rocm_attn import (\n"
            "    RocmAttentionBackend,\n"
            "    RocmAttentionImpl,\n"
            "    RocmAttentionMetadataBuilder,\n"
            ")\n"
        )
        new_imports = (
            "from vllm.v1.attention.backends.rocm_attn import (\n"
            "    RocmAttentionBackend,\n"
            "    RocmAttentionImpl,\n"
            "    RocmAttentionMetadata,\n"
            "    RocmAttentionMetadataBuilder,\n"
            ")\n"
        )
        if old_imports in txt:
            txt = txt.replace(old_imports, new_imports, 1)
            applied = True

        # Add explicit supports_non_causal=False (this backend doesn't support it)
        if "def supports_non_causal" not in txt:
            old_sink_block = (
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            new_sink_block = (
                "    def supports_sink(cls) -> bool:\n"
                "        return True\n\n"
                "    @classmethod\n"
                "    def supports_non_causal(cls) -> bool:\n"
                "        return False\n\n"
                "    forward_includes_kv_cache_update: bool = False\n"
            )
            if old_sink_block in txt:
                txt = txt.replace(old_sink_block, new_sink_block, 1)
                applied = True

        # Type annotation fixup
        if "attn_metadata: FlashAttentionMetadata" in txt:
            txt = txt.replace(
                "attn_metadata: FlashAttentionMetadata",
                "attn_metadata: RocmAttentionMetadata",
            )
            applied = True

        if applied:
            p_rocm_aiter_uni.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_unified_attn.py (PR #40176: explicit non-causal=False)")

    # 13c: chunked_prefill_paged_decode.py  -  add causal kwarg + forward to context_attention_fwd
    p_chunked = Path('vllm/v1/attention/ops/chunked_prefill_paged_decode.py')
    if p_chunked.exists():
        txt = p_chunked.read_text()
        applied = False

        # Add causal: bool = True parameter to public function
        old_sig_tail = (
            "    # Optional tensor for sinks\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "):\n"
        )
        new_sig_tail = (
            "    # Optional tensor for sinks\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "    causal: bool = True,\n"
            "):\n"
        )
        if old_sig_tail in txt and "causal: bool = True" not in txt.split("def chunked_prefill_paged_decode")[1].split(":\n", 1)[0]:
            txt = txt.replace(old_sig_tail, new_sig_tail, 1)
            applied = True

        # Forward causal= into context_attention_fwd call
        old_inner_call_tail = (
            "            skip_decode=True,\n"
            "            fp8_out_scale=output_scale,\n"
            "            sinks=sinks,\n"
            "        )\n"
        )
        new_inner_call_tail = (
            "            skip_decode=True,\n"
            "            fp8_out_scale=output_scale,\n"
            "            sinks=sinks,\n"
            "            causal=causal,\n"
            "        )\n"
        )
        if old_inner_call_tail in txt and "sinks=sinks,\n            causal=causal" not in txt:
            txt = txt.replace(old_inner_call_tail, new_inner_call_tail, 1)
            applied = True

        if applied:
            p_chunked.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/chunked_prefill_paged_decode.py (PR #40176: causal kwarg)")

    # 13d: prefix_prefill.py  -  Triton _fwd_kernel CAUSAL constexpr + context_attention_fwd causal arg
    p_prefix = Path('vllm/v1/attention/ops/prefix_prefill.py')
    if p_prefix.exists():
        txt = p_prefix.read_text()
        applied = False

        # 1. Add CAUSAL: tl.constexpr to _fwd_kernel signature (before MAX_Q_LEN)
        old_kernel_sig = (
            "    SKIP_DECODE: tl.constexpr,\n"
            "    USE_SINKS: tl.constexpr,\n"
            "    USE_FP8: tl.constexpr,\n"
            "    MAX_Q_LEN: tl.constexpr = 0,\n"
        )
        new_kernel_sig = (
            "    SKIP_DECODE: tl.constexpr,\n"
            "    USE_SINKS: tl.constexpr,\n"
            "    USE_FP8: tl.constexpr,\n"
            "    CAUSAL: tl.constexpr = True,\n"
            "    MAX_Q_LEN: tl.constexpr = 0,\n"
        )
        if "CAUSAL: tl.constexpr" not in txt and old_kernel_sig in txt:
            txt = txt.replace(old_kernel_sig, new_kernel_sig, 1)
            applied = True

        # 2. Replace the inner-loop upper bound to branch on CAUSAL
        old_loop_block = (
            "    # compute query against itself (with causal mask)\n"
            "    for start_n in tl.range(\n"
            "        0,\n"
            "        block_mask * (start_m + 1) * BLOCK_M,\n"
            "        BLOCK_N,\n"
            "        loop_unroll_factor=num_unroll_request,\n"
            "    ):\n"
        )
        new_loop_block = (
            "    # compute query against itself (causal among queries by default;\n"
            "    # CAUSAL=False for bidirectional attention over query tokens, e.g. DFlash.)\n"
            "    if CAUSAL:\n"
            "        key_range_upper = block_mask * (start_m + 1) * BLOCK_M\n"
            "    else:\n"
            "        q_len_pad = (cur_batch_query_len + BLOCK_N - 1) // BLOCK_N * BLOCK_N\n"
            "        key_range_upper = block_mask * q_len_pad\n"
            "\n"
            "    for start_n in tl.range(\n"
            "        0,\n"
            "        key_range_upper,\n"
            "        BLOCK_N,\n"
            "        loop_unroll_factor=num_unroll_request,\n"
            "    ):\n"
        )
        if "key_range_upper" not in txt and old_loop_block in txt:
            txt = txt.replace(old_loop_block, new_loop_block, 1)
            applied = True

        # 3. Replace the qk causal-mask + sliding-window block with conditional logic
        old_mask_block = (
            "        qk *= sm_scale\n"
            "        # apply causal mask\n"
            "        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float(\"-inf\"))\n"
            "        if SLIDING_WINDOW > 0:\n"
            "            qk = tl.where(\n"
            "                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,\n"
            "                qk,\n"
            "                float(\"-inf\"),\n"
            "            )\n"
        )
        new_mask_block = (
            "        qk *= sm_scale\n"
            "\n"
            "        valid_kv = (start_n + offs_n[None, :]) < cur_batch_query_len\n"
            "        if CAUSAL:\n"
            "            attn_mask = valid_kv & (offs_m[:, None] >= (start_n + offs_n[None, :]))\n"
            "        else:\n"
            "            attn_mask = valid_kv\n"
            "        if SLIDING_WINDOW > 0:\n"
            "            attn_mask = attn_mask & (\n"
            "                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW\n"
            "            )\n"
            "        qk = tl.where(attn_mask, qk, float(\"-inf\"))\n"
        )
        if "valid_kv = " not in txt and old_mask_block in txt:
            txt = txt.replace(old_mask_block, new_mask_block, 1)
            applied = True

        # 4. Add causal: bool = True parameter to context_attention_fwd
        old_ctx_sig_tail = (
            "    fp8_out_scale=None,\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "):\n"
        )
        new_ctx_sig_tail = (
            "    fp8_out_scale=None,\n"
            "    sinks=None,\n"
            "    is_block_table_ptr: bool = False,\n"
            "    causal: bool = True,\n"
            "):\n"
        )
        # The chunked_prefill file might also match this pattern, but in this
        # file it occurs once at context_attention_fwd's signature. Use rfind
        # discipline by checking that we have not already added causal here.
        if old_ctx_sig_tail in txt and txt.count("def context_attention_fwd") == 1:
            # Only add if context_attention_fwd doesn't already have causal:
            ctx_def_start = txt.find("def context_attention_fwd")
            ctx_def_end = txt.find("):\n", ctx_def_start) + 3
            ctx_signature = txt[ctx_def_start:ctx_def_end]
            if "causal: bool" not in ctx_signature:
                txt = txt.replace(old_ctx_sig_tail, new_ctx_sig_tail, 1)
                applied = True

        # 5. Add alibi+non-causal assert
        old_alibi = (
            "    if alibi_slopes is not None:\n"
            "        assert sinks is None, \"Sinks arg is not supported with alibi\"\n"
        )
        new_alibi = (
            "    if alibi_slopes is not None:\n"
            "        assert causal, \"Non-causal prefix attention is not supported with alibi\"\n"
            "        assert sinks is None, \"Sinks arg is not supported with alibi\"\n"
        )
        if old_alibi in txt and "Non-causal prefix attention is not supported with alibi" not in txt:
            txt = txt.replace(old_alibi, new_alibi, 1)
            applied = True

        # 6. Pass CAUSAL=causal into the kernel call
        old_kernel_call_tail = (
            "        num_warps=4,\n"
            "        num_stages=1,\n"
            "        USE_SINKS=sinks is not None,\n"
            "        **extra_kargs,\n"
            "    )\n"
        )
        new_kernel_call_tail = (
            "        num_warps=4,\n"
            "        num_stages=1,\n"
            "        USE_SINKS=sinks is not None,\n"
            "        CAUSAL=causal,\n"
            "        **extra_kargs,\n"
            "    )\n"
        )
        if "CAUSAL=causal" not in txt and old_kernel_call_tail in txt:
            txt = txt.replace(old_kernel_call_tail, new_kernel_call_tail, 1)
            applied = True

        if applied:
            p_prefix.write_text(txt)
            print(" -> Patched vllm/v1/attention/ops/prefix_prefill.py (PR #40176: CAUSAL constexpr in _fwd_kernel + context_attention_fwd)")

    # Patch 14 (local cherry-pick of vLLM PR #40898): "[Spec Decode] Add Sliding
    # Window Attention support to DFlash drafter". OPEN at 2026-04-26 (not yet
    # merged) but author jianc99 (also the DFlash paper author) explicitly
    # recommends installing this PR for vanilla vLLM compatibility with the
    # z-lab/Qwen3.6-27B-DFlash drafter, which has interleaved SWA layers
    # (4x sliding_attention + 1x full_attention per the drafter's config.json).
    # Without this patch:
    #   1. SWA layers in drafter run as full attention -> ~25% lower acceptance
    #      length on long-context inputs (per author's HumanEval bench).
    #   2. target_layer_ids is OFF BY ONE in gpu_model_runner.py (correctness
    #      issue, not just optimization) -> drafter reads wrong target hidden
    #      states and acceptance plummets at any context length.
    # Diff is 156+ / 1- across 5 files (4 production + 1 test). Production-only
    # cherry-pick saved at .research/vllm-dflash-prs/raw/PR-40898.patch.

    # 14a: qwen3_dflash.py  -  multiple structural changes
    p_qwen3_dflash = Path('vllm/model_executor/models/qwen3_dflash.py')
    if p_qwen3_dflash.exists():
        txt = p_qwen3_dflash.read_text()
        applied = False

        # 14a-1: Helper function and frozenset before DFlashQwen3Attention class
        if "_DFLASH_VALID_LAYER_TYPES" not in txt:
            old_anchor = "logger = init_logger(__name__)\n\n\nclass DFlashQwen3Attention"
            new_block = (
                "logger = init_logger(__name__)\n"
                "\n"
                "\n"
                "_DFLASH_VALID_LAYER_TYPES = frozenset({\"full_attention\", \"sliding_attention\"})\n"
                "\n"
                "\n"
                "def _get_dflash_layer_types(config) -> tuple[str, ...]:\n"
                "    layer_types = getattr(config, \"layer_types\", None)\n"
                "    if layer_types is None:\n"
                "        return (\"full_attention\",) * config.num_hidden_layers\n"
                "    if len(layer_types) != config.num_hidden_layers:\n"
                "        raise ValueError(\n"
                "            f\"DFlash layer_types length {len(layer_types)} does not match \"\n"
                "            f\"num_hidden_layers {config.num_hidden_layers}.\"\n"
                "        )\n"
                "    invalid = set(layer_types) - _DFLASH_VALID_LAYER_TYPES\n"
                "    if invalid:\n"
                "        raise ValueError(f\"Invalid DFlash layer_type(s): {sorted(invalid)}.\")\n"
                "    if \"sliding_attention\" in layer_types and not getattr(\n"
                "        config, \"sliding_window\", None\n"
                "    ):\n"
                "        raise ValueError(\n"
                "            \"DFlash sliding_attention layers require `sliding_window` in config.\"\n"
                "        )\n"
                "    return tuple(layer_types)\n"
                "\n"
                "\n"
                "class DFlashQwen3Attention"
            )
            if old_anchor in txt:
                txt = txt.replace(old_anchor, new_block, 1)
                applied = True

        # 14a-2: Add sliding_window param to DFlashQwen3Attention.__init__ signature
        old_attn_sig = (
            "        attention_bias: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        prefix: str = \"\",\n"
            "        attn_type: str = AttentionType.DECODER,\n"
            "    ) -> None:\n"
        )
        new_attn_sig = (
            "        attention_bias: bool = False,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        sliding_window: int | None = None,\n"
            "        prefix: str = \"\",\n"
            "        attn_type: str = AttentionType.DECODER,\n"
            "    ) -> None:\n"
        )
        if old_attn_sig in txt:
            # Only one signature matches this exact 6-line tail  -  replace once.
            txt = txt.replace(old_attn_sig, new_attn_sig, 1)
            applied = True

        # 14a-3: Add per_layer_sliding_window to Attention() call + post-call zero-out
        old_attn_call = (
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            prefix=f\"{prefix}.attn\",\n"
            "            attn_type=attn_type,\n"
            "        )\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n"
        )
        new_attn_call = (
            "            num_kv_heads=self.num_kv_heads,\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            per_layer_sliding_window=sliding_window,\n"
            "            prefix=f\"{prefix}.attn\",\n"
            "            attn_type=attn_type,\n"
            "        )\n"
            "        if sliding_window is not None:\n"
            "            # DFlash keeps full KV allocation while using SWA only for compute.\n"
            "            self.attn.sliding_window = None\n"
            "        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n"
        )
        if old_attn_call in txt and "per_layer_sliding_window=sliding_window" not in txt:
            txt = txt.replace(old_attn_call, new_attn_call, 1)
            applied = True

        # 14a-4: Add layer_type param + body changes to DFlashQwen3DecoderLayer.__init__
        old_dec_init = (
            "        config: Qwen3Config,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        prefix: str = \"\",\n"
            "    ) -> None:\n"
            "        super().__init__()\n"
            "        self.hidden_size = config.hidden_size\n"
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n"
        )
        new_dec_init = (
            "        config: Qwen3Config,\n"
            "        cache_config: CacheConfig | None = None,\n"
            "        quant_config: QuantizationConfig | None = None,\n"
            "        layer_type: str = \"full_attention\",\n"
            "        prefix: str = \"\",\n"
            "    ) -> None:\n"
            "        super().__init__()\n"
            "        self.hidden_size = config.hidden_size\n"
            "        self.layer_type = layer_type\n"
            "        set_default_rope_theta(config, default_theta=1000000)\n"
            "        attn_type = AttentionType.DECODER\n"
            "        sliding_window = (\n"
            "            config.sliding_window if layer_type == \"sliding_attention\" else None\n"
            "        )\n"
            "\n"
            "        self.self_attn = DFlashQwen3Attention(\n"
        )
        if old_dec_init in txt and "self.layer_type = layer_type" not in txt:
            txt = txt.replace(old_dec_init, new_dec_init, 1)
            applied = True

        # 14a-5: Pass sliding_window into DFlashQwen3Attention( ) call site
        old_attn_call2 = (
            "            head_dim=getattr(config, \"head_dim\", None),\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            rope_parameters=config.rope_parameters,\n"
        )
        new_attn_call2 = (
            "            head_dim=getattr(config, \"head_dim\", None),\n"
            "            cache_config=cache_config,\n"
            "            quant_config=quant_config,\n"
            "            sliding_window=sliding_window,\n"
            "            rope_parameters=config.rope_parameters,\n"
        )
        if old_attn_call2 in txt and "sliding_window=sliding_window," not in txt:
            txt = txt.replace(old_attn_call2, new_attn_call2, 1)
            applied = True

        # 14a-6: DFlashQwen3Model.__init__  -  compute layer_types, propagate, build set
        old_layers = (
            "        self.layers = nn.ModuleList(\n"
            "            [\n"
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n"
            "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
            "                    config=self.config,\n"
            "                )\n"
            "                for layer_idx in range(self.config.num_hidden_layers)\n"
            "            ]\n"
            "        )\n"
            "        if self.use_aux_hidden_state:\n"
        )
        new_layers = (
            "        self.layer_types = _get_dflash_layer_types(self.config)\n"
            "        self.layers = nn.ModuleList(\n"
            "            [\n"
            "                DFlashQwen3DecoderLayer(\n"
            "                    current_vllm_config,\n"
            "                    prefix=maybe_prefix(prefix, f\"layers.{layer_idx + start_layer_id}\"),\n"
            "                    config=self.config,\n"
            "                    layer_type=self.layer_types[layer_idx],\n"
            "                )\n"
            "                for layer_idx in range(self.config.num_hidden_layers)\n"
            "            ]\n"
            "        )\n"
            "        self.sliding_attention_layer_names = {\n"
            "            layer.self_attn.attn.layer_name\n"
            "            for layer in self.layers\n"
            "            if layer.layer_type == \"sliding_attention\"\n"
            "        }\n"
            "        if self.use_aux_hidden_state:\n"
        )
        if old_layers in txt and "self.layer_types = _get_dflash_layer_types" not in txt:
            txt = txt.replace(old_layers, new_layers, 1)
            applied = True

        # 14a-7: Add @property sliding_attention_layer_names to DFlashQwen3ForCausalLM
        old_precompute_tail = (
            "        \"\"\"Precompute projected + RoPE'd K/V and write to cache.\"\"\"\n"
            "        self.model.precompute_and_store_context_kv(\n"
            "            context_states, context_positions, context_slot_mapping\n"
            "        )\n"
            "\n"
            "    def combine_hidden_states(\n"
        )
        new_precompute_tail = (
            "        \"\"\"Precompute projected + RoPE'd K/V and write to cache.\"\"\"\n"
            "        self.model.precompute_and_store_context_kv(\n"
            "            context_states, context_positions, context_slot_mapping\n"
            "        )\n"
            "\n"
            "    @property\n"
            "    def sliding_attention_layer_names(self) -> set[str]:\n"
            "        return self.model.sliding_attention_layer_names\n"
            "\n"
            "    def combine_hidden_states(\n"
        )
        if old_precompute_tail in txt and "def sliding_attention_layer_names(self)" not in txt:
            txt = txt.replace(old_precompute_tail, new_precompute_tail, 1)
            applied = True

        if applied:
            p_qwen3_dflash.write_text(txt)
            print(" -> Patched vllm/model_executor/models/qwen3_dflash.py (PR #40898: SWA support in DFlash drafter)")

    # 14b: algos.py  -  preserve SWA fields when extracting HF config
    p_algos = Path('vllm/transformers_utils/configs/speculators/algos.py')
    if p_algos.exists():
        txt = p_algos.read_text()
        old_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        new_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "    for key in (\n"
            "        \"layer_types\",\n"
            "        \"use_sliding_window\",\n"
            "        \"sliding_window\",\n"
            "        \"max_window_layers\",\n"
            "    ):\n"
            "        if key in config_dict:\n"
            "            pre_trained_config[key] = config_dict[key]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        if old_target in txt and '"layer_types",' not in txt:
            txt = txt.replace(old_target, new_target, 1)
            p_algos.write_text(txt)
            print(" -> Patched vllm/transformers_utils/configs/speculators/algos.py (PR #40898: preserve SWA config)")

    # 14c: dflash.py  -  SWA branch in build_per_group_and_layer_attn_metadata
    p_dflash_proposer = Path('vllm/v1/spec_decode/dflash.py')
    if p_dflash_proposer.exists():
        txt = p_dflash_proposer.read_text()
        old_block = (
            "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
            "            cad, draft_index\n"
            "        )\n"
            "        for layer_name, attn_metadata in per_layer.items():\n"
            "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
            "                f\"Attention metadata for layer {layer_name} does not have\"\n"
            "                \" non-causal support, which is required for DFlash.\"\n"
            "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
            "            )\n"
            "        return per_group, per_layer\n"
        )
        new_block = (
            "        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(\n"
            "            cad, draft_index\n"
            "        )\n"
            "        sliding_layer_names = getattr(self.model, \"sliding_attention_layer_names\", set())\n"
            "        if sliding_layer_names:\n"
            "            causal_cad = cad.replace(causal=True)\n"
            "            for attn_group in self.draft_attn_groups:\n"
            "                causal_layers = sliding_layer_names & set(attn_group.layer_names)\n"
            "                if not causal_layers:\n"
            "                    continue\n"
            "                attn_metadata = attn_group.get_metadata_builder().build_for_drafting(\n"
            "                    common_attn_metadata=causal_cad, draft_index=draft_index\n"
            "                )\n"
            "                for layer_name in causal_layers:\n"
            "                    per_layer[layer_name] = attn_metadata\n"
            "\n"
            "        for layer_name, attn_metadata in per_layer.items():\n"
            "            if layer_name in sliding_layer_names:\n"
            "                assert getattr(attn_metadata, \"causal\", None) is True, (\n"
            "                    f\"Attention metadata for sliding layer {layer_name} does not have\"\n"
            "                    \" causal support, which is required for DFlash SWA.\"\n"
            "                )\n"
            "                continue\n"
            "            assert getattr(attn_metadata, \"causal\", None) is False, (\n"
            "                f\"Attention metadata for layer {layer_name} does not have\"\n"
            "                \" non-causal support, which is required for DFlash.\"\n"
            "                \" Consider using a different attention backend, such as FlashAttention.\"\n"
            "            )\n"
            "        return per_group, per_layer\n"
        )
        if old_block in txt and "sliding_layer_names" not in txt:
            txt = txt.replace(old_block, new_block, 1)
            p_dflash_proposer.write_text(txt)
            print(" -> Patched vllm/v1/spec_decode/dflash.py (PR #40898: SWA branch + SWA causal assertion)")

    # 14d: gpu_model_runner.py  -  fix target_layer_ids +1 shift for dflash method
    p_gmr = Path('vllm/v1/worker/gpu_model_runner.py')
    if p_gmr.exists():
        txt = p_gmr.read_text()
        old_block = (
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "\n"
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
            "\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None\n"
        )
        new_block = (
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "\n"
            "        is_dflash = self.speculative_config.method == \"dflash\"\n"
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if is_dflash or not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
            "\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            if is_dflash:\n"
            "                return tuple(layer_id + 1 for layer_id in layer_ids)\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None\n"
        )
        if old_block in txt and "is_dflash = self.speculative_config.method" not in txt:
            txt = txt.replace(old_block, new_block, 1)
            p_gmr.write_text(txt)
            print(" -> Patched vllm/v1/worker/gpu_model_runner.py (PR #40898: target_layer_ids +1 shift fix for dflash)")

    # Patch 15 (local): thread chat_template_kwargs through /v1/responses.
    #
    # Without this, ResponsesRequest.to_chat_params() builds chat_template_kwargs
    # from a hardcoded dict and never reads the request body's
    # chat_template_kwargs field. Effect on Qwen3.6: clients that send
    # `chat_template_kwargs: {"enable_thinking": false}` get reasoning anyway,
    # while the same kwarg works on /v1/chat/completions (different code path).
    # The chat template ITSELF supports enable_thinking - this gap is purely
    # in vLLM's request-to-template wiring on the responses endpoint.
    #
    # Fix is two changes:
    #   15a: add a chat_template_kwargs field to the ResponsesRequest model
    #   15b: pass it as `defaults` to merge_kwargs() so user-supplied kwargs
    #        live alongside vLLM's hardcoded add_generation_prompt etc.
    #        (vLLM's overrides still win for keys it controls).
    #
    # Worth filing upstream as a vLLM PR; the gap looks accidental.
    p_responses_proto = Path('vllm/entrypoints/openai/responses/protocol.py')
    if p_responses_proto.exists():
        txt = p_responses_proto.read_text()

        # 15a: add chat_template_kwargs field, sandwiched between `user` (last
        # OpenAI-spec field) and `skip_special_tokens` (first vLLM extension).
        field_anchor = "    user: str | None = None\n    skip_special_tokens: bool = True\n"
        field_replacement = (
            "    user: str | None = None\n"
            "    chat_template_kwargs: dict[str, Any] | None = None\n"
            "    skip_special_tokens: bool = True\n"
        )
        if "chat_template_kwargs: dict[str, Any] | None = None" not in txt and field_anchor in txt:
            txt = txt.replace(field_anchor, field_replacement, 1)
            print(" -> Patched protocol.py (15a: ResponsesRequest gains chat_template_kwargs field)")

        # 15b: in to_chat_params(), feed the user kwargs into merge_kwargs as
        # the `defaults` argument. The hardcoded dict stays as `overrides` so
        # vLLM-managed keys (add_generation_prompt, continue_final_message,
        # reasoning_effort) keep precedence, while user-supplied keys
        # (enable_thinking, etc.) flow through to the chat template renderer.
        # Indents: the call sits inside `return ChatParams(` so it's 12 spaces
        # for the kwarg line and 16 spaces for the inner positional args.
        merge_anchor = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        merge_replacement = (
            "            chat_template_kwargs=merge_kwargs(  # To remove unset values\n"
            "                self.chat_template_kwargs or {},\n"
            "                dict(\n"
            "                    add_generation_prompt=not continue_final,\n"
        )
        if "self.chat_template_kwargs or {}" not in txt and merge_anchor in txt:
            txt = txt.replace(merge_anchor, merge_replacement, 1)
            print(" -> Patched protocol.py (15b: to_chat_params merges user chat_template_kwargs)")

        p_responses_proto.write_text(txt)

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
