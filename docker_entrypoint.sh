#!/bin/bash
# Patch vLLM to handle: 1) missing num_soft_tokens (use smaller fallback), 2) ignore list suffix matching
set -e

VLLM_UNIFIED_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_unified.py"
VLLM_MM_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py"
VLLM_UTILS_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/utils.py"

echo "=== Patching gemma4_unified.py for missing num_soft_tokens ==="
# Patch line 165: tokens_per_image = config.vision_config.num_soft_tokens
# Use 256 as fallback (NOT mm_posemb_size which is position embedding dimension)
if grep -q "tokens_per_image = config.vision_config.num_soft_tokens" "$VLLM_UNIFIED_FILE"; then
	sed -i \
		's/tokens_per_image = config.vision_config.num_soft_tokens/tokens_per_image = getattr(config.vision_config, "num_soft_tokens", getattr(config.vision_config, "default_output_length", 256))/' \
		"$VLLM_UNIFIED_FILE"
	echo "✅ gemma4_unified.py line 165 patched (using 256 fallback)."
else
	echo "⚠️  gemma4_unified.py line 165 pattern not found — already patched."
fi

# Patch line 196: max_soft_tokens = vision_cfg.num_soft_tokens
if grep -q "max_soft_tokens = vision_cfg.num_soft_tokens" "$VLLM_UNIFIED_FILE"; then
	sed -i \
		's/max_soft_tokens = vision_cfg.num_soft_tokens/max_soft_tokens = getattr(vision_cfg, "num_soft_tokens", getattr(vision_cfg, "default_output_length", 256))/' \
		"$VLLM_UNIFIED_FILE"
	echo "✅ gemma4_unified.py line 196 patched (using 256 fallback)."
else
	echo "⚠️  gemma4_unified.py line 196 pattern not found — already patched."
fi

echo "=== Patching gemma4_mm.py for missing default_output_length ==="
# gemma4_mm.py uses default_output_length instead of num_soft_tokens
# Patch line 297: max_soft_tokens = vision_cfg.default_output_length
if grep -q "max_soft_tokens = vision_cfg.default_output_length" "$VLLM_MM_FILE"; then
	sed -i \
		's/max_soft_tokens = vision_cfg.default_output_length/max_soft_tokens = getattr(vision_cfg, "default_output_length", getattr(vision_cfg, "num_soft_tokens", 256))/' \
		"$VLLM_MM_FILE"
	echo "✅ gemma4_mm.py patched for default_output_length."
else
	echo "⚠️  gemma4_mm.py default_output_length pattern not found — already patched."
fi

echo "=== Patching compressed_tensors utils.py for suffix matching ==="
# Make _is_equal_or_regex_match also check if target is a suffix of value (or vice versa)
if grep -q "elif target == value:" "$VLLM_UTILS_FILE"; then
	sed -i \
		's/elif target == value:/elif target == value or value.endswith(target) or target.endswith(value):/' \
		"$VLLM_UTILS_FILE"
	echo "✅ suffix matching patch applied."
else
	echo "⚠️  suffix matching pattern not found — different version."
fi

echo "=== Starting vLLM ==="
exec vllm serve "$@"
