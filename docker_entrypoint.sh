#!/bin/bash
# Patch vLLM to handle: 1) missing num_soft_tokens, 2) ignore list suffix matching
set -e

VLLM_MODEL_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_unified.py"
VLLM_UTILS_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/utils.py"

echo "=== Patching gemma4_unified.py for missing num_soft_tokens ==="
if grep -q "config.vision_config.num_soft_tokens" "$VLLM_MODEL_FILE"; then
	sed -i \
		's/tokens_per_image = config.vision_config.num_soft_tokens/tokens_per_image = getattr(config.vision_config, "num_soft_tokens", getattr(config.vision_config, "mm_posemb_size", 256))/' \
		"$VLLM_MODEL_FILE"
	echo "✅ num_soft_tokens patch applied."
else
	echo "⚠️  num_soft_tokens pattern not found — already patched or different version."
fi

echo "=== Patching compressed_tensors utils.py for suffix matching ==="
# Make _is_equal_or_regex_match also check if target is a suffix of value (or vice versa)
# This handles cases like "model.vision_embedder.patch_dense" matching "vision_embedder.patch_dense"
# or "vision_embedder.patch_dense" matching "model.vision_embedder.patch_dense"
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
