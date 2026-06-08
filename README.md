# vLLM Inference Server - Gemma 4 Unified Multimodal

Docker-based vLLM inference server for **Gemma 4 12B Unified** - a multimodal model supporting text, image, audio, and video inputs.

## Model Details

- **Model:** `gemma-4-12B-it-assistant` (Gemma4UnifiedForConditionalGeneration)
- **Quantization:** W4A16 QAT (4-bit weights, 16-bit activations) via compressed-tensors
- **Capabilities:** Multimodal (text + image + audio + video)
- **Context Length:** 128K (131,072 tokens)
- **KV Cache:** FP8 (reduces memory ~50%)

## Hardware Requirements

- **GPU:** NVIDIA GPU with ~16GB VRAM (tested on RTX 5070 Ti)
- **Docker:** With nvidia-container-toolkit installed
- **Disk:** ~10GB for model weights

## Quick Start

```bash
# Start the server
./start_vllm_server.sh

# Server will be available at:
# http://localhost:1234/v1/models
# http://localhost:1234/v1/chat/completions
```

## API Endpoints

OpenAI-compatible API:

| Endpoint | Description |
|----------|-------------|
| `/v1/models` | List available models |
| `/v1/chat/completions` | Chat completions (text + multimodal) |
| `/v1/completions` | Text completions |
| `/v1/responses` | Response API |
| `/health` | Health check |

## Example Usage

### Text-only Request

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/gemma-4-12B-it-assistant",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Image Request

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/gemma-4-12B-it-assistant",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_DATA>"}}
        ]
      }
    ],
    "max_tokens": 500
  }'
```

## Configuration

Current settings in `start_vllm_server.sh`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max-model-len` | 131072 | 128K context |
| `--kv-cache-dtype` | fp8 | FP8 KV cache for memory efficiency |
| `--gpu-memory-utilization` | 0.90 | GPU memory fraction |
| `--max-num-seqs` | 16 | Max concurrent sequences |
| `--tensor-parallel-size` | 1 | Single GPU |
| `--tool-call-parser` | gemma4 | Tool calling support |
| `--reasoning-parser` | gemma4 | Reasoning token support |

## Patches Applied

This project includes runtime patches applied via `docker_entrypoint.sh` to fix missing attributes in the vLLM code:

### 1. num_soft_tokens Fallback

**Problem:** The model's `vision_config` lacks `num_soft_tokens` and `default_output_length` attributes, causing crashes during multimodal processing.

**Fix:** Patched the following files to use `getattr()` with 256 fallback:

- `gemma4_unified.py` line 165: `tokens_per_image`
- `gemma4_unified.py` line 196: `max_soft_tokens`
- `gemma4_mm.py` line 297: `default_output_length`

**Why 256?** Using `mm_posemb_size` (1120) as fallback created too many placeholder tokens, causing shape mismatch errors. 256 matches typical vision encoder output.

### 2. Ignore List Suffix Matching

**Problem:** The compressed-tensors quantization ignore list wasn't matching layer names due to prefix differences.

**Fix:** Patched `compressed_tensors/utils.py` to check suffix matching:
```python
elif target == value or value.endswith(target) or target.endswith(value):
```

This allows `"model.vision_embedder.patch_dense"` to match `"vision_embedder.patch_dense"`.

## Memory Profile

On RTX 5070 Ti (16GB):

| Component | Memory |
|-----------|--------|
| Model weights | ~8.3 GiB |
| KV cache (FP8) | ~5.1 GiB |
| CUDA graphs | ~0.06 GiB |
| **Total** | ~13.5 GiB |

**Performance:**
- KV cache tokens: 433,456
- Max concurrency (128K/request): 3.31x

## Files

```
.
├── start_vllm_server.sh    # Main startup script
├── docker_entrypoint.sh     # Runtime patches + vLLM launch
├── models/
│   └── gemma-4-12B-it-assistant/  # Model weights (~9.6GB)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
└── README.md
```

## Troubleshooting

### Container won't start / OOM errors

- Lower `--gpu-memory-utilization` (e.g., 0.85)
- Lower `--max-num-seqs` (e.g., 8)
- Reduce `--max-model-len` (e.g., 65536 for 64K)

### Image processing fails

- Check docker logs: `docker logs vllm-gemma4-server`
- Verify patches were applied (look for "✅" messages in logs)
- Ensure image is in supported format (JPEG, PNG)

### CUDA not detected

- Install nvidia-container-toolkit
- Restart Docker daemon: `sudo systemctl restart docker`

## Docker Image

```
docker.xuanyuan.run/vllm/vllm-openai:gemma4-unified-x86_64-cu130
```

Custom vLLM image with Gemma 4 support and CUDA 13.0.

## Stop Server

```bash
docker stop vllm-gemma4-server
docker rm vllm-gemma4-server  # Optional: remove container
```

## License

Model: Google Gemma 4 (check model license for usage terms)
vLLM: Apache 2.0