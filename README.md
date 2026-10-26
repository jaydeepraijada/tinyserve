# tinyserve

A small vLLM-style LLM inference engine in one file ([`serve.py`](serve.py)), written to understand how modern serving engines work from the inside.

It implements the core ideas behind vLLM end to end:

- **Paged KV cache**: K/V lives in fixed-size pages, and each request has a block table mapping its tokens to pages anywhere in GPU memory.
- **Prefix caching**: full pages are hashed (chained through the parent page's hash), so requests that share a prefix, like a system prompt, reuse its K/V instead of recomputing it. Freed pages stay cached on an LRU free list until they're needed.
- **Continuous batching**: requests join and leave the batch every step, instead of waiting for a whole static batch to finish.
- **Chunked prefill**: long prompts are split to fit a per-step token budget shared with decodes, so one long prompt never stalls everyone else.
- **Preemption**: when the cache is full, the newest request is evicted and recomputed later, usually cheaply thanks to the prefix cache.
- **Paged attention**: FlashAttention's varlen kernel with a block table, or a pure PyTorch reference implementation.
- **CUDA graphs** for decode steps, captured per batch-size bucket.
- **Batched sampling** with per-request temperature, top-k and top-p (Gumbel-max).
- **OpenAI-compatible HTTP server**: `/v1/completions` and `/v1/chat/completions`, streaming and non-streaming.

The model code (RMSNorm, RoPE, GQA attention, SwiGLU MLP) is written from scratch and loads Hugging Face safetensors weights directly. It supports Qwen3, Qwen2 and Llama-style models with plain RoPE.

## Results

`python serve.py bench --model Qwen/Qwen3-0.6B` on an NVIDIA DGX Spark (GB10), with flash-attn: 100 real prompts from Alpaca and Dolly, spread across short, medium and long lengths, all sharing a ~256-token system prompt, greedy decoding, up to 512 output tokens.

| | Wall time | Requests/s | Output tokens/s | p50 / p90 finish | Prompt tokens from cache |
|---|---|---|---|---|---|
| Hugging Face `generate`, static batches of 16 | 99.3 s | 1.0 | 51 | 66.6 / 94.7 s | 0% |
| tinyserve, no prefix cache | 7.7 s | 13.0 | 663 | 4.9 / 5.9 s | 0% |
| tinyserve, prefix cache | **5.7 s** | **17.7** | **893** | **3.0 / 3.9 s** | 40% |

With the same weights, prompts and GPU, continuous batching over a paged cache is about 13× faster than static batching, and prefix caching adds another 1.4×.

## Setup

Requires Python 3.10+ and an NVIDIA GPU for the engine. The two demos also run on CPU or Apple Silicon.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On a GPU with CUDA 13 (e.g. Blackwell / GB10), install torch from the matching index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### flash-attn (optional, recommended)

Without flash-attn, tinyserve falls back to the PyTorch reference attention: 16-token pages, no CUDA graphs, and about 8× slower in our tests. To install it:

```bash
pip install flash-attn --no-build-isolation
```

flash-attn has no prebuilt wheels for ARM (e.g. DGX Spark), so it compiles from source. To build only for your GPU, set its compute capability (12.0 covers the GB10's 12.1) and limit parallel jobs to save memory:

```bash
FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 NVCC_THREADS=2 \
  pip install flash-attn --no-build-isolation
```

flash-attn 2.8.3 compiles with `-std=c++17`, but torch 2.14's headers need C++20. If the build fails with `namespace "std" has no member "strong_ordering"`, change `c++17` to `c++20` in its `setup.py` and build from the source directory.

## Usage

```bash
# one prompt, raw text continuation
python serve.py generate --model Qwen/Qwen3-0.6B --prompt "The meaning of life is" --max-tokens 64

# OpenAI-compatible server
python serve.py serve --model Qwen/Qwen3-8B --port 8000

# benchmark against Hugging Face static batching
python serve.py bench --model Qwen/Qwen3-0.6B

# demos (no model weights needed)
python serve.py kv-demo            # scheduler + KV cache: chunked prefill, sharing, LRU reuse, preemption
python serve.py kv-manager-demo    # KVCacheManager on its own, step by step
python serve.py attn-demo          # paged attention vs dense attention (CPU, MPS or CUDA)
python serve.py attn-compare-demo  # flash-attn vs the torch reference (CUDA + flash-attn)
```

`generate` feeds the prompt as raw text, so the model continues it rather than answering. Use the chat endpoint for assistant-style answers.

### Talking to the server

```bash
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 64,
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Any OpenAI client works by pointing its base URL at `http://localhost:8000/v1`. Other endpoints:

- `GET /health`
- `GET /v1/models`
- `GET /metrics`: running and waiting requests, free KV blocks

Responses include `usage.prompt_tokens_details.cached_tokens`, the number of prompt tokens served from the prefix cache. When a request doesn't set `temperature`, `top_p` or `top_k`, the defaults from the model's `generation_config.json` are used, as in vLLM.

### Options

| Flag | Default | |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B` | Hugging Face model id or local path |
| `--attention-backend` | `flash_attn` | `flash_attn` or `torch`; falls back to `torch` if flash-attn isn't installed |
| `--max-model-len` | 8192 | max prompt + output tokens per request |
| `--max-num-seqs` | 128 | max requests in one batch |
| `--max-num-batched-tokens` | 2048 | token budget per step, shared by prefill chunks and decodes |
| `--gpu-memory-utilization` | 0.85 | share of GPU memory tinyserve may use; the KV cache gets what's left after weights and activations |
| `--no-cuda-graphs` | | run decode steps eagerly |
| `--no-prefix-caching` | | disable prefix caching |
| `--temperature`, `--max-tokens`, `--prompt` | | for `generate` |
| `--bench-*`, `--ignore-eos` | | benchmark settings, see `python serve.py --help` |

On shared machines, lower `--gpu-memory-utilization` to leave room for other processes. On GPUs that share system memory (DGX Spark, Jetson), memory is measured with the kernel's `MemAvailable`, so the OS page cache isn't counted as used.

## How a request flows

1. **`Engine.add_request`** tokenizes the prompt and queues a `Request`.
2. **`Scheduler.schedule`** decides who runs this step and how many tokens each. Running requests go first; if the cache is full it preempts the newest one. Then it admits waiting requests, looks up their cached prefix, and chunks long prompts to fit the token budget.
3. **`KVCacheManager.allocate`** reserves pages: cache hits are attached by reference count, new pages come off the LRU free list, and newly filled pages are hashed into the prefix cache.
4. **`ModelRunner._prepare`** packs the batch into one flat token stream (no padding) plus attention metadata: positions, slot mapping, query/key offsets and block tables. Decode-only batches replay a captured CUDA graph.
5. **The model** writes each layer's new K/V into the pages, then runs paged attention over each request's full context.
6. **`sample`** draws the next token for each request that finished its prompt this step.
7. **`Engine.step`** appends the tokens, checks stop conditions, frees finished requests' pages, and detokenizes incrementally.
8. **`AsyncEngine`** runs this loop in a background thread and streams outputs to the FastAPI handlers.

## Layout

Everything is in `serve.py`, in reading order:

| Section | What |
|---|---|
| `Config`, `SamplingParams`, `Request` | settings and per-request state |
| `KVCacheManager` | page allocation, ref counts, prefix hashing, LRU free list |
| `ScheduledRequest`, `Scheduler` | continuous batching, chunked prefill, preemption |
| `AttnMetadata`, `torch_ref_attention`, `flash_attention` | paged attention backends |
| `RMSNorm`, `rotary`, `Attention`, `MLP`, `DecoderLayer`, `Model`, `load_model` | the transformer and weight loading |
| `sample` | batched temperature / top-k / top-p sampling |
| `ModelRunner` | KV memory sizing, batch packing, CUDA graphs |
| `Detokenizer`, `Engine` | the step loop and incremental text output |
| `AsyncEngine`, `build_app` | background engine thread and the OpenAI-compatible server |
| `kv_cache_demo`, `kv_manager_demo`, `attention_demo`, `attention_compare_demo` | runnable walkthroughs |
| `benchmark` and helpers | the comparison against Hugging Face static batching |
