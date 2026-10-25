"""
tinyserve: a small vLLM-style inference engine in one file.

This is a simpler implementation of Paged KV Cache with prefix caching, 
continuous batching with chunked prefill, FlashAttention (or a pure torch reference), 
CUDA graphs for decoding and an OpenAI-compatible HTTP Server
"""


import argparse #argpasr is used for parsing command line arguments
import asyncio #asyncio is used for asynchronous programming
import glob #glob is used for file pattern matching, as in finding all files in a directory that match a certain pattern
import json #json is used for parsing and generating JSON data
import os #os is used for interacting with the operating system, such as reading environment variables and file paths
import queue #queue is used for creating and managing queues. queues are useful for managing tasks and data between different threads or processes
import random 
import threading #threading is used for creating and managing threads. threads allow for concurrent execution of code, which can improve performance in certain scenarios
import time #time is used for measuring time intervals, such as how long a certain operation takes to complete
import uuid #uuid is used for generating unique identifiers, which can be useful for tracking requests or sessions in a distributed system
from collections import OrderedDict, deque 
from dataclasses import dataclass
import traceback
import warnings
import gc 

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers import AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset

from fastapi import Request as HTTPRequest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False



@dataclass
class Config:

    model: str
    dtype: torch.dtype = torch.bfloat16 # bfloat16 inference by default
    max_model_len: str = 4096 # max prompt + output we accept
    attention_backend: str = "flash_attn" # "flash_attn" (fast) | "torch" (reference)

    # toks per kv-cache page. 0 = pick for backend, flash-attn only supports
    # pages that are multiples of 256, vllm actually has its own fork for
    # smaller pages (more efficient if we use shorter prompts on average)
    # the torch reference code can use anything
    block_size: int = 0 
    max_num_seqs: int = 128 # max requests we can have in a batch

    # token budget per step, 
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = True
    seed: int = 0


    def __post_init__(self):
        # if flash_attn not available dont use
        if self.attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            warnings.warn("flash attention not available, default to torch")
            self.attention_backend = "torch"

        # set page size as small as possible based on backend
        if self.block_size == 0:
            self.block_size = 256 if self.attention_backend == "flash_attn" else 16

        # our manually looped torch implementation cant be cuda graphed
        if self.attention_backend == "torch":
            self.use_cuda_graphs = False  # its Python loop can't be captured

@dataclass
class SamplingParams:
    temperature: float = 1.0 # 0 means greedy decode
    top_p: float = 1.0
    top_k: int = -1          # -1 means off
    max_tokens: int = 64     # max tokens to gen
    ignore_eos: bool = False # normally we stop gen on eos token

    ### for testing ###
    forced_token_ids: list[int] | None = None # teacher forcing to compare log probs
    return_logprobs: bool = False


class Request: 
    """
    This is the metadata that is needed by the scheduler about a single request.

    The main field we care about is num_computed_tokens, it is how 
    many token_ids already have K/V in the cache, each step the scheduler
    picks how many new tokens to process for this request:

    for eg,

    if we have a prompt of 1000 tokens but a budget of 512,
    then we will first do 512 tokens, then the remaining 488 (this is chunked prefill),
    afterwards during generation it is one token per step
    """

    def __init__(self, request_id: str, #just a unique identifier for each request
                 prompt_ids: list[int], #input tokens for specific prompts
                 params: SamplingParams): #stuff you have when decoding with an llm, like temp, top_p, top_k, etc

        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        #prompt + generated so far
        self.token_ids = list(prompt_ids)
        self.params = params

        #how many of those tokens already have K/V written
        self.num_computed_tokens = 0

        #this requests page table(what blocks contain the KV cache for this specific request?)
        self.block_ids: list[int] = []

        #One has per full block, for prefix reuse
        self.block_hashes: list[int] = [] 
        self.num_cached_tokens = 0 #prefix-cache hits at admisson
        self.logprobs: list[torch.Tensor] =[] 
        self.finished = False
        self.finish_reason: str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # everything after the prompt is what was generated
        return self.token_ids[self.prompt_len:]


class KVCacheManager:
    #Manage which physical Kv-cache block each request owns

    def __init__(self, num_blocks, block_size, enable_caching):

        self.block_size = block_size
        self.enable_caching = enable_caching

        #therefore block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        #if a physical block contains a cached prefix, what prefix does it represent
        self.block_hash : list[int | None] = [None] * num_blocks

        #ordered dict is double linked list, allows for O(1) pop-front
        #and O(1) removal form the middle
        #first we start at index 1, as block 0 is the null block used for padding tokens
        #their value doesn't matter as we mask it during attention anyway

        #This is effectively our LRU queue, it is a list of our unused blocks
        self.free : OrderedDict[int, None] = OrderedDict((i, None) for i in range(1, num_blocks))

        #given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int, int] = {} 



        


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["serve", "generate", "kv-demo", "attn-demo", "bench"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attention-backend", default="flash_attn", choices=["flash_attn", "torch"])
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--bench-num-prompts", type=int, default=100)
    ap.add_argument("--bench-bins", default="1,64,512,2048",
                    help="prompt length bin edges in tokens, prompts are split equally across bins")
    ap.add_argument("--bench-max-tokens", type=int, default=512)
    ap.add_argument("--bench-hf-batch-size", type=int, default=16)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="bench: every request generates exactly --bench-max-tokens")
    ap.add_argument("--bench-prefix-tokens", type=int, default=256,
                    help="bench: length of the system prompt shared by every request (0 = none)")
    ap.add_argument("--bench-raw", action="store_true",
                    help="bench: use the raw instruction text instead of the chat template")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.command == "kv-demo":       # pure bookkeeping: no model, no GPU
        kv_cache_demo()
        return
    if a.command == "attn-demo":     # needs a GPU, but no model weights
        attention_demo()
        return
    if a.command == "bench":         # tinyserve vs HF static batching, offline
        benchmark(a)
        return

    cfg = Config(model=a.model, attention_backend=a.attention_backend,
                 max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                 max_num_batched_tokens=a.max_num_batched_tokens,
                 gpu_memory_utilization=a.gpu_memory_utilization,
                 use_cuda_graphs=not a.no_cuda_graphs,
                 enable_prefix_caching=not a.no_prefix_caching)
    engine = Engine(cfg)
    if a.command == "generate":
        out = engine.generate([a.prompt], SamplingParams(temperature=a.temperature,
                                                         max_tokens=a.max_tokens))[0]
        print(a.prompt + out.text)
        return
    
    import uvicorn
    print(f"[tinyserve] serving {a.model} on http://localhost:{a.port}")
    uvicorn.run(build_app(AsyncEngine(engine), a.model), host="0.0.0.0", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()