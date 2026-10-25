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
    max_model_len: int = 4096 # max prompt + output we accept
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

    @property
    def num_free(self):
        return len(self.free)

    def _update_hashes(self, req):

        bs = self.block_size
        while len(req.block_hashes) < req.num_tokens // bs:
            i = len(req.block_hashes)
            parent = req.block_hashes[-1] if req.block_hashes else None
            req.block_hashes.append(
                hash(
                    (parent, tuple(req.token_ids[i * bs: (i + 1) * bs]))
                )
            )
    def find_cached_prefix(self, req):

        if not self.enable_caching:
            return []

        self._update_hashes(req)

        # num_token = 8 (8-1)//4 == 1
        max_blocks = (req.num_tokens - 1) // self.block_size

        hits = []
        for h in req.block_hashes[:max_blocks]:
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hits.append(b)

        return hits

    # This is  
    def allocate(self, req, num_new_tokens, cached=()):

        total = (
            req.num_computed_tokens 
            + len(cached) * self.block_size
            + num_new_tokens
        )

        total_needed_blocks = -(-total // self.block_size)

        need = total_needed_blocks - len(req.block_ids) - len(cached)
        reclaim = sum(1 for b in cached if self.ref_cnt[b] == 0)

        if need + reclaim > self.num_free:
            return False

        # claim the cached blocks
        for b in cached:

            if self.ref_cnt[b] == 0:
                del self.free[b]

            self.ref_cnt[b] += 1
            req.block_ids.append(b)

        # now we allocate new blocks

        for _ in range(max(need, 0)):
            b, _ = self.free.popitem(last = False) # gives the first item

            if self.block_hash[b] is not None:
                self.hash_to_block.pop(self.block_hash[b], None)
                self.block_hash[b] = None

            self.ref_cnt[b] = 1
            req.block_ids.append(b)

        self._register_full_blocks(req, total)
        return True

    def _register_full_blocks(self, req, num_tokens):
        if not self.enable_caching:
            return 

        self._update_hashes(req)

        for i in range(min(num_tokens, req.num_tokens)// self.block_size):
            b, h = req.block_ids[i], req.block_hashes[i]

            if self.block_hash[b] is None:
                self.block_hash[b] = h
                self.hash_to_block.setdefault(h,b)

    def free_request(self, req):
        for b in reversed(req.block_ids):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free[b] = None
        req.block_ids = []


@dataclass
class ScheduledRequest:
    req: Request # the live request
    num_computed: int # KV present before this forward pass
    num_new: int # how many tokens this forward will process

    @property
    def complete_prompt(self):
        """
        True: This step reaches the end of the current sequence (finish prefill or decode). 
        The sampled next token is real and we append it.

        False: chunked prefill stops mid prompt, you still run those tokens to write KV, but 
        the last positions sampled token is junk and we throw it away, the next step continues the prompt

        num_tokens here is len(token_ids) which during prefill is just the prompt,
        after a real sample, the scheduler's caller appends a token, so next step num_tokens grows by 1 and decode
        is num_new = 1
        """

        return self.num_computed + self.num_new >= self.req.num_tokens


class Scheduler:

    def __init__(self, cfg, kv):

        self.cfg = cfg
        self.kv = kv
        self.waiting = deque()
        self.running = []

    def add(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def schedule(self):

        budget = self.cfg.max_num_batched_tokens

        # for each request id, how many tokens will we run this step?
        plan = {}

        # if we evict anyone don't admit new work
        preempted = False

        # running requests
        i = 0

        while i < len(self.running) and budget > 0:

            # what request?
            req = self.running[i]

            n = min(req.num_tokens - req.num_computed_tokens, budget)

            while not self.kv.allocate(req, n):
                # A B C -> A B | C (C is the victim)
                victim = self.running.pop()

                self.kv.free_request(victim)
                victim.num_computed_tokens = 0
                self.waiting.appendleft(victim)
                budget += plan.pop(victim.request_id, 0)
                preempted = True

                if victim is req:
                    break

            if req not in self.running:
                break

            plan[req.request_id] = n
            budget -= n
            i +=1 

        # currently waiting request
        while(not preempted and self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs):
            req = self.waiting[0]

            cached = self.kv.find_cached_prefix(req)
            n = min(req.num_tokens - len(cached) * self.cfg.block_size, budget)

            if not self.kv.allocate(req, n, cached):
                break

            self.waiting.popleft()

            req.num_computed_tokens = len(cached) * self.cfg.block_size
            req.num_cached_tokens = max(req.num_cached_tokens, req.num_computed_tokens)

            self.running.append(req)

            plan[req.request_id] = n
            budget -= n

        # emit the batch in running order
        batch = []
        for req in self.running:
            if req.request_id in plan:
                batch.append(ScheduledRequest(
                    req, req.num_computed_tokens, plan[req.request_id]
                ))
        return batch

    def finish(self, req: Request, reason: str) -> None:
        req.finished, req.finish_reason = True, reason
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free_request(req)

@dataclass
class AttnMetadata:
    slot_mapping: torch.Tensor #T
    query_start_loc: torch.Tensor # [B + 1]
    cu_seqlens_k: torch.Tensor
    block_table: torch.Tensor # [B x max_blocks]
    num_seqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc_cpu: list[int]
    seq_lens_cpu: list[int]

def torch_ref_attention(q, # T x H x D
                        k_cache, # [n_pages x bs x Hkv x D]
                        v_cache, # [n_pages x bs x Hkv x D]
                        meta, # query_start_loc, seq_lens, block_table
                        scale # 1/sqrt(D)
                        ):
    out = torch.empty_like(q)

    # H = query heads, Hkv = KV heads (GQA if H > Hkv), bs = page size
    H, Hkv, bs = q.shape[1], k_cache.shape[2], k_cache.shape[1]

    # qsl[i]:qsl[i+1] = this request's slice of packed q
    # seq_lens[i] = full context length after this step (kv_len)
    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu

    for i in range(meta.num_seqs):
        s, e, kv_len = qsl[i], qsl[i+1], seq_lens[i]
        q_len = e - s 

        blocks = meta.block_table[i, : -(-kv_len//bs)].long()

        # k_cache[blocks] -> [16 x Hkv x D]
        k = k_cache[blocks].flatten(0,1)[:kv_len]
        v = v_cache[blocks].flatten(0,1)[:kv_len]

        k = k.transpose(0, 1).repeat_interleave(H // Hkv, 0)
        v = v.transpose(0, 1).repeat_interleave(H // Hkv, 0)


        # full prefill: q_len == kv_len -> square causal mask
        # chunked_prefill: q_len < kv_len -> suffix strip
        # decode: q_len == 1 -> 1 row, all keys 
        q_pos = torch.arange(kv_len - q_len, kv_len, device = q.device)
        mask = torch.arange(kv_len, device = q.device)[None, :] <= q_pos[:, None]

        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k, v, attn_mask=mask, scale = scale
        )

        out[s:e] = o.transpose(0,1)

    return out 

def flash_attention(q, k_cache, v_cache, meta, scale):

    """
    identical to the torch_ref but using flash_attn
    """

    return flash_attn_varlen_func(
        q, k_cache, v_cache,
        cu_seqlens_q = meta.query_start_loc, cu_seqlens_k = meta.cu_seqlens_k,
        max_seqlen_q = meta.max_query_len, max_seqlen_k = meta.max_seq_len,
        softmax_scale = scale, causal = True, block_table = meta.block_table
    )

### Some global variables for selecting attention
ATTENTION_BACKENDS = {'flash_attn': flash_attention, "torch": torch_ref_attention}
ATTN_META: AttnMetadata | None = None
ATTN_FN = flash_attention


def attention_demo():
    """
    Check paged attention against plain dense attention on random data, no model needed.

    One batch mixes all three cases the engine produces:
    full prefill (q_len == kv_len), chunked prefill (q_len < kv_len) and decode (q_len == 1)
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    H, Hkv, D = 8, 2, 64 # 8 query heads sharing 2 kv heads (GQA)
    scale = D ** -0.5
    cases = [("prefill", 37, 37), ("chunked", 20, 50), ("decode", 1, 70)] # (name, q_len, kv_len)

    backends = [("torch", 16, torch.float32)]
    if FLASH_ATTN_AVAILABLE and device == "cuda":
        backends.append(("flash_attn", 256, torch.bfloat16)) # flash needs pages of 256

    for backend, bs, dtype in backends:
        num_pages = 1 + sum(-(-kv_len // bs) for _, _, kv_len in cases) # +1 for the null block 0
        k_cache = torch.zeros(num_pages, bs, Hkv, D, device=device, dtype=dtype)
        v_cache = torch.zeros_like(k_cache)

        # hand out pages in shuffled order so each sequence is scattered across the cache
        free_pages = (torch.randperm(num_pages - 1) + 1).tolist()
        max_blocks = max(-(-kv_len // bs) for _, _, kv_len in cases)
        block_table = torch.zeros(len(cases), max_blocks, dtype=torch.int32)

        qs, ks, vs, slots = [], [], [], []
        for i, (_, q_len, kv_len) in enumerate(cases):
            pages = [free_pages.pop() for _ in range(-(-kv_len // bs))]
            block_table[i, :len(pages)] = torch.tensor(pages, dtype=torch.int32)

            k = torch.randn(kv_len, Hkv, D, device=device, dtype=dtype)
            v = torch.randn(kv_len, Hkv, D, device=device, dtype=dtype)
            for pos in range(kv_len): # token pos lives in page pos // bs, offset pos % bs
                k_cache[pages[pos // bs], pos % bs] = k[pos]
                v_cache[pages[pos // bs], pos % bs] = v[pos]

            qs.append(torch.randn(q_len, H, D, device=device, dtype=dtype))
            ks.append(k)
            vs.append(v)
            slots += [pages[pos // bs] * bs + pos % bs for pos in range(kv_len - q_len, kv_len)]

        q_lens = [q_len for _, q_len, _ in cases]
        seq_lens = [kv_len for _, _, kv_len in cases]
        qsl = [0]
        for n in q_lens:
            qsl.append(qsl[-1] + n)
        cu_k = [0]
        for n in seq_lens:
            cu_k.append(cu_k[-1] + n)

        meta = AttnMetadata(
            slot_mapping=torch.tensor(slots, device=device),
            query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
            block_table=block_table.to(device),
            num_seqs=len(cases),
            max_query_len=max(q_lens),
            max_seq_len=max(seq_lens),
            query_start_loc_cpu=qsl,
            seq_lens_cpu=seq_lens,
        )

        q = torch.cat(qs)
        out = ATTENTION_BACKENDS[backend](q, k_cache, v_cache, meta, scale)

        print(f"[attn-demo] backend={backend} device={device} page_size={bs} dtype={dtype}")
        for i, (name, q_len, kv_len) in enumerate(cases):
            # dense reference: the query at position p sees keys 0..p
            k = ks[i].float().transpose(0, 1).repeat_interleave(H // Hkv, 0)
            v = vs[i].float().transpose(0, 1).repeat_interleave(H // Hkv, 0)
            scores = qs[i].float().transpose(0, 1) @ k.transpose(1, 2) * scale
            q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
            causal = torch.arange(kv_len, device=device)[None, :] <= q_pos[:, None]
            ref = (scores.masked_fill(~causal, float("-inf")).softmax(-1) @ v).transpose(0, 1)

            err = (out[qsl[i]:qsl[i + 1]].float() - ref).abs().max().item()
            tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
            print(f"  {name:8s} q_len={q_len:3d} kv_len={kv_len:3d}  max_err={err:.2e}  "
                  f"{'OK' if err < tol else 'MISMATCH'}")

        







     





        


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
    # if a.command == "bench":         # tinyserve vs HF static batching, offline
    #     benchmark(a)
    #     return

    # cfg = Config(model=a.model, attention_backend=a.attention_backend,
    #              max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
    #              max_num_batched_tokens=a.max_num_batched_tokens,
    #              gpu_memory_utilization=a.gpu_memory_utilization,
    #              use_cuda_graphs=not a.no_cuda_graphs,
    #              enable_prefix_caching=not a.no_prefix_caching)
    # engine = Engine(cfg)
    # if a.command == "generate":
    #     out = engine.generate([a.prompt], SamplingParams(temperature=a.temperature,
    #                                                      max_tokens=a.max_tokens))[0]
    #     print(a.prompt + out.text)
    #     return
    
    # import uvicorn
    # print(f"[tinyserve] serving {a.model} on http://localhost:{a.port}")
    # uvicorn.run(build_app(AsyncEngine(engine), a.model), host="0.0.0.0", port=a.port,
    #             log_level="warning")


if __name__ == "__main__":
    main()