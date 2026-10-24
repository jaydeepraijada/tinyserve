"""
tinyserve: a small vLLM-style inference engine in one file.

This is a simpler implementation of Paged KV Cache with prefix caching, 
continuous batching with chunked prefill, FlashAttention (or a pure torch reference), 
CUDA graphs for decoding and an OpenAI-compatible HTTP Server
"""


import argparse
import asyncio
import glob
import json
import os
import queue
import random
import threading
import time
import uuid
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