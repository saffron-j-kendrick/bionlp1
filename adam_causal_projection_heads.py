from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from nnsight import LanguageModel

from adam_causal_projection import read_dataset, dataset_fingerprint, find_spans, token_index_for_span, locate_target_tokens

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
N_LAYERS = 32
N_HEADS = 32
D_HEAD = 128

REQUIRED_COLUMNS = {"ABBR", "Sentence_A", "Sentence_C", "target"}

## ARGS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--dataset", type = Path, default = Path("adam_dataset_patching_filled2_small.csv"))
    p.add_argument("--output-prefix", type = Path, default = Path("outputs/causal_projection_heads_llama8b"))
    p.add_argument("--accumulator", type = Path)
    p.add_argument("--start-row", type = int, default = 0)
    p.add_argument("--max-rows", type = int, default = 0)
    p.add_argument("--patch-batch-size", type = int, default = 24)
    p.add_argument("--cache-dtype", choices = ("float32", "float16", "bfloat16"), default = "bfloat16")
    p.add_argument("--epsilon", type = float, default = 1e-6)
    p.add_argument("--device-map", default = "auto")
    p.add_argument("--dry-run", action = "store_true")
    p.add_argument("--plot-only", action = "store_true", help = "plot from the accumulator without loading a model")
    return p.parse_args()




def to_numpy(value: Any) -> np.ndarray:
    value = value.value if hasattr(value, "value") else value
    if isinstance(value, np.ndarray):
        return value
    return value.detach().float().cpu().numpy()


def attention_output_input(model: LanguageModel, layer: int) -> Any:
    """Input to o_proj, arranged as concatenated per-head value outputs."""
    return model.model.layers[layer].self_attn.o_proj.input


