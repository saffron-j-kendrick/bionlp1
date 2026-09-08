## IMPORTS

from IPython.display import clear_output
import nnsight
from nnsight import CONFIG
from nnsight import LanguageModel, util
from nnsight.intervention.tracing.globals import Object
import plotly.express as px
import plotly.io as pio
import numpy as np
import torch
import kaleido
import matplotlib.pyplot as plt
import seaborn as sns
import einops
import pickle
import pandas as pd
import argparse
import json
import os
import random
import gc

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
})

## PARSER ARGS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3b", help="Model to use: gpt2")
parser.add_argument("--intervention", type=str, default="residual_stream", help="Intervention to use: residual_stream, attention_heads")
parser.add_argument("--dataset", type=str, default="extended_dataset.json", help="Dataset to use: data/combined_dataset.json")
parser.add_argument("--averaging", type=bool, default=True, help="Whether to average the intervention results over the dataset")
parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "gpu"], help="Device placement: auto, cpu, or gpu")
parser.add_argument("--cache_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Precision for cached intervention tensors")

## FUNCTIONS

def _to_numpy(saved_or_tensor):
    """Resolve nnsight-saved objects or tensors to a CPU numpy array."""
    tensor = saved_or_tensor.value if hasattr(saved_or_tensor, "value") else saved_or_tensor
    
    if isinstance(tensor, (tuple, list)) and len(tensor) == 1:
        tensor = tensor[0]
    if isinstance(tensor, np.ndarray):
        return tensor
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().cpu().numpy()

def _to_scalar(saved_or_tensor):
    """Resolve nnsight-saved objects or tensors to a Python float."""
    tensor = saved_or_tensor.value if hasattr(saved_or_tensor, "value") else saved_or_tensor
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().item()
    if isinstance(tensor, np.ndarray):
        return float(tensor.item())
    return float(tensor)

def _dtype_from_name(dtype_name: str):
    """ Precision types mapping """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]

def _find_delimiter_token_index(tokenizer, text, delimiter=","):
    """Tokenizer-agnostic delimiter lookup using character offsets."""
    char_idx = text.find(delimiter)
    if char_idx == -1:
        raise ValueError(f"Delimiter '{delimiter}' not found in text.")

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset mapping.")

    for tok_idx, (start, end) in enumerate(offsets):
        # Skip special tokens that often map to (0, 0)
        if start == end:
            continue
        if start <= char_idx < end:
            return tok_idx

    raise ValueError(f"Could not map delimiter '{delimiter}' to token index.")



def plot_attention_patching_results(model, model_name, ioi_patching_results, x_labels, prompt_idiom, plot_title="Normalized Logit Difference"):
    """ Plot the attention head patching results """

    if model_name in ("gpt2", "gpt2med"):
        N_LAYERS = len(model.transformer.h)
    elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
        N_LAYERS = len(model.model.layers)
    else:
        raise ValueError(f"Invalid model: {model_name}")
    unwrapped_results = []
    for layer_results in ioi_patching_results:
        layer_values = []
        for result in layer_results:
            # Check if it's already a float/int or if it's a Proxy
            if isinstance(result, Object):
                layer_values.append(result.value)
            else:
                layer_values.append(result)
        unwrapped_results.append(layer_values)

    data = np.array(unwrapped_results)

    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels, # Shows layer numbers
        cmap="RdBu",
        center=0.0,
        cbar_kws={'label': 'Norm. Logit Diff'}
    )

    
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")

    # Rotate x-labels if they are crowded
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention.eps")
    plt.show()

    return plt.gcf()



def attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, model_name, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len):
    """ Replaces the attention head at the end of each layer with the attention head from the clean prompt """
    batch = 1
    # clean run
    z_hs = {}
    with model.trace() as tracer:
        with tracer.invoke(prompt_idiom) as invoker:
            clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            for layer_idx in range(N_LAYERS):
                if model_name in ("gpt2", "gpt2med"):
                    z = model.transformer.h[layer_idx].attn.c_proj.input
                elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                    z = model.model.layers[layer_idx].self_attn.o_proj.input
                else:
                    raise ValueError(f"Invalid model: {model_name}")
                z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                for head_idx in range(N_HEADS):
                    z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
            
            clean_logits = model.lm_head.output
            clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
            print(f"clean logit diff {clean_logit_diff}")
    
    # Extract values from Proxy objects after trace context exits
    z_hs_np = {}
    for (layer_idx, head_idx), proxy_obj in z_hs.items():
        z_hs_np[layer_idx, head_idx] = _to_numpy(proxy_obj)
    
    # Now pickle the actual numpy arrays
    z_hs_file = open(f"z_hs_{model_name}.pkl", "wb")
    pickle.dump(z_hs_np, z_hs_file)
    z_hs_file.close()

    # corrupted run
    with model.trace() as tracer:
        with tracer.invoke(prompt_literal) as invoker:
            corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            corrupted_logits = model.lm_head.output
            corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
            print(f"corrupt logit diff {corrupted_logit_diff}")
    
    
    #patching
    attention_head_patching_intervention = []
    #load pickle
    z_hs_file = open(f"z_hs_{model_name}.pkl", "rb")
    z_hs_np = pickle.load(z_hs_file)
    z_hs_file.close()
    
    # Convert numpy arrays back to torch tensors
    z_hs = {}
    for (layer_idx, head_idx), np_array in z_hs_np.items():
        z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
    
    with model.trace() as tracer:
        for layer_idx in range(N_LAYERS):
            _attention_head_patching_intervention = []
            for head_idx in range(N_HEADS):
                with tracer.invoke(prompt_literal) as invoker:
                    if model_name in ("gpt2", "gpt2med"):
                        z = model.transformer.h[layer_idx].attn.c_proj.input
                    elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                        z = model.model.layers[layer_idx].self_attn.o_proj.input
                    else:
                        raise ValueError(f"Invalid model: {model_name}")
                    z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                    z_corrupt[:,:,head_idx,:] = z_hs[layer_idx, head_idx]
                    patched_logits = model.lm_head.output
                    patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                    patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                    _attention_head_patching_intervention.append(patched_result.item().save())
            attention_head_patching_intervention.append(_attention_head_patching_intervention)
    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens[:min_token_len]]
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    fig = plot_attention_patching_results(model, model_name, attention_head_patching_intervention, x_labels, prompt_idiom, f"Patching {model_name} Attention Head on Idiomatic Prompts")
    return fig
    
def run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name):
    for pair in dataset:
        prompt_idiom = pair["prompt_idiom"]
        prompt_literal = pair["prompt_literal"]
        correct_answer = pair["correct_answer"]
        incorrect_answer = pair["incorrect_answer"]

        # tokens
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        min_token_len = min(len(idiom_tokens), len(literal_tokens))

        correct_token = model.tokenizer.encode(correct_answer)[0]
        incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

        correct_answer_idx = correct_token
        incorrect_answer_idx = incorrect_token

        attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, model_name, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len)


def average_attention_head_patching(
    N_LAYERS,
    N_HEADS,
    D_HEADS,
    dataset,
    model_name,
    cache_dtype=torch.float32,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    accumulated_results, total_combinations = _load_accumulator(accumulator_path, N_LAYERS, N_HEADS)
    processed_idioms = 0
    for idiom_idx, idiom_entry in enumerate(dataset):
        if idiom_idx < start_idiom:
            continue
        if max_idioms > 0 and processed_idioms >= max_idioms:
            break
        processed_idioms += 1
        idiom_id = idiom_entry["id"]
        pairs = idiom_entry["pairs"]
        for pair_idx, pair in enumerate(pairs):
            if max_pairs_per_idiom > 0 and pair_idx >= max_pairs_per_idiom:
                break
            prompt_idiom = pair["prompt_idiom"]
            prompt_literal = pair["prompt_literal"]
            idiom_answers = pair["idiom_answers"]
            literal_answers = pair["literal_answers"]
            
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            min_token_len = min(len(idiom_tokens), len(literal_tokens))
            
            n_answers = min(max_answers, len(idiom_answers), len(literal_answers))
            for answer_idx in range(n_answers):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                if model_name == "gpt2" or model_name == "gpt2med" or model_name == "falcon7b" or model_name == "qwen7b" or model_name == "qwen3b":
                    correct_token = model.tokenizer.encode(correct_answer)[0]
                    incorrect_token = model.tokenizer.encode(incorrect_answer)[0]
                else:
                    correct_token = model.tokenizer.encode(correct_answer)[1]
                    incorrect_token = model.tokenizer.encode(incorrect_answer)[1]
           
                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token
                total_combinations += 1


                # clean / corrupted baseline runs
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom) as invoker:
                        clean_logits = model.lm_head.output
                        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"clean logit diff {clean_logit_diff}")
                    with tracer.invoke(prompt_literal) as invoker:
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")

                denom = abs(_to_scalar(clean_logit_diff) - _to_scalar(corrupted_logit_diff))
                if denom < 1.0:
                    print(f"Skipping degenerate pair (denom={denom:.4f}): {prompt_idiom[:60]}")
                    continue

                # Memory-lean streaming patching: process one head at a time.
                for layer_idx in range(N_LAYERS):
                    for head_idx in range(N_HEADS):
                        with model.trace() as tracer:
                            with tracer.invoke(prompt_idiom) as invoker:
                                if model_name in ("gpt2", "gpt2med"):
                                    z_clean = model.transformer.h[layer_idx].attn.c_proj.input
                                elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                                    z_clean = model.model.layers[layer_idx].self_attn.o_proj.input
                                else:
                                    raise ValueError(f"Invalid model: {model_name}")
                                z_clean_head = einops.rearrange(
                                    z_clean, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS
                                )[:, :min_token_len, head_idx, :].save()

                            with tracer.invoke(prompt_literal) as invoker:
                                if model_name in ("gpt2", "gpt2med"):
                                    z = model.transformer.h[layer_idx].attn.c_proj.input
                                elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                                    z = model.model.layers[layer_idx].self_attn.o_proj.input
                                else:
                                    raise ValueError(f"Invalid model: {model_name}")

                                z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
                                clean_head_tensor = torch.from_numpy(_to_numpy(z_clean_head)).to(cache_dtype)
                                if clean_head_tensor.ndim == 2:
                                    clean_head_tensor = clean_head_tensor.unsqueeze(0)
                                actual_seq_len = min(clean_head_tensor.shape[1], min_token_len)
                                z_corrupt[:, :actual_seq_len, head_idx, :] = clean_head_tensor[:, :actual_seq_len, :].to(z_corrupt.dtype)

                                patched_logits = model.lm_head.output
                                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
                                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                                accumulated_results[layer_idx, head_idx] += _to_scalar(patched_result.save())

                        gc.collect()
                _save_accumulator(accumulator_path, accumulated_results, total_combinations)
                
    if total_combinations == 0:
        raise ValueError("No combinations were processed. Check start/max arguments.")
    average_patching_results = accumulated_results / total_combinations

    if model_name =="llama3b":
        flat_results = average_patching_results.flatten()
        top_values, top_indices = torch.topk(flat_results, 67)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 67 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_67_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)


        
        top_values, top_indices = torch.topk(flat_results, 168)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 168 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_168_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
    elif model_name=="gpt2":
        flat_results = average_patching_results.flatten()
        top_values, top_indices = torch.topk(flat_results, 14)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 14 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_14_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
        top_values, top_indices = torch.topk(flat_results, 36)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 36 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_36_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
    elif model_name =="gpt2med":
        flat_results = average_patching_results.flatten()
        top_values, top_indices = torch.topk(flat_results, 38)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 38 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_38_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
        
        top_values, top_indices = torch.topk(flat_results, 96)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 96 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_96_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)

    elif model_name =="qwen3b":
        flat_results = average_patching_results.flatten()
        top_values, top_indices = torch.topk(flat_results, 58)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 58 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_58_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)

        top_values, top_indices = torch.topk(flat_results, 144)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 144 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_144_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
    elif model_name =="qwen7b":
        flat_results = average_patching_results.flatten()
        top_values, top_indices = torch.topk(flat_results, 78)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 78 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_78_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)
            
        top_values, top_indices = torch.topk(flat_results, 196)

        # Convert flat indices back to (layer, head) tuples
        top_heads = []
        for idx in top_indices:
            layer = idx.item() // N_HEADS
            head = idx.item() % N_HEADS
            top_heads.append((layer, head))

        print(f"Top 196 heads to patch/zero: {top_heads}")
        # save as a json file
        with open(f"top_196_heads_{model_name}.json", "w") as f:
            json.dump(top_heads, f)

    return average_patching_results




def run_average_attention_head_patching(
    dataset,
    N_LAYERS,
    N_HEADS,
    D_HEADS,
    model_name,
    cache_dtype=torch.float32,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    average_patching_results = average_attention_head_patching(
        N_LAYERS, N_HEADS, D_HEADS, dataset, model_name,
        cache_dtype=cache_dtype,
        max_idioms=max_idioms,
        max_pairs_per_idiom=max_pairs_per_idiom,
        max_answers=max_answers,
        start_idiom=start_idiom,
        accumulator_path=accumulator_path,
    )
    x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
    prompt_idiom = f"averaged_attention_head_patching_{model_name}_multiple"
    marginalised_layers = torch.mean(average_patching_results, dim=1)
    # plot the marginalised layers
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(10, 6))
    plt.bar(y_labels, marginalised_layers.tolist(), color='skyblue', edgecolor='navy')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Normalized Logit Difference')
    #plt.title(f'Marginalised Layer Importance (Average of {len(dataset)} Idioms for {model_name})')
    plt.xticks(y_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"figures/marginalised_layers_{model_name}_multiple.png")
    plt.savefig(f"figures/marginalised_layers_{model_name}_multiple.eps")
    plt.show()

    # calculate the highest value per layer
    highest_values = torch.max(average_patching_results, dim=1)
    highest_values_list = highest_values.values
    # plot the highest values per layer
    plt.figure(figsize=(10, 6))
    plt.bar(y_labels, highest_values_list.tolist(), color='skyblue', edgecolor='navy')
    plt.xlabel('Layer Index')
    plt.ylabel('Highest Normalized Logit Difference')
    #plt.title(f'Highest Importance per Layer (Average of {len(dataset)} Idioms for {model_name})')
    plt.xticks(y_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"figures/highest_values_{model_name}_multiple.png")
    plt.savefig(f"figures/highest_values_{model_name}_multiple.eps")

    fig = plot_attention_patching_results(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Average {model_name} Attention Head Patching across {len(dataset)} Idioms")
    return fig