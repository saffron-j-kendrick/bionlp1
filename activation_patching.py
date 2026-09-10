### IMPORTS

from IPython.display import clear_output
import nnsight
from nnsight import CONFIG
from nnsight import LanguageModel, util
# from nnsight.tracing.graph.proxy import Proxy
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

### PARSER ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3b", help="Model to use: gpt2, gpt2med, llama-3.2-3b, mistral7b, falcon7b, qwen7b, qwen3b")
parser.add_argument("--intervention", type=str, default="residual_stream", help="Intervention to use: residual_stream, attention_heads")
parser.add_argument("--dataset", type=str, default="extended_dataset.json", help="Dataset to use: data/combined_dataset.json")
parser.add_argument("--averaging", type=bool, default=True, help="Whether to average the intervention results over the dataset")
parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "gpu"], help="Device placement: auto, cpu, or gpu")
parser.add_argument("--cache_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Precision for cached intervention tensors")
parser.add_argument("--max_idioms", type=int, default=0, help="If >0, only process this many idioms from dataset")
parser.add_argument("--max_pairs_per_idiom", type=int, default=0, help="If >0, only process this many pairs per idiom")
parser.add_argument("--max_answers", type=int, default=3, help="Number of answer options per pair to evaluate")
parser.add_argument("--start_idiom", type=int, default=0, help="Start index in dataset for batched runs")
parser.add_argument("--accumulator_path", type=str, default="", help="Path to .npz accumulation checkpoint for batched averaging")


### FUNCTIONS

def _to_numpy(saved_or_tensor):
    """Resolve nnsight-saved objects or tensors to a CPU numpy array."""
    tensor = saved_or_tensor.value if hasattr(saved_or_tensor, "value") else saved_or_tensor
    # nnsight resolves module .input as a tuple (tensor,) — unwrap if needed
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

def _load_accumulator(path, n_layers, n_heads):
    if not path or not os.path.exists(path):
        return torch.zeros((n_layers, n_heads)), 0
    data = np.load(path)
    acc = torch.from_numpy(data["accumulated_results"]).to(torch.float32)
    total = int(data["total_combinations"])
    return acc, total

def _save_accumulator(path, accumulated_results, total_combinations):
    if not path:
        return
    np.savez(
        path,
        accumulated_results=accumulated_results.detach().cpu().numpy(),
        total_combinations=np.array(total_combinations, dtype=np.int64),
    )

def _load_residual_accumulator(path, n_layers, window_size):
    if not path or not os.path.exists(path):
        return np.zeros((n_layers, window_size)), np.zeros((n_layers, window_size))
    data = np.load(path)
    if "total_results" in data and "counts" in data:
        return data["total_results"], data["counts"]
    # Backward compatibility: no valid residual accumulator content
    return np.zeros((n_layers, window_size)), np.zeros((n_layers, window_size))

def _save_residual_accumulator(path, total_results, counts):
    if not path:
        return
    np.savez(
        path,
        total_results=total_results,
        counts=counts,
    )

def _load_mlp_accumulator(path, n_layers, window_size):
    if not path or not os.path.exists(path):
        return np.zeros((n_layers, window_size)), np.zeros((n_layers, window_size)), None, None
    data = np.load(path)
    total_results = data["total_results"] if "total_results" in data else np.zeros((n_layers, window_size))
    counts = data["counts"] if "counts" in data else np.zeros((n_layers, window_size))
    mlp_dim_totals = data["mlp_dim_totals"] if "mlp_dim_totals" in data else None
    mlp_dim_counts = data["mlp_dim_counts"] if "mlp_dim_counts" in data else None
    return total_results, counts, mlp_dim_totals, mlp_dim_counts

def _save_mlp_accumulator(path, total_results, counts, mlp_dim_totals, mlp_dim_counts):
    if not path:
        return
    save_dict = dict(total_results=total_results, counts=counts)
    if mlp_dim_totals is not None:
        save_dict["mlp_dim_totals"] = mlp_dim_totals
        save_dict["mlp_dim_counts"] = mlp_dim_counts
    np.savez(path, **save_dict)

def plot_ioi_patching_results(model, model_name,
                              ioi_patching_results,
                              x_labels,
                              prompt_idiom,
                              plot_title="Normalized Logit Difference"):
    if model_name in ("gpt2", "gpt2med"):
        N_LAYERS = len(model.transformer.h)
    elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
        N_LAYERS = len(model.model.layers)
    else:
        raise ValueError(f"Invalid model: {model_name}")

    # 1. Extract values from nnsight Proxies
    # Manually extract values since Proxies are evaluated after trace context
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

    # 2. Convert to a 2D numpy array for plotting
    data = np.array(unwrapped_results)

    # 3. Create the plot
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

    #plt.title(plot_title)
    plt.xlabel("Token Position")
    plt.ylabel("Layer")

    # Rotate x-labels if they are crowded
    plt.xticks(rotation=45, ha='right')

    # Save the file
    plt.tight_layout()
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}.eps")
    plt.show()

    return plt.gcf()

def plot_ioi_patching_results_attention(model, model_name,
                              ioi_patching_results,
                              x_labels,
                              prompt_idiom,
                              plot_title="Normalized Logit Difference"):

    if model_name in ("gpt2", "gpt2med"):
        N_LAYERS = len(model.transformer.h)
    elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
        N_LAYERS = len(model.model.layers)
    else:
        raise ValueError(f"Invalid model: {model_name}")

    # 1. Extract values from nnsight Proxies
    # Manually extract values since Proxies are evaluated after trace context
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

    # 2. Convert to a 2D numpy array for plotting
    data = np.array(unwrapped_results)

    # 3. Create the plot
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

    #plt.title(plot_title)
    plt.xlabel("Attention Head")
    plt.ylabel("Layer")

    # Rotate x-labels if they are crowded
    plt.xticks(rotation=45, ha='right')

    # Save the file
    plt.tight_layout()
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention.eps")
    plt.show()

    return plt.gcf()

def visualize_attention(model, model_name, prompt, layer, head, prompt_idiom):
    with model.trace(prompt) as tracer:
        # Access attention weights: [batch, head, query_pos, key_pos]
        # When output_attentions=True, the attention weights are the 2nd element (index 1)
        # in the output tuple of the GPT2Attention module.
        if model_name in ("gpt2", "gpt2med"):
            attn_weights = model.transformer.h[layer].attn.output[1][0, head].save()
        elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
            attn_weights = model.model.layers[layer].self_attn.output[1][0, head].save()
        else:
            raise ValueError(f"Invalid model: {model_name}")

    # Convert to numpy for plotting
    weights = attn_weights.detach().cpu().numpy()

    # Tokenize for labels
    tokens = [model.tokenizer.decode(t) for t in model.tokenizer.encode(prompt)]
   
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Purples",
        cbar_kws={'label': 'Attention Probability'}
    )
    #plt.title(f"Attention Pattern: Layer {layer}, Head {head}")
    plt.xlabel("Key (Source Token)")
    plt.ylabel("Query (Target Token)")
    plt.xticks(rotation=45)
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention_patterns.png")
    plt.savefig(f"figures/patching_results_{prompt_idiom}_{model_name}_attention_patterns.eps")
    plt.show()
    return plt.gcf()


def residual_stream_patching(N_LAYERS, model_name, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len):
    """ Replaces the residual stream at the end of each layer with the residual stream from the clean prompt """
    with model.trace(prompt_idiom) as tracer:
        clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        if model_name in ("gpt2", "gpt2med"):
            clean_hs = [model.transformer.h[i].output[0].save() for i in range(N_LAYERS)]
        elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
            clean_hs = [model.model.layers[i].output[0].save() for i in range(N_LAYERS)]
        else:
            raise ValueError(f"Invalid model: {model_name}")
        clean_logits = model.lm_head.output
        clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
        print(f"clean logit diff {clean_logit_diff}")
    
    # Convert Proxy objects to numpy arrays after trace context exits
    clean_hs_np = [_to_numpy(act) for act in clean_hs]
    for layer_idx, act in enumerate(clean_hs_np):
        np.save(f"data/clean_hidden_state_{layer_idx}_{model_name}.npy", act)
    
    with model.trace(prompt_literal) as tracer:
        corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        corrupted_logits = model.lm_head.output
        corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
        print(f"corrupted logit diff {corrupted_logit_diff}")

    denom = abs(_to_scalar(clean_logit_diff) - _to_scalar(corrupted_logit_diff))
    if denom < 1.0:
        print(f"Skipping degenerate pair (denom={denom:.4f}): {prompt_idiom[:60]}")
        return None

    residual_stream_patching_intervention = []
    for layer_idx in range(N_LAYERS):
        clean_hs_np = np.load(f"data/clean_hidden_state_{layer_idx}_{model_name}.npy")
        clean_hs = torch.from_numpy(clean_hs_np)
        _residual_stream_patching_intervention = []
        for token_idx in range(min_token_len):
            with model.trace(prompt_literal) as tracer:
                if model_name in ("gpt2", "gpt2med"):
                    model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[:, token_idx, :]
                elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                    model.model.layers[layer_idx].output[0][ token_idx, :] = clean_hs[ token_idx, :]
                else:
                    raise ValueError(f"Invalid model: {model_name}")
                patched_logits = model.lm_head.output
                patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx])
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
                _residual_stream_patching_intervention.append(patched_result.item().save())
        residual_stream_patching_intervention.append(_residual_stream_patching_intervention)
    clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens[:min_token_len]]
    token_labels = [f"{token}_{i}" for i, token in enumerate(clean_decoded_tokens)]
    
    fig = plot_ioi_patching_results(model, model_name, residual_stream_patching_intervention, token_labels, prompt_idiom, f"Patching {model_name} Residual Stream on Idiomatic Prompts")
    return fig


def run_residual_stream_patching(dataset, N_LAYERS, model_name):
    for pair in dataset:
        prompt_idiom = pair["prompt_idiom"]
        prompt_literal = pair["prompt_literal"]
        correct_answer = pair["correct_answer"]
        incorrect_answer = pair["incorrect_answer"]

        # tokens
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
        min_token_len = min(len(idiom_tokens), len(literal_tokens))
        print(len(idiom_tokens), len(literal_tokens))
        print(f"min token len {min_token_len}")

        if model_name in ("gpt2", "gpt2med"):

            correct_token = model.tokenizer.encode(correct_answer)[0]
            incorrect_token = model.tokenizer.encode(incorrect_answer)[0]
        else:
            correct_token = model.tokenizer.encode(correct_answer)[1]
            incorrect_token = model.tokenizer.encode(incorrect_answer)[1]

        correct_answer_idx = correct_token
        incorrect_answer_idx = incorrect_token
        residual_stream_patching(N_LAYERS, model_name, prompt_idiom, prompt_literal, correct_answer_idx, incorrect_answer_idx, min_token_len)


def average_residual_stream_patching(
    N_LAYERS,
    model_name,
    dataset,
    cache_dtype=torch.bfloat16,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    """ Replaces the residual stream at the end of each layer with the residual stream from the clean prompt and plot the averaged results"""
    
    
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1

    total_results, counts = _load_residual_accumulator(accumulator_path, N_LAYERS, WINDOW_SIZE)
    


    # for pair in dataset:
    #     prompt_idiom = pair["prompt_idiom"]
    #     prompt_literal = pair["prompt_literal"]
    #     correct_answer = pair["correct_answer"]
    #     incorrect_answer = pair["incorrect_answer"]
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
          

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            print(len(idiom_tokens), len(literal_tokens))
            min_token_len = min(len(idiom_tokens), len(literal_tokens))
            print(min_token_len)

            # Find delimiter token position in a tokenizer-agnostic way
            try:
                and_token_idx = _find_delimiter_token_index(model.tokenizer, prompt_literal, delimiter=",")
                print(f"comma token idx {and_token_idx}")
            except (RuntimeError, ValueError, IndexError):
                print(f"Skipping idiom={idiom_id}, pair_idx={pair_idx}: comma delimiter not found.")
                continue


            n_answers = min(max_answers, len(idiom_answers), len(literal_answers))
            for answer_idx in range(n_answers):
                correct_answer = idiom_answers[answer_idx]
                incorrect_answer = literal_answers[answer_idx]
                if model_name == "gpt2" or model_name == "falcon7b" or model_name == "qwen7b" or model_name == "qwen3b" or model_name == "gpt2med":
                    correct_token = model.tokenizer.encode(correct_answer)[0]
                    incorrect_token = model.tokenizer.encode(incorrect_answer)[0]
                else:
                    correct_token = model.tokenizer.encode(correct_answer)[1]
                    incorrect_token = model.tokenizer.encode(incorrect_answer)[1]
                correct_answer_idx = correct_token
                incorrect_answer_idx = incorrect_token

            
                clean_diff = []
                corrupt_diff = []
                # clean/corrupt baseline run (single pass)
                with model.trace() as tracer:
                    with tracer.invoke(prompt_idiom):
                        clean_logits = model.lm_head.output
                        clean_logit_diff = (
                            clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]
                        ).save()
                        print(f"clean logit diff {clean_logit_diff}")
                        clean_diff.append(clean_logit_diff)
                    with tracer.invoke(prompt_literal):
                        corrupted_logits = model.lm_head.output
                        corrupted_logit_diff = (
                            corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]
                        ).save()
                        print(f"corrupt logit diff {corrupted_logit_diff}")
                        corrupt_diff.append(corrupted_logit_diff)
                for i in range(len(clean_diff)):
                    for j in range(len(corrupt_diff)):
                        if i == j:
                            print("--------------------------------")
                            print(f"clean diff {clean_diff[i]}")
                            print(f"corrupt diff {corrupt_diff[j]}")
                            print(f"diff {clean_diff[i] - corrupt_diff[j]}")
                            print("--------------------------------")

                denom = abs(_to_scalar(clean_logit_diff) - _to_scalar(corrupted_logit_diff))
                if denom < 1.0:
                    print(f"Skipping degenerate pair (denom={denom:.4f}): {prompt_idiom[:60]}")
                    continue

                # Streaming patching: materialize one layer clean residual at a time
                for layer_idx in range(N_LAYERS):
                    with model.trace(prompt_idiom):
                        if model_name in ("gpt2", "gpt2med"):
                            clean_layer_saved = model.transformer.h[layer_idx].output[0].save()
                        elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                            clean_layer_saved = model.model.layers[layer_idx].output[0].save()
                        else:
                            raise ValueError(f"Invalid model: {model_name}")

                    clean_hs = torch.from_numpy(_to_numpy(clean_layer_saved)).to(cache_dtype)

                    for offset in range(-BEFORE, AFTER + 1):
                        token_idx = and_token_idx + offset
                        matrix_col = offset + BEFORE
                        if 0 <= token_idx < min_token_len:
                            with model.trace(prompt_literal):
                                if model_name in ("gpt2", "gpt2med"):
                                    layer_out = model.transformer.h[layer_idx].output[0]
                                    layer_out[:, token_idx, :] = clean_hs[:, token_idx, :].to(layer_out.dtype)
                                elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
                                    layer_out = model.model.layers[layer_idx].output[0]
                                    if clean_hs.ndim == 3:
                                        layer_out[token_idx, :] = clean_hs[0, token_idx, :].to(layer_out.dtype)
                                    else:
                                        layer_out[token_idx, :] = clean_hs[token_idx, :].to(layer_out.dtype)
                                else:
                                    raise ValueError(f"Invalid model: {model_name}")

                                patched_logits = model.lm_head.output
                                patched_logit_diff = (
                                    patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
                                )
                                patched_result = (
                                    (patched_logit_diff - corrupted_logit_diff)
                                    / (clean_logit_diff - corrupted_logit_diff)
                                )
                                patched_result_saved = patched_result.save()

                            total_results[layer_idx, matrix_col] += _to_scalar(patched_result_saved)
                            counts[layer_idx, matrix_col] += 1
                    del clean_hs
                    gc.collect()
                _save_residual_accumulator(accumulator_path, total_results, counts)
        
    return total_results, counts
        
   



def run_average_residual_stream_patching(
    dataset,
    N_LAYERS,
    model_name,
    cache_dtype=torch.bfloat16,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    
        total_results, counts = average_residual_stream_patching(
            N_LAYERS,
            model_name,
            dataset,
            cache_dtype=cache_dtype,
            max_idioms=max_idioms,
            max_pairs_per_idiom=max_pairs_per_idiom,
            max_answers=max_answers,
            start_idiom=start_idiom,
            accumulator_path=accumulator_path,
        )
        avg_results = np.divide(total_results, counts, out=np.zeros_like(total_results), where=counts!=0)
        # Use first idiom's first pair for labels
        prompt_idiom = dataset[0]["pairs"][0]["prompt_idiom"]
        
        idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
        try:
            and_token_idx = _find_delimiter_token_index(model.tokenizer, prompt_idiom, delimiter=",")
        except ValueError:
            # Fallback: center the label window near sentence end if comma mapping fails.
            and_token_idx = min(len(idiom_tokens) - 1, 8)

        label_start = max(0, and_token_idx - 8)
        label_end = min(len(idiom_tokens), and_token_idx + 5)
        labels = [model.tokenizer.decode(token) for token in idiom_tokens[label_start:label_end]]
        # Calculate the absolute maximum to make the scale symmetric
        y_labels = list(range(1, N_LAYERS + 1))
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            avg_results,
            xticklabels=labels,
            yticklabels=y_labels,
            cmap="RdBu",
            cbar_kws={'label': 'Norm. Logit Diff'},
            center=0
        )
        plt.xlabel("Token position")
        plt.ylabel("Layer")
        #plt.title(f"Dataset-Averaged Activation Patching on {model_name}")
        plt.savefig(f"figures/averaged_patching_results_{model_name}.png")
        plt.savefig(f"figures/averaged_patching_results_{model_name}.eps")
        plt.gca() # Often helpful to have Layer 0 at the bottom
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
    fig = plot_ioi_patching_results_attention(model, model_name, attention_head_patching_intervention, x_labels, prompt_idiom, f"Patching {model_name} Attention Head on Idiomatic Prompts")
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

    fig = plot_ioi_patching_results_attention(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Average {model_name} Attention Head Patching across {len(dataset)} Idioms")
    return fig

def average_mlp_patching(
    N_LAYERS,
    dataset,
    model_name,
    cache_dtype=torch.bfloat16,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    """ patch the MLP layers"""

    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1

    total_results, counts, mlp_dim_totals, mlp_dim_counts = _load_mlp_accumulator(
        accumulator_path, N_LAYERS, WINDOW_SIZE
    )

    is_gpt2_family = model_name in ("gpt2", "gpt2med")
    is_llama_style_family = model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b")
    if not (is_gpt2_family or is_llama_style_family):
        raise ValueError(
            f"Invalid model for average_mlp_patching: {model_name}. "
            "Expected one of: gpt2, gpt2med, llama3b, mistral7b, falcon7b, qwen7b, qwen3b."
        )

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

            # tokens
            idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
            literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
            print(len(idiom_tokens), len(literal_tokens))
            min_token_len = min(len(idiom_tokens), len(literal_tokens))
            print(min_token_len)

            try:
                and_token_idx = _find_delimiter_token_index(model.tokenizer, prompt_literal, delimiter=",")
                print(f"comma token idx {and_token_idx}")
            except (RuntimeError, ValueError, IndexError):
                print(f"Skipping idiom={idiom_id}, pair_idx={pair_idx}: comma delimiter not found.")
                continue

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

                # clean run
                clean_mlp = []
                with model.trace(prompt_idiom) as tracer:
                    if is_gpt2_family:
                        for i in range(N_LAYERS):
                            clean_mlp.append(model.transformer.h[i].mlp.c_proj.input.save())
                    elif is_llama_style_family:
                        for i in range(N_LAYERS):
                            clean_mlp.append(model.model.layers[i].mlp.down_proj.input.save())
                    clean_logits = model.lm_head.output.save()
                    clean_logit_diff = (
                        clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]
                    ).save()
                if not clean_mlp:
                    raise RuntimeError(
                        f"Failed to capture clean MLP activations for model={model_name}. "
                        "The trace block completed without setting clean_mlp."
                    )

                clean_mlp_np = [_to_numpy(act) for act in clean_mlp]
                for layer_idx, act in enumerate(clean_mlp_np):
                    np.save(f"data/clean_mlp_{layer_idx}_{model_name}.npy", act)

                # corrupted run
                corrupted_mlp = []
                with model.trace(prompt_literal) as tracer:
                    if is_gpt2_family:
                        for i in range(N_LAYERS):
                            corrupted_mlp.append(model.transformer.h[i].mlp.c_proj.input.save())
                    elif is_llama_style_family:
                        for i in range(N_LAYERS):
                            corrupted_mlp.append(model.model.layers[i].mlp.down_proj.input.save())
                    corrupted_logits = model.lm_head.output
                    corrupted_logit_diff = (
                        corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]
                    ).save()
                if not corrupted_mlp:
                    raise RuntimeError(
                        f"Failed to capture corrupted MLP activations for model={model_name}. "
                        "The trace block completed without setting corrupted_mlp."
                    )

                denom = abs(_to_scalar(clean_logit_diff) - _to_scalar(corrupted_logit_diff))
                if denom < 1.0:
                    print(f"Skipping degenerate pair (denom={denom:.4f}): {prompt_idiom[:60]}")
                    continue

                if answer_idx == 0:
                    corrupted_mlp_np = [_to_numpy(act) for act in corrupted_mlp]
                    if mlp_dim_totals is None:
                        mlp_dim = clean_mlp_np[0].shape[-1]
                        mlp_dim_totals = np.zeros((N_LAYERS, mlp_dim))
                        mlp_dim_counts = np.zeros((N_LAYERS, mlp_dim))

                    if 0 <= and_token_idx < min_token_len:
                        for layer_idx in range(N_LAYERS):
                            clean_vec = clean_mlp_np[layer_idx][0, and_token_idx, :]
                            corrupted_vec = corrupted_mlp_np[layer_idx][0, and_token_idx, :]
                            mlp_dim_totals[layer_idx, :] += np.abs(clean_vec - corrupted_vec)
                            mlp_dim_counts[layer_idx, :] += 1

                # patching within the window
                for layer_idx in range(N_LAYERS):
                    clean_mlp_layer = np.load(f"data/clean_mlp_{layer_idx}_{model_name}.npy")
                    clean_mlp_tensor = torch.from_numpy(clean_mlp_layer).to(cache_dtype)

                    for offset in range(-BEFORE, AFTER + 1):
                        token_idx = and_token_idx + offset
                        matrix_col = offset + BEFORE
                        if 0 <= token_idx < min_token_len:
                            with model.trace(prompt_literal) as tracer:
                                if is_gpt2_family:
                                    model.transformer.h[layer_idx].mlp.c_proj.input[:, token_idx, :] = \
                                        clean_mlp_tensor[:, token_idx, :].to(
                                            model.transformer.h[layer_idx].mlp.c_proj.input.dtype
                                        )
                                elif is_llama_style_family:
                                    model.model.layers[layer_idx].mlp.down_proj.input[:, token_idx, :] = \
                                        clean_mlp_tensor[:, token_idx, :].to(
                                            model.model.layers[layer_idx].mlp.down_proj.input.dtype
                                        )
                                patched_logits = model.lm_head.output
                                patched_logit_diff = (
                                    patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
                                )
                                patched_result = (
                                    (patched_logit_diff - corrupted_logit_diff)
                                    / (clean_logit_diff - corrupted_logit_diff)
                                )
                                patched_result_saved = patched_result.save()

                            total_results[layer_idx, matrix_col] += _to_scalar(patched_result_saved)
                            counts[layer_idx, matrix_col] += 1

                    del clean_mlp_tensor
                    gc.collect()
                _save_mlp_accumulator(accumulator_path, total_results, counts, mlp_dim_totals, mlp_dim_counts)

    return total_results, counts, mlp_dim_totals, mlp_dim_counts


def run_average_mlp_patching(
    dataset,
    N_LAYERS,
    model_name,
    cache_dtype=torch.bfloat16,
    max_idioms=0,
    max_pairs_per_idiom=0,
    max_answers=3,
    start_idiom=0,
    accumulator_path="",
):
    total_results, counts, mlp_dim_totals, mlp_dim_counts = average_mlp_patching(
        N_LAYERS,
        dataset,
        model_name,
        cache_dtype=cache_dtype,
        max_idioms=max_idioms,
        max_pairs_per_idiom=max_pairs_per_idiom,
        max_answers=max_answers,
        start_idiom=start_idiom,
        accumulator_path=accumulator_path,
    )
    avg_results = np.divide(total_results, counts, out=np.zeros_like(total_results), where=counts!=0)
    avg_mlp_dim_scores = np.divide(
        mlp_dim_totals,
        mlp_dim_counts,
        out=np.zeros_like(mlp_dim_totals),
        where=mlp_dim_counts != 0
    )

    # get the top mlp components to patch
    BEFORE = 8
    AFTER = 4
    WINDOW_SIZE = BEFORE + AFTER + 1
    #flat_results = avg_results.flatten()
    flat_results = torch.from_numpy(avg_results.flatten()).float()
    
    # Top actual MLP neuron dimensions: [layer, mlp_component]
    flat_dim_scores = torch.from_numpy(avg_mlp_dim_scores.flatten()).float()
    mlp_dim_size = avg_mlp_dim_scores.shape[1]

    # get 10% of mlp diimensions
    num_mlp_dimensions = int(mlp_dim_size * 0.1)

    top_values, top_indices = torch.topk(flat_dim_scores, num_mlp_dimensions)
    top_mlp_dimensions = []
    for idx in top_indices:
        layer = idx.item() // mlp_dim_size
        mlp_component = idx.item() % mlp_dim_size
        top_mlp_dimensions.append((layer, mlp_component))
    print(f"Top 10% MLP neuron dimensions: {top_mlp_dimensions}")
    with open(f"top_10_percent_mlp_dimensions_{model_name}.json", "w") as f:
        json.dump(top_mlp_dimensions, f)

    num_mlp_dimensions = int(mlp_dim_size * 0.25)
    top_values, top_indices = torch.topk(flat_dim_scores, num_mlp_dimensions)
    top_mlp_dimensions = []
    for idx in top_indices:
        layer = idx.item() // mlp_dim_size
        mlp_component = idx.item() % mlp_dim_size
        top_mlp_dimensions.append((layer, mlp_component))
    print(f"Top 25% MLP neuron dimensions: {top_mlp_dimensions}")
    with open(f"top_25_percent_mlp_dimensions_{model_name}.json", "w") as f:
        json.dump(top_mlp_dimensions, f)


    # Use first idiom's first pair for labels
    prompt_idiom = dataset[0]["pairs"][0]["prompt_idiom"]
        
    idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
    if model_name in ("gpt2", "gpt2med"):
        and_token = model.tokenizer.encode(",")[0]
    elif model_name in ("llama3b", "mistral7b", "falcon7b", "qwen7b", "qwen3b"):
        and_token = model.tokenizer.encode(",")[1]
    else:
        raise ValueError(f"Invalid model: {model_name}")
    and_token_idx = (idiom_tokens == and_token).nonzero(as_tuple=True)[0].item()

    labels = [model.tokenizer.decode(token) for token in idiom_tokens[and_token_idx-8:and_token_idx+5]]
    # Calculate the absolute maximum to make the scale symmetric
    y_labels = list(range(1, N_LAYERS + 1))
    plt.figure(figsize=(12, 8))
    sns.heatmap(
            avg_results,
            xticklabels=labels,
            yticklabels=y_labels,
            cmap="RdBu",
            cbar_kws={'label': 'Norm. Logit Diff'},
            center=0
        )
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    #plt.title(f"Dataset-Averaged MLP Activation Patching on {model_name}")
    plt.savefig(f"figures/averaged_mlp_patching_results_{model_name}_multiple.png")
    plt.savefig(f"figures/averaged_mlp_patching_results_{model_name}_multiple.eps")
    plt.gca() # Often helpful to have Layer 0 at the bottom
    plt.show()
    return plt.gcf()





# def top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset, model_name)
     
#     flat_results = averaged_patching_results.flatten()
#     top_values, top_indices = torch.topk(flat_results, k)

#     # Convert flat indices back to (layer, head) tuples
#     top_heads = []
#     for idx in top_indices:
#         layer = idx.item() // N_HEADS
#         head = idx.item() % N_HEADS
#         top_heads.append((layer, head))

#     print(f"Top {k} heads to patch/zero: {top_heads}")

#     accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
#     accumulated_kl_divs = torch.zeros((N_LAYERS, N_HEADS))
#     total_combinations = 0
   
#     for idiom_entry in dataset:
#         idiom_id = idiom_entry["id"]
#         pairs = idiom_entry["pairs"]

#         for pair in pairs:
#             prompt_idiom = pair["prompt_idiom"]
#             prompt_literal = pair["prompt_literal"]
#             idiom_answers = pair["idiom_answers"]
#             literal_answers = pair["literal_answers"]

#             # tokens
#             idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#             literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#             min_token_len = min(len(idiom_tokens), len(literal_tokens))

#             # For each pair, iterate through the 3 answer combinations
#             for answer_idx in range(3):
#                 correct_answer = idiom_answers[answer_idx]
#                 incorrect_answer = literal_answers[answer_idx]
                
#                 correct_token = model.tokenizer.encode(correct_answer)[1]
#                 incorrect_token = model.tokenizer.encode(incorrect_answer)[1]

#                 correct_answer_idx = correct_token
#                 incorrect_answer_idx = incorrect_token
                
#                 total_combinations += 1


        

#                 # clean run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_idiom) as invoker:
#                         clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#                         z_hs = {}
#                         for layer_idx in range(N_LAYERS):
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                             for head_idx in range(N_HEADS):
#                                 z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
#                         clean_logits = model.lm_head.output
#                         clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"clean logit diff {clean_logit_diff}")
                
#                 # Extract values from Proxy objects after trace context exits
#                 z_hs_np = {}
#                 for (layer_idx, head_idx), proxy_obj in z_hs.items():
#                     z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
#                 # Now pickle the actual numpy arrays
#                 z_hs_file = open(f"z_hs_{model_name}.pkl", "wb")
#                 pickle.dump(z_hs_np, z_hs_file)
#                 z_hs_file.close()
                

#                 # corrupted run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_literal) as invoker:
#                         corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#                         corrupted_logits = model.lm_head.output
#                         corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
#                 #patching
                
#                 #load pickle
#                 z_hs_file = open(f"z_hs_{model_name}.pkl", "rb")
#                 z_hs_np = pickle.load(z_hs_file)
#                 z_hs_file.close()
                
#                 # Convert numpy arrays back to torch tensors
#                 z_hs = {}
#                 for (layer_idx, head_idx), np_array in z_hs_np.items():
#                     z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
#                 # Compute actual_seq_len outside trace context to avoid Proxy issues
#                 sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
#                 if len(sample_z_hs.shape) == 2:
#                     saved_seq_len = sample_z_hs.shape[0]
#                 else:
#                     saved_seq_len = sample_z_hs.shape[1]
#                 actual_seq_len = min(saved_seq_len, min_token_len)
                
#                 # Prepare z_hs tensors with proper shapes
#                 z_hs_prepared = {}
#                 for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
#                     if len(z_hs_tensor.shape) == 2:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
#                     else:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
#                 patched_logits_dict = {}
                
#                 with model.trace() as tracer:
#                     for layer_idx in range(N_LAYERS):
#                         for head_idx in range(N_HEADS):
#                             with tracer.invoke(prompt_literal) as invoker:
#                                 if model_name in ("gpt2", "gpt2med"):
#                                     z = model.transformer.h[layer_idx].attn.c_proj.input
#                                 elif model_name == "llama3b":
#                                     z = model.model.layers[layer_idx].self_attn.o_proj.input
#                                 else:
#                                     raise ValueError(f"Invalid model: {model_name}")
#                                 z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                                 z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]
#                                 if (layer_idx, head_idx) in top_heads:
#                                     z_corrupt[:, :actual_seq_len, head_idx, :] = 0


#                                 patched_logits = model.lm_head.output
#                                 patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
#                                 patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
#                                 patched_result_saved = patched_result.save()
#                                 patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
#                             accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
#     average_patching_results = accumulated_results / total_combinations
#     return average_patching_results


# def run_top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     average_patching_results = top_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name)
#     x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
#     prompt_idiom = f"top_{k}_attention_head_ablation_{model_name}_multiple"
#     fig = plot_ioi_patching_results_attention(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Top {k} {model_name} Attention Head Ablation across {len(dataset)} Idioms")
#     return fig


# def random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset, model_name)
     
#     flat_results = averaged_patching_results.flatten()
#     # top_values, top_indices = torch.topk(flat_results, k)
#     top_indices = torch.randperm(len(flat_results))[:k]

#     # Convert flat indices back to (layer, head) tuples
#     top_heads = []
#     for idx in top_indices:
#         layer = idx.item() // N_HEADS
#         head = idx.item() % N_HEADS
#         top_heads.append((layer, head))

#     print(f"Random {k} heads to patch/zero: {top_heads}")

#     accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
#     accumulated_kl_divs = torch.zeros((N_LAYERS, N_HEADS))

#     total_combinations = 0
#     for idiom_entry in dataset:
#         idiom_id = idiom_entry["id"]
#         pairs = idiom_entry["pairs"]
        
#         # For each idiom, iterate through pairs
#         for pair in pairs:
#             prompt_idiom = pair["prompt_idiom"]
#             prompt_literal = pair["prompt_literal"]
#             idiom_answers = pair["idiom_answers"]
#             literal_answers = pair["literal_answers"]

#             # tokens
#             idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#             literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#             min_token_len = min(len(idiom_tokens), len(literal_tokens))

#             # For each pair, iterate through the 3 answer combinations
#             for answer_idx in range(3):
#                 correct_answer = idiom_answers[answer_idx]
#                 incorrect_answer = literal_answers[answer_idx]
                
#                 correct_token = model.tokenizer.encode(correct_answer)[1]
#                 incorrect_token = model.tokenizer.encode(incorrect_answer)[1]

#                 correct_answer_idx = correct_token
#                 incorrect_answer_idx = incorrect_token
                
#                 total_combinations += 1


#                 # clean run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_idiom) as invoker:
#                         clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#                         z_hs = {}
#                         for layer_idx in range(N_LAYERS):
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                             for head_idx in range(N_HEADS):
#                                 z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
#                         clean_logits = model.lm_head.output
#                         clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"clean logit diff {clean_logit_diff}")
                
#                 # Extract values from Proxy objects after trace context exits
#                 z_hs_np = {}
#                 for (layer_idx, head_idx), proxy_obj in z_hs.items():
#                     z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
#                 # Now pickle the actual numpy arrays
#                 z_hs_file = open(f"z_hs_{model_name}.pkl", "wb")
#                 pickle.dump(z_hs_np, z_hs_file)
#                 z_hs_file.close()

#                 # corrupted run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_literal) as invoker:
#                         corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#                         corrupted_logits = model.lm_head.output
#                         corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
#                 #patching
                
#                 #load pickle
#                 z_hs_file = open(f"z_hs_{model_name}.pkl", "rb")
#                 z_hs_np = pickle.load(z_hs_file)
#                 z_hs_file.close()
                
#                 # Convert numpy arrays back to torch tensors
#                 z_hs = {}
#                 for (layer_idx, head_idx), np_array in z_hs_np.items():
#                     z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
#                 # Compute actual_seq_len outside trace context to avoid Proxy issues
#                 sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
#                 if len(sample_z_hs.shape) == 2:
#                     saved_seq_len = sample_z_hs.shape[0]
#                 else:
#                     saved_seq_len = sample_z_hs.shape[1]
#                 actual_seq_len = min(saved_seq_len, min_token_len)
                
#                 # Prepare z_hs tensors with proper shapes
#                 z_hs_prepared = {}
#                 for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
#                     if len(z_hs_tensor.shape) == 2:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
#                     else:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
#                 patched_logits_dict = {}
                
#                 with model.trace() as tracer:
#                     for layer_idx in range(N_LAYERS):
#                         for head_idx in range(N_HEADS):
#                             with tracer.invoke(prompt_literal) as invoker:
#                                 if model_name in ("gpt2", "gpt2med"):
#                                     z = model.transformer.h[layer_idx].attn.c_proj.input
#                                 elif model_name == "llama3b":
#                                     z = model.model.layers[layer_idx].self_attn.o_proj.input
#                                 else:
#                                     raise ValueError(f"Invalid model: {model_name}")
#                                 z_corrupt = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                                 # Only patch the overlapping sequence length
#                                 z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]
#                                 if (layer_idx, head_idx) in top_heads:
#                                     z_corrupt[:, :actual_seq_len, head_idx, :] = 0


#                                 patched_logits = model.lm_head.output
#                                 patched_logit_diff = (patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]).save()
#                                 patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
#                                 patched_result_saved = patched_result.save()
#                                 patched_logits_dict[(layer_idx, head_idx)] = patched_logits[0, :min_token_len, :].save()
#                             accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
                
#     average_patching_results = accumulated_results / total_combinations
#     return average_patching_results


# def run_random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     average_patching_results = random_k_attention_head_ablation(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name)
#     x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
#     prompt_idiom = f"random_{k}_attention_head_ablation_{model_name}_multiple"
#     fig = plot_ioi_patching_results_attention(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Random {k} {model_name} Attention Head Ablation across {len(dataset)} Idioms")
#     return fig


# ### need to only patch the top k heads and then print the logit difference


# def patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset)
     
#     flat_results = averaged_patching_results.flatten()
#     top_values, top_indices = torch.topk(flat_results, k)

#     # Convert flat indices back to (layer, head) tuples

#     # Convert flat indices back to (layer, head) tuples
#     top_heads = []
#     for idx in top_indices:
#         layer = idx.item() // N_HEADS
#         head = idx.item() % N_HEADS
#         top_heads.append((layer, head))

#     print(f"Top {k} heads to patch/zero: {top_heads}")

#     accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
#     # total_kl = torch.zeros((N_LAYERS, N_HEADS))
#     # kl_counts = torch.zeros((N_LAYERS, N_HEADS))
    
#     total_combinations = 0
    
#     # Iterate through idioms
#     for idiom_entry in dataset:
#         idiom_id = idiom_entry["id"]
#         pairs = idiom_entry["pairs"]
        
#         # For each idiom, iterate through pairs
#         for pair in pairs:
#             prompt_idiom = pair["prompt_idiom"]
#             prompt_literal = pair["prompt_literal"]
#             idiom_answers = pair["idiom_answers"]
#             literal_answers = pair["literal_answers"]

#             # tokens
#             idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#             literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#             min_token_len = min(len(idiom_tokens), len(literal_tokens))

#             # For each pair, iterate through the 3 answer combinations
#             for answer_idx in range(3):
#                 correct_answer = idiom_answers[answer_idx]
#                 incorrect_answer = literal_answers[answer_idx]
                
#                 correct_token = model.tokenizer.encode(correct_answer)[0]
#                 incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

#                 correct_answer_idx = correct_token
#                 incorrect_answer_idx = incorrect_token
                
#                 total_combinations += 1

#                 # clean run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_idiom) as invoker:
#                         clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#                         z_hs = {}
#                         for layer_idx in range(N_LAYERS):
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                             for head_idx in range(N_HEADS):
#                                 z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
#                         clean_logits = model.lm_head.output.save()
#                         clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"clean logit diff {clean_logit_diff}")
#                 clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
#                 np.save(f"data/clean_logits_value.npy", clean_logits_value)
#                 # Extract values from Proxy objects after trace context exits
#                 z_hs_np = {}
#                 for (layer_idx, head_idx), proxy_obj in z_hs.items():
#                     z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
#                 # Now pickle the actual numpy arrays
#                 z_hs_file = open(f"z_hs_top_k_{model_name}.pkl", "wb")
#                 pickle.dump(z_hs_np, z_hs_file)
#                 z_hs_file.close()

#                 # corrupted run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_literal) as invoker:
#                         corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#                         corrupted_logits = model.lm_head.output
#                         corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
#                 #patching
#                 #load pickle
#                 z_hs_file = open(f"z_hs_top_k_{model_name}.pkl", "rb")
#                 z_hs_np = pickle.load(z_hs_file)
#                 z_hs_file.close()
                
#                 # Convert numpy arrays back to torch tensors
#                 z_hs = {}
#                 for (layer_idx, head_idx), np_array in z_hs_np.items():
#                     z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
#                 # Compute actual_seq_len outside trace context to avoid Proxy issues
#                 sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
#                 if len(sample_z_hs.shape) == 2:
#                     saved_seq_len = sample_z_hs.shape[0]
#                 else:
#                     saved_seq_len = sample_z_hs.shape[1]
#                 actual_seq_len = min(saved_seq_len, min_token_len)
                
#                 # Prepare z_hs tensors with proper shapes
#                 z_hs_prepared = {}
#                 for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
#                     if len(z_hs_tensor.shape) == 2:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
#                     else:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
#                 patched_logits_dict = {}
#                 # Patch each selected head independently (one forward pass per head).
#                 with model.trace() as tracer:
#                     for (layer_idx, head_idx) in top_heads:
#                         with tracer.invoke(prompt_literal) as invoker:
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_corrupt = einops.rearrange(
#                                 z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS
#                             )
#                             z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

#                             patched_logits = model.lm_head.output.save()
#                             patched_logit_diff = (
#                                 patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
#                             ).save()
#                             patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
#                             print(f"patched result ({layer_idx}, {head_idx}) {patched_result}")
#                             patched_result_saved = patched_result.save()

#                             accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
#     average_patching_results = accumulated_results / total_combinations
#     return average_patching_results


# def run_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     average_patching_results = patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name)
#     print(f"Average patching results: {average_patching_results}")
#     x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
#     prompt_idiom = f"top_{k}_heads_patching_only_{model_name}"
#     fig = plot_ioi_patching_results_attention(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Only Patching Top {k} {model_name} Attention Head across {len(dataset)} Idioms")
#     return fig

# def random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     averaged_patching_results = average_attention_head_patching(N_LAYERS, N_HEADS, D_HEADS, dataset, model_name)
     
#     flat_results = averaged_patching_results.flatten()
#     # top_values, top_indices = torch.topk(flat_results, k)
#     top_indices = torch.randperm(len(flat_results))[:k]

#     # Convert flat indices back to (layer, head) tuples
#     top_heads = []
#     for idx in top_indices:
#         layer = idx.item() // N_HEADS
#         head = idx.item() % N_HEADS
#         top_heads.append((layer, head))

#     print(f"Random {k} heads to patch/zero: {top_heads}")

#     accumulated_results = torch.zeros((N_LAYERS, N_HEADS))
#     # total_kl = torch.zeros((N_LAYERS, N_HEADS))
#     # kl_counts = torch.zeros((N_LAYERS, N_HEADS))


#     total_combinations = 0
    
#     # Iterate through idioms
#     for idiom_entry in dataset:
#         idiom_id = idiom_entry["id"]
#         pairs = idiom_entry["pairs"]
        
#         # For each idiom, iterate through pairs
#         for pair in pairs:
#             prompt_idiom = pair["prompt_idiom"]
#             prompt_literal = pair["prompt_literal"]
#             idiom_answers = pair["idiom_answers"]
#             literal_answers = pair["literal_answers"]

#             # tokens
#             idiom_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#             literal_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#             min_token_len = min(len(idiom_tokens), len(literal_tokens))

#             # For each pair, iterate through the 3 answer combinations
#             for answer_idx in range(3):
#                 correct_answer = idiom_answers[answer_idx]
#                 incorrect_answer = literal_answers[answer_idx]
                
#                 correct_token = model.tokenizer.encode(correct_answer)[0]
#                 incorrect_token = model.tokenizer.encode(incorrect_answer)[0]

#                 correct_answer_idx = correct_token
#                 incorrect_answer_idx = incorrect_token
                
#                 total_combinations += 1

#                 # clean run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_idiom) as invoker:
#                         clean_tokens = model.tokenizer(prompt_idiom, return_tensors="pt")["input_ids"][0]
#                         z_hs = {}
#                         for layer_idx in range(N_LAYERS):
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS)
#                             for head_idx in range(N_HEADS):
#                                 z_hs[layer_idx, head_idx] = z_reshaped[:, :min_token_len, head_idx, :].save()
                        
#                         clean_logits = model.lm_head.output.save()
#                         clean_logit_diff = (clean_logits[0, -1, correct_answer_idx] - clean_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"clean logit diff {clean_logit_diff}")
#                 clean_logits_value = clean_logits[0, :min_token_len, :].detach().cpu().numpy()
#                 np.save(f"data/clean_logits_value.npy", clean_logits_value)
#                 # Extract values from Proxy objects after trace context exits
#                 z_hs_np = {}
#                 for (layer_idx, head_idx), proxy_obj in z_hs.items():
#                     z_hs_np[layer_idx, head_idx] = proxy_obj.value.detach().cpu().numpy()
                
#                 # Now pickle the actual numpy arrays
#                 z_hs_file = open(f"z_hs_top_k_{model_name}.pkl", "wb")
#                 pickle.dump(z_hs_np, z_hs_file)
#                 z_hs_file.close()

#                 # corrupted run
#                 with model.trace() as tracer:
#                     with tracer.invoke(prompt_literal) as invoker:
#                         corrupted_tokens = model.tokenizer(prompt_literal, return_tensors="pt")["input_ids"][0]
#                         corrupted_logits = model.lm_head.output
#                         corrupted_logit_diff = (corrupted_logits[0, -1, correct_answer_idx] - corrupted_logits[0, -1, incorrect_answer_idx]).save()
#                         print(f"corrupt logit diff {corrupted_logit_diff}")
                
                
#                 #patching
#                 #load pickle
#                 z_hs_file = open(f"z_hs_top_k_{model_name}.pkl", "rb")
#                 z_hs_np = pickle.load(z_hs_file)
#                 z_hs_file.close()
                
#                 # Convert numpy arrays back to torch tensors
#                 z_hs = {}
#                 for (layer_idx, head_idx), np_array in z_hs_np.items():
#                     z_hs[layer_idx, head_idx] = torch.from_numpy(np_array)
                
#                 # Compute actual_seq_len outside trace context to avoid Proxy issues
#                 sample_z_hs = z_hs[0, 0]  # Use first layer, first head as sample
#                 if len(sample_z_hs.shape) == 2:
#                     saved_seq_len = sample_z_hs.shape[0]
#                 else:
#                     saved_seq_len = sample_z_hs.shape[1]
#                 actual_seq_len = min(saved_seq_len, min_token_len)
                
#                 # Prepare z_hs tensors with proper shapes
#                 z_hs_prepared = {}
#                 for (layer_idx, head_idx), z_hs_tensor in z_hs.items():
#                     if len(z_hs_tensor.shape) == 2:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor.unsqueeze(0)[:, :actual_seq_len, :]
#                     else:
#                         z_hs_prepared[layer_idx, head_idx] = z_hs_tensor[:, :actual_seq_len, :]
                
#                 patched_logits_dict = {}
#                 # Patch each selected head independently (one forward pass per head).
#                 with model.trace() as tracer:
#                     for (layer_idx, head_idx) in top_heads:
#                         with tracer.invoke(prompt_literal) as invoker:
#                             if model_name in ("gpt2", "gpt2med"):
#                                 z = model.transformer.h[layer_idx].attn.c_proj.input
#                             elif model_name == "llama3b":
#                                 z = model.model.layers[layer_idx].self_attn.o_proj.input
#                             else:
#                                 raise ValueError(f"Invalid model: {model_name}")
#                             z_corrupt = einops.rearrange(
#                                 z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS, dh=D_HEADS
#                             )
#                             z_corrupt[:, :actual_seq_len, head_idx, :] = z_hs_prepared[layer_idx, head_idx]

#                             patched_logits = model.lm_head.output.save()
#                             patched_logit_diff = (
#                                 patched_logits[0, -1, correct_answer_idx] - patched_logits[0, -1, incorrect_answer_idx]
#                             ).save()
#                             patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
#                             print(f"patched result ({layer_idx}, {head_idx}) {patched_result}")
#                             patched_result_saved = patched_result.save()

#                             accumulated_results[layer_idx, head_idx] += patched_result_saved.item()
#     average_patching_results = accumulated_results / total_combinations
#     return average_patching_results


# def run_random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name):
#     average_patching_results = random_patch_top_heads(dataset, N_LAYERS, N_HEADS, D_HEADS, k, model_name)
#     print(f"Average patching results: {average_patching_results}")
#     x_labels = [f"Head {i+1}" for i in range(N_HEADS)]
#     prompt_idiom = f"random_{k}_heads_patching_only_{model_name}"
#     fig = plot_ioi_patching_results_attention(model, model_name, average_patching_results.tolist(), x_labels, prompt_idiom, f"Only Patching Random {k} {model_name} Attention Head across {len(dataset)} Idioms")
#     return fig









### MAIN





if __name__ == "__main__":
    args = parser.parse_args()
    dataset = json.load(open(f"data/{args.dataset}"))
    if args.device == "cpu":
        device_map = "cpu"
    elif args.device == "gpu":
        device_map = "cuda"
    else:
        device_map = "auto"
    cache_dtype = _dtype_from_name(args.cache_dtype)
    # print device map
    print(f"Device map: {device_map}")
   
    for idiom_entry in dataset:
        print("--------------------------------")
        print(f"Idiom: {idiom_entry['id']}")
        for pair in idiom_entry["pairs"]:
            print(f"  Idiom prompt: {pair['prompt_idiom']}")
            print(f"  Literal prompt: {pair['prompt_literal']}")
        print("--------------------------------")


    if args.model == "gpt2":
        model_name = "gpt2"
        model = LanguageModel("openai-community/gpt2", device_map=device_map)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.transformer.h)
        N_HEADS = 12
        D_MODEL = 768
        D_HEADS = D_MODEL // N_HEADS

        
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        elif args.intervention =="mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    
    elif args.model == "gpt2med":
        model_name = "gpt2med"
        model = LanguageModel("openai-community/gpt2-medium", device_map=device_map)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.transformer.h)
        N_HEADS = 16
        D_MODEL = 1024
        D_HEADS = D_MODEL // N_HEADS

        
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        elif args.intervention =="mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")

    elif args.model == "llama3b":
        model_name = "llama3b"
        # token

        access_token = os.environ.get('HF_TOKEN_LLAMA')
        if access_token is None:
            raise ValueError("HF_TOKEN_LLAMA is not set")
        model = LanguageModel("meta-llama/Llama-3.2-3B", device_map=device_map, token = access_token)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 24
        D_MODEL = 3072
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    elif args.model == "mistral7b":
        model_name = "mistral7b"
        # token

        access_token = os.environ.get('HF_TOKEN_LLAMA')
        if access_token is None:
            raise ValueError("HF_TOKEN_LLAMA is not set")
        model = LanguageModel("mistralai/Mistral-7B-v0.1", device_map=device_map, token = access_token)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 32
        D_MODEL = 4096
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    elif args.model == "falcon7b":
        model_name = "falcon7b"
        # token

        access_token = os.environ.get('HF_TOKEN_LLAMA')
        if access_token is None:
            raise ValueError("HF_TOKEN_LLAMA is not set")
        model = LanguageModel("tiiuae/Falcon3-7B-Base", device_map=device_map, token = access_token)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 24
        D_MODEL = 3072
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    elif args.model == "qwen7b":
        model_name = "qwen7b"
        # token

        access_token = os.environ.get('HF_TOKEN_LLAMA')
        if access_token is None:
            raise ValueError("HF_TOKEN_LLAMA is not set")
        model = LanguageModel("Qwen/Qwen2.5-7B", device_map=device_map, token = access_token)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 28
        D_MODEL = 3584
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    elif args.model == "qwen3b":
        model_name = "qwen3b"
        # token

        access_token = os.environ.get('HF_TOKEN_LLAMA')
        if access_token is None:
            raise ValueError("HF_TOKEN_LLAMA is not set")
        model = LanguageModel("Qwen/Qwen2.5-3B", device_map=device_map, token = access_token)
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        elif hasattr(model, "transformer"):
            model.transformer.eval()
        else:
            raise AttributeError("Could not locate the underlying PyTorch model.")
        N_LAYERS = len(model.model.layers)
        N_HEADS = 16
        D_MODEL = 2048
        D_HEADS = D_MODEL // N_HEADS
        if args.intervention == "residual_stream":
            if args.averaging:
                print("Running average residual stream patching")
                run_average_residual_stream_patching(
                    dataset, N_LAYERS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running residual stream patching")
                run_residual_stream_patching(dataset, N_LAYERS, model_name)
        elif args.intervention == "mlp":
            if args.averaging:
                print("Running average MLP patching")
                run_average_mlp_patching(
                    dataset,
                    N_LAYERS,
                    model_name,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
        elif args.intervention == "attention_head":
            if args.averaging:
                print("Running average attention head patching")
                run_average_attention_head_patching(
                    dataset, N_LAYERS, N_HEADS, D_HEADS, model_name,
                    cache_dtype=cache_dtype,
                    max_idioms=args.max_idioms,
                    max_pairs_per_idiom=args.max_pairs_per_idiom,
                    max_answers=args.max_answers,
                    start_idiom=args.start_idiom,
                    accumulator_path=args.accumulator_path,
                )
            else:
                print("Running attention head patching")
                run_attention_head_patching(dataset, N_LAYERS, N_HEADS, D_HEADS, model_name)
        else:
            raise ValueError(f"Invalid intervention: {args.intervention}")
    else:
        raise ValueError(f"Invalid model: {args.model}")
