## IMPORTS

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nnsight import LanguageModel


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama3b": {
        "model_id": "meta-llama/Llama-3.2-3B",
        "n_layers": 28,
        "n_heads": 24,
        "d_head": 128,
    },
    "llama8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B",
        "n_layers": 32,
        "n_heads": 32,
        "d_head": 128,
    },
}

N_LAYERS: int
N_HEADS: int
D_HEAD: int

REQUIRED_COLUMNS = {
    "ABBR", "Sentence_A", "Sentence_C",
    "Primary_Expression", "Alternative_Expression",
}


## PARSER ARGUMENTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("adam_dataset_patching_filled2.csv"))
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), default="llama3b")
    parser.add_argument("--intervention", choices=("attention_head", "mlp"), required=True)
    parser.add_argument("--accumulator", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--patch-batch-size", type=int, default=24, help="patch conditions evaluated in one model batch")
    parser.add_argument("--answer-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-total-effect", type=float, default=1e-6, help="minimum |clean_A_ld − clean_C_ld| for normalised restoration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="plot from the accumulator without loading a model")
    return parser.parse_args()


## DATASET

def read_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    abbreviations = list(dict.fromkeys(r["ABBR"].strip() for r in rows))
    if not rows or any(not a for a in abbreviations):
        raise ValueError("Dataset is empty or contains a blank ABBR")
    return rows, abbreviations


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_context(sentence: str, abbreviation: str) -> str:
    return (f"{sentence}\n\n"
        f"Question: What does {abbreviation} stand for?\n"
        "Answer:")


## TOKEN FINDER

def find_spans(text: str, value: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(re.escape(value), text, re.IGNORECASE)]


def token_index_for_span(tokenizer: Any, text: str, span: tuple[int, int],  *, use_last_token: bool) -> int:
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    overlapping = [idx for idx, (s, e) in enumerate(encoded["offset_mapping"]) if s < span[1] and e > span[0]]
    if not overlapping:
        raise ValueError(f"No token overlaps character span {span} in: {text!r}")
    return overlapping[-1] if use_last_token else overlapping[0]


def locate_ac_tokens(tokenizer: Any, row: dict[str, str]) -> tuple[str, str, int, int]:
    """
    source_a_index : last token of primary_expression  in context_a
    source_c_index : last token of alternative_expression in context_c

    For A→C patching : clean activation comes from source_a_index in A,
                       patch target is source_c_index in C.
    For C→A patching : reversed.
    """
    abbreviation = row["ABBR"].strip()
    primary      = row["Primary_Expression"].strip()
    alternative  = row["Alternative_Expression"].strip()

    spans_a = find_spans(row["Sentence_A"], primary)
    if not spans_a:
        raise ValueError(f"Primary expression {primary!r} not found in Sentence_A")

    spans_c = find_spans(row["Sentence_C"], alternative)
    if not spans_c:
        raise ValueError(f"Alternative expression {alternative!r} not found in Sentence_C")

    context_a = build_context(row["Sentence_A"], abbreviation)
    context_c = build_context(row["Sentence_C"], abbreviation)

    source_a_index = token_index_for_span(
        tokenizer, context_a, spans_a[0], use_last_token=True
    )
    source_c_index = token_index_for_span(
        tokenizer, context_c, spans_c[0], use_last_token=True
    )
    return context_a, context_c, source_a_index, source_c_index

## ANSWER

def answer_encoding(tokenizer: Any, context: str, answer: str) -> tuple[str, list[int], list[int]]:
    full_text   = context + " " + answer.strip()
    context_ids = tokenizer(context,   add_special_tokens=True)["input_ids"]
    full_ids    = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("Answer boundary changed context tokenisation")
    answer_ids = full_ids[len(context_ids):]
    if not answer_ids:
        raise ValueError(f"Answer {answer!r} produced no tokens")
    positions = list(range(len(context_ids) - 1, len(full_ids) - 1))
    return full_text, answer_ids, positions

## UTILS 

def to_numpy(value: Any) -> np.ndarray:
    value = value.value if hasattr(value, "value") else value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray):
        return value
    return value.detach().float().cpu().numpy()


def attention_module(model: LanguageModel, layer: int) -> Any:
    return model.model.layers[layer].self_attn.o_proj


def mlp_module(model: LanguageModel, layer: int) -> Any:
    return model.model.layers[layer].mlp.down_proj


def all_conditions(intervention: str) -> list[tuple[int, int | None]]:
    if intervention == "attention_head":
        return [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    return [(l, None) for l in range(N_LAYERS)]


## CLEAN ACTIVATIONS

def capture_clean_activations(model: LanguageModel, intervention: str, context: str, source_index: int, cache_dtype: torch.dtype) -> list[torch.Tensor]:
    saved: list[Any] = []
    with model.trace(context):
        for layer in range(N_LAYERS):
            if intervention == "attention_head":
                act = attention_module(model, layer).input
                val = act[0, source_index, :].reshape(N_HEADS, D_HEAD)
            else:
                act = mlp_module(model, layer).input
                val = act[0, source_index, :]
            saved.append(val.save())
    return [torch.from_numpy(to_numpy(v)).to(cache_dtype) for v in saved]


def capture_clean_logits(model: LanguageModel, context: str, last_ctx_pos: int) -> torch.Tensor:
    """Return float32 logit vector at *last_ctx_pos* from a clean forward pass."""
    with model.trace(context):
        saved = model.lm_head.output[0, last_ctx_pos, :].save()
    return torch.from_numpy(to_numpy(saved)).float()


def score_answer(model: LanguageModel, context: str, answer: str, reduction: str) -> float:
    full_text, answer_ids, positions = answer_encoding(model.tokenizer, context, answer)
    targets = torch.tensor(answer_ids, dtype=torch.long)
    with model.trace(full_text):
        logits   = model.lm_head.output[0, positions, :]
        targets  = targets.to(logits.device)
        tok_lp   = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        score = (tok_lp.mean() if reduction == "mean" else tok_lp.sum()).save()
    return float(to_numpy(score).item())

## PATCHED RUN

def _apply_patch(intervention: str, model: LanguageModel, layer: int, head: int | None, batch_idx: int, target_index: int, clean_activations: list[torch.Tensor]) -> None:
    if intervention == "attention_head":
        start = int(head) * D_HEAD
        end   = start + D_HEAD
        act   = attention_module(model, layer).input
        act[batch_idx, target_index, start:end] = (clean_activations[layer][int(head)].to(device=act.device, dtype=act.dtype))
    else:
        act = mlp_module(model, layer).input
        act[batch_idx, target_index, :] = (clean_activations[layer].to(device=act.device, dtype=act.dtype))


def patched_full_metrics_batch(model: LanguageModel, intervention: str, context_target: str, primary: str, alternative: str, target_index: int, conditions: list[tuple[int, int | None]], clean_activations: list[torch.Tensor], clean_log_probs: np.ndarray, clean_probs: np.ndarray, reduction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    answer_logprob_diff : [batch]  primary_logprob − alternative_logprob after patching
    primary_logprob     : [batch]  log P(primary   | patched context)
    alt_logprob         : [batch]  log P(alternative | patched context)
    kl_div              : [batch]  KL(clean‖patched) at last context token
    """
    batch_size = len(conditions)

    # Primary answer
    full_primary, primary_ids, primary_positions = answer_encoding(model.tokenizer, context_target, primary)
    last_ctx_pos = primary_positions[0]   # == len(context_ids) - 1
    primary_targets = torch.tensor(primary_ids, dtype=torch.long)

    with model.trace([full_primary] * batch_size):
        for bi, (layer, head) in enumerate(conditions):
            _apply_patch(intervention, model, layer, head, bi, target_index, clean_activations)
        all_logits = model.lm_head.output                         # [B, seq, vocab]
        ans_logits = all_logits[:, primary_positions, :]          # [B, n_tok, vocab]
        kl_logits  = all_logits[:, last_ctx_pos,      :]          # [B, vocab]

        pt = primary_targets.to(ans_logits.device)
        exp_pt = pt.unsqueeze(0).expand(batch_size, -1)
        tok_lp = torch.log_softmax(ans_logits, dim=-1).gather(-1, exp_pt.unsqueeze(-1)).squeeze(-1)
        primary_logprobs_s = (tok_lp.mean(dim=-1) if reduction == "mean" else tok_lp.sum(dim=-1)).save()
        kl_logits_s = kl_logits.save()

    # Alternative answer
    full_alt, alt_ids, alt_positions = answer_encoding(model.tokenizer, context_target, alternative)
    alt_targets = torch.tensor(alt_ids, dtype=torch.long)

    with model.trace([full_alt] * batch_size):
        for bi, (layer, head) in enumerate(conditions):
            _apply_patch(intervention, model, layer, head, bi, target_index, clean_activations)
        all_logits_alt = model.lm_head.output
        ans_logits_alt = all_logits_alt[:, alt_positions, :]

        at = alt_targets.to(ans_logits_alt.device)
        exp_at = at.unsqueeze(0).expand(batch_size, -1)
        tok_lp_alt = torch.log_softmax(ans_logits_alt, dim=-1).gather(-1, exp_at.unsqueeze(-1)).squeeze(-1)
        alt_logprobs_s = (tok_lp_alt.mean(dim=-1) if reduction == "mean" else tok_lp_alt.sum(dim=-1)).save()

    
    primary_lp          = to_numpy(primary_logprobs_s).astype(np.float64)
    alt_lp              = to_numpy(alt_logprobs_s).astype(np.float64)
    answer_logprob_diff = primary_lp - alt_lp

    # KL(clean ‖ patched) 
    kl_raw = to_numpy(kl_logits_s).astype(np.float64)           # [B, vocab]
    kl_max   = kl_raw.max(axis=-1, keepdims=True)
    kl_lp    = kl_raw - kl_max - np.log(np.sum(np.exp(kl_raw - kl_max), axis=-1, keepdims=True))
    kl_div = np.sum(clean_probs[np.newaxis, :] * (clean_log_probs[np.newaxis, :] - kl_lp), axis=-1)                      # [B]

    return answer_logprob_diff, primary_lp, alt_lp, kl_div


def evaluate_patches_direction(model: LanguageModel, intervention: str, context_source: str, context_target: str, primary: str, alternative: str, source_index: int, target_index: int, clean_logits_target: torch.Tensor, cache_dtype: torch.dtype, reduction: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays of shape (N_LAYERS, N_HEADS) [attention] or (N_LAYERS,) [mlp]
    answer_logprob_diff, primary_logprob, alt_logprob, kl_div
    """
    clean_activations = capture_clean_activations(model, intervention, context_source, source_index, cache_dtype)

    # Prepare clean distribution for KL reference
    cl_lp_t  = torch.log_softmax(clean_logits_target.float(), dim=-1)
    cl_lp    = cl_lp_t.numpy().astype(np.float64)          # [vocab]
    cl_p     = np.exp(cl_lp)                                 # [vocab]

    conditions = all_conditions(intervention)
    n          = len(conditions)
    ald_flat = np.empty(n, dtype=np.float64)
    pl_flat  = np.empty(n, dtype=np.float64)
    al_flat  = np.empty(n, dtype=np.float64)
    kl_flat  = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        batch = conditions[start: start + batch_size]
        ald, pl, al, kl = patched_full_metrics_batch(model, intervention, context_target, primary, alternative, target_index, batch, clean_activations, cl_lp, cl_p, reduction)
        end = start + len(batch)
        ald_flat[start:end] = ald
        pl_flat[start:end]  = pl
        al_flat[start:end]  = al
        kl_flat[start:end]  = kl

    shape = ((N_LAYERS, N_HEADS) if intervention == "attention_head" else (N_LAYERS,))
    return (ald_flat.reshape(shape), pl_flat.reshape(shape), al_flat.reshape(shape), kl_flat.reshape(shape))

## ACCUMULATOR

def _field_specs(row_count: int, abbreviation_count: int, result_shape: tuple[int, ...]) -> dict[str, tuple[tuple[int, ...], type, float]]:
    
    abbr = (abbreviation_count,) + result_shape
    row  = (row_count,)          + result_shape
    specs: dict[str, tuple[tuple[int, ...], type, float]] = {
        # Per row status
        "baseline_status": ((row_count,), np.uint8,   0),
        "status_ac":       ((row_count,), np.uint8,   0),
        "status_ca":       ((row_count,), np.uint8,   0),
        "status_aa":       ((row_count,), np.uint8,   0),
        "status_cc":       ((row_count,), np.uint8,   0),
        "bie_status":      ((row_count,), np.uint8,   0),
        # Per row storage
        "row_baselines": ((row_count, 4), np.float64, np.nan),
        "row_ac_ie":     (row,            np.float64, np.nan),
        "row_ca_ie":     (row,            np.float64, np.nan),
        # Baseline
        "baseline_sums":   ((abbreviation_count, 4), np.float64, 0),
        "baseline_counts": ((abbreviation_count,),   np.int64,   0),
        # Paired BIE: IE_AC − IE_CA
        "bie_sums":   (abbr, np.float64, 0),
        "bie_counts": (abbr, np.int64,   0),
    }
    # Per-direction aggregates for A→C, C→A, A→A (null ctrl), C→C (null ctrl)
    for pfx in ("ac", "ca", "aa", "cc"):
        for metric in ("answer_logprob_diff", "primary_lp", "alt_lp", "kl_div", "normalized", "ie" ):
            specs[f"{pfx}_{metric}_sums"] = (abbr, np.float64, 0)
        specs[f"{pfx}_counts"]            = (abbr, np.int64, 0)
        specs[f"{pfx}_normalized_counts"] = (abbr, np.int64, 0)
        specs[f"{pfx}_ie_counts"]         = (abbr, np.int64, 0)
    return specs


def new_accumulator(row_count: int, abbreviation_count: int, metadata: dict[str, Any], result_shape: tuple[int, ...]) -> dict[str, Any]:
    acc: dict[str, Any] = {"metadata": metadata, "skip_reasons": {}}
    for field, (shape, dtype, fill) in _field_specs(row_count, abbreviation_count, result_shape).items():
        acc[field] = np.full(shape, fill, dtype=dtype)
    return acc


def load_accumulator(path: Path, row_count: int, abbreviation_count: int, metadata: dict[str, Any], result_shape: tuple[int, ...]) -> dict[str, Any]:
    if not path.exists():
        return new_accumulator(row_count, abbreviation_count, metadata, result_shape)
    with np.load(path, allow_pickle=False) as data:
        saved_meta = json.loads(str(data["metadata"].item()))
        if saved_meta != metadata:
            raise ValueError(f"{path} was created with different settings")
        acc: dict[str, Any] = {"metadata":     saved_meta, "skip_reasons": json.loads(str(data["skip_reasons"].item()))}
        for field, (shape, dtype, fill) in _field_specs(row_count, abbreviation_count, result_shape).items():
            acc[field] = (data[field].copy() if field in data else np.full(shape, fill, dtype=dtype))
    return acc


def save_accumulator(path: Path, acc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arrays = {k: v for k, v in acc.items() if isinstance(v, np.ndarray)}
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, metadata=np.array(json.dumps(acc["metadata"], sort_keys=True)), skip_reasons=np.array(json.dumps(acc["skip_reasons"])), **arrays)
    os.replace(tmp, path)


def safe_average(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts != 0)


def nanmean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float("nan") if finite.size == 0 else float(np.mean(finite))


def nanmedian_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float("nan") if finite.size == 0 else float(np.median(finite))


def directional_probability(values: np.ndarray, positive: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite > 0 if positive else finite < 0))


def bootstrap_mean_interval(values: np.ndarray, n_resamples: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(values.shape[0], -1)
    complete = np.all(np.isfinite(flat), axis=1)
    usable = flat[complete]
    out_shape = values.shape[1:]
    if usable.shape[0] == 0:
        e = np.full(out_shape, np.nan, dtype=np.float64)
        return e.copy(), e.copy()
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(usable.shape[0], np.full(usable.shape[0], 1.0 / usable.shape[0]), size=n_resamples)
    means = weights @ usable / usable.shape[0]
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return lo.reshape(out_shape), hi.reshape(out_shape)

## SAVE SUMMARIES

def write_summaries(prefix: Path, intervention: str, abbreviations: list[str], acc: dict[str, Any], include_bootstrap: bool = False) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)

    bl = safe_average(acc["baseline_sums"], acc["baseline_counts"][:, None])
    # bl columns: [a_primary_lp, a_alt_lp, c_primary_lp, c_alt_lp]
    bl_a_ald = bl[:, 0] - bl[:, 1]   # clean answer_logprob_diff in A context
    bl_c_ald = bl[:, 2] - bl[:, 3]   # clean answer_logprob_diff in C context

    def means(pfx: str) -> dict[str, np.ndarray]:
        d = {m: safe_average(acc[f"{pfx}_{m}_sums"], acc[f"{pfx}_counts"]) for m in ("answer_logprob_diff", "primary_lp", "alt_lp", "kl_div")}
        d["normalized"] = safe_average(acc[f"{pfx}_normalized_sums"], acc[f"{pfx}_normalized_counts"])
        d["ie"] = safe_average(acc[f"{pfx}_ie_sums"], acc[f"{pfx}_ie_counts"])
        return d

    ac = means("ac")
    ca = means("ca")
    aa = means("aa")
    cc = means("cc")

  
    bie = safe_average(acc["bie_sums"], acc["bie_counts"])

    valid = acc["baseline_counts"] > 0

    
    loc_cols = ["layer", "head"] if intervention == "attention_head" else ["layer"]
    by_abbr = prefix.with_name(prefix.name + "_by_abbreviation.csv")
    with by_abbr.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ABBR", *loc_cols,
            "ac_answer_logprob_diff", "ac_primary_lp", "ac_alt_lp",
            "ac_kl_div",              "ac_normalized",  "ac_ie",
            "ca_answer_logprob_diff", "ca_primary_lp", "ca_alt_lp",
            "ca_kl_div",              "ca_normalized",  "ca_ie",
            "aa_ie",                  "cc_ie",
            "bidirectional_ie",
            "ac_count", "ca_count", "bie_count",
        ])
        for ai, abbr in enumerate(abbreviations):
            for layer, head in all_conditions(intervention):
                idx = ((ai, layer, int(head)) if head is not None else (ai, layer))
                loc = [layer, head] if head is not None else [layer]
                w.writerow([
                    abbr, *loc,
                    ac["answer_logprob_diff"][idx], ac["primary_lp"][idx],
                    ac["alt_lp"][idx],              ac["kl_div"][idx],
                    ac["normalized"][idx],           ac["ie"][idx],
                    ca["answer_logprob_diff"][idx], ca["primary_lp"][idx],
                    ca["alt_lp"][idx],              ca["kl_div"][idx],
                    ca["normalized"][idx],           ca["ie"][idx],
                    aa["ie"][idx],                   cc["ie"][idx],
                    bie[idx],
                    acc["ac_counts"][idx],
                    acc["ca_counts"][idx],
                    acc["bie_counts"][idx],
                ])

    if include_bootstrap:
        bs_lo, bs_hi = bootstrap_mean_interval(bie[valid])
    else:
        s = bie.shape[1:]
        bs_lo = np.full(s, np.nan, dtype=np.float64)
        bs_hi = np.full(s, np.nan, dtype=np.float64)

    global_csv = prefix.with_name(prefix.name + "_global.csv")
    with global_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            *loc_cols,
            "ac_answer_logprob_diff_mean", "ac_primary_lp_mean",
            "ac_alt_lp_mean",              "ac_kl_div_mean",
            "ac_normalized_mean",          "ac_ie_mean",   "ac_ie_median",
            "ca_answer_logprob_diff_mean", "ca_primary_lp_mean",
            "ca_alt_lp_mean",              "ca_kl_div_mean",
            "ca_normalized_mean",          "ca_ie_mean",   "ca_ie_median",
            "aa_ie_mean",                  "aa_ie_median",
            "cc_ie_mean",                  "cc_ie_median",
            "ac_answer_logprob_diff_median", "ca_answer_logprob_diff_median",
            "bidirectional_ie_mean",         "bidirectional_ie_median",
            "p_ac_ie_gt_zero",
            "p_ca_ie_lt_zero",
            "p_bidirectional_ie_gt_zero",
            "bidirectional_ie_bootstrap_ci_2_5",
            "bidirectional_ie_bootstrap_ci_97_5",
            "bootstrap_resamples",
        ])
        for layer, head in all_conditions(intervention):
            cell = ((slice(None), layer, int(head)) if head is not None else (slice(None), layer))
            loc = [layer, head] if head is not None else [layer]
            bi  = (layer, int(head)) if head is not None else layer
            w.writerow([
                *loc,
                nanmean_or_nan(ac["answer_logprob_diff"][cell][valid]),
                nanmean_or_nan(ac["primary_lp"][cell][valid]),
                nanmean_or_nan(ac["alt_lp"][cell][valid]),
                nanmean_or_nan(ac["kl_div"][cell][valid]),
                nanmean_or_nan(ac["normalized"][cell][valid]),
                nanmean_or_nan(ac["ie"][cell][valid]),
                nanmedian_or_nan(ac["ie"][cell][valid]),
                nanmean_or_nan(ca["answer_logprob_diff"][cell][valid]),
                nanmean_or_nan(ca["primary_lp"][cell][valid]),
                nanmean_or_nan(ca["alt_lp"][cell][valid]),
                nanmean_or_nan(ca["kl_div"][cell][valid]),
                nanmean_or_nan(ca["normalized"][cell][valid]),
                nanmean_or_nan(ca["ie"][cell][valid]),
                nanmedian_or_nan(ca["ie"][cell][valid]),
                nanmean_or_nan(aa["ie"][cell][valid]),
                nanmedian_or_nan(aa["ie"][cell][valid]),
                nanmean_or_nan(cc["ie"][cell][valid]),
                nanmedian_or_nan(cc["ie"][cell][valid]),
                nanmedian_or_nan(ac["answer_logprob_diff"][cell][valid]),
                nanmedian_or_nan(ca["answer_logprob_diff"][cell][valid]),
                nanmean_or_nan(bie[cell][valid]),
                nanmedian_or_nan(bie[cell][valid]),
                directional_probability(ac["ie"][cell][valid], positive=True),
                directional_probability(ca["ie"][cell][valid], positive=False),
                directional_probability(bie[cell][valid], positive=True),
                bs_lo[bi],
                bs_hi[bi],
                5000 if include_bootstrap else 0,
            ])

    bl_csv = prefix.with_name(prefix.name + "_baselines.csv")
    with bl_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ABBR",
            "a_primary_lp", "a_alt_lp", "a_answer_logprob_diff",
            "c_primary_lp", "c_alt_lp", "c_answer_logprob_diff",
            "row_count",
        ])
        for i, abbr in enumerate(abbreviations):
            w.writerow([
                abbr,
                bl[i, 0], bl[i, 1], bl_a_ald[i],
                bl[i, 2], bl[i, 3], bl_c_ald[i],
                acc["baseline_counts"][i],
            ])

    def _status(key: str) -> dict[str, int]:
        a = acc[key]
        return {
            "processed": int(np.sum(a == 1)),
            "skipped":   int(np.sum(a == 2)),
            "pending":   int(np.sum(a == 0)),
        }

    status_json = prefix.with_name(prefix.name + "_status.json")
    status_json.write_text(
        json.dumps({
            "baseline":       _status("baseline_status"),
            "ac":             _status("status_ac"),
            "ca":             _status("status_ca"),
            "aa":             _status("status_aa"),
            "cc":             _status("status_cc"),
            "bie":            _status("bie_status"),
            "skip_reasons":   acc["skip_reasons"],
        }, indent=2),
        encoding="utf-8",
    )


## PLOTS

def macro_average_map(sums: np.ndarray, counts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    means  = safe_average(sums, counts)
    vals   = means[valid]
    finite = np.isfinite(vals)
    return np.divide(np.nansum(vals, axis=0), finite.sum(axis=0), out=np.full(vals.shape[1:], np.nan, dtype=np.float64), where=finite.sum(axis=0) != 0)


def macro_median_map(sums: np.ndarray, counts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    means = safe_average(sums, counts)
    with np.errstate(all="ignore"):
        return np.nanmedian(means[valid], axis=0)


def _heatmap(axis: Any, values: np.ndarray, title: str, metric_name: str) -> None:
    finite = values[np.isfinite(values)]
    if metric_name == "kl_div":
        cmap, center = "viridis", None
        vmin = 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        cmap, center = "RdBu_r", 0
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        limit = limit or 1.0
        vmin, vmax = -limit, limit

    sns.heatmap(values, ax=axis, cmap=cmap, center=center, vmin=vmin, vmax=vmax, xticklabels=np.arange(N_HEADS), yticklabels=np.arange(1, N_LAYERS + 1), cbar_kws={"label": metric_name.replace("_", " ").title()})
    axis.set_xlabel("Attention head")
    axis.set_ylabel("Layer")
    axis.set_title(title)
    axis.tick_params(axis="y", rotation=0)


def plot_attention_heatmaps(prefix: Path, acc: dict[str, Any]) -> list[Path]:
    valid = acc["baseline_counts"] > 0
    if not np.any(valid):
        print("No processed results for heatmaps.")
        return []

    metrics_to_plot: list[tuple[str, str, np.ndarray]] = []
    for pfx, direction in (("ac", "A→C"), ("ca", "C→A"), ("aa", "A→A (null ctrl)"), ("cc", "C→C (null ctrl)")):
        for metric in ("answer_logprob_diff", "primary_lp", "alt_lp", "kl_div", "normalized", "ie"):
            count_key = (f"{pfx}_normalized_counts" if metric == "normalized" else f"{pfx}_ie_counts" if metric == "ie" else f"{pfx}_counts")
            values = macro_average_map(acc[f"{pfx}_{metric}_sums"], acc[count_key], valid)
            label = f"{direction} {metric.replace('_', ' ')}"
            metrics_to_plot.append((f"{pfx}_{metric}", label, values))

    # Bidirectional IE from the per-row-paired accumulator
    bie_macro = macro_average_map(acc["bie_sums"], acc["bie_counts"], valid)
    metrics_to_plot.append(("bidirectional_ie", "Bidirectional IE (IE_AC − IE_CA)", bie_macro))

    output_paths: list[Path] = []
    for name, title, values in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(14, 10))
        _heatmap(ax, values, title, name)
        fig.tight_layout()
        out = prefix.with_name(prefix.name + f"_{name}_heatmap.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out)
        print(f"  Heatmap: {out.name}")

    return output_paths


def plot_mlp_results(prefix: Path, abbreviations: list[str], acc: dict[str, Any]) -> list[Path]:
    valid_mask = acc["baseline_counts"] > 0
    if not np.any(valid_mask):
        print("No MLP results to plot.")
        return []

    valid_abbr = np.asarray(abbreviations)[valid_mask]
    layers = np.arange(N_LAYERS)
    output_paths: list[Path] = []

    # Build (name, title, values[n_valid, n_layers]) tuples to plot
    plot_specs: list[tuple[str, str, np.ndarray]] = []
    for pfx, direction in (("ac", "A→C"), ("ca", "C→A"), ("aa", "A→A (null ctrl)"), ("cc", "C→C (null ctrl)")):
        for metric in ("answer_logprob_diff", "primary_lp", "alt_lp", "kl_div", "normalized", "ie"):
            count_key = (f"{pfx}_normalized_counts" if metric == "normalized" else f"{pfx}_ie_counts" if metric == "ie" else f"{pfx}_counts")
            vals = safe_average(acc[f"{pfx}_{metric}_sums"], acc[count_key])[valid_mask]
            title = f"{direction} {metric.replace('_', ' ')}"
            plot_specs.append((f"{pfx}_{metric}", title, vals))

    # Bidirectional IE from the per-row-paired accumulator
    bie_abbr = safe_average(acc["bie_sums"], acc["bie_counts"])[valid_mask]
    plot_specs.append(("bidirectional_ie", "Bidirectional IE (IE_AC − IE_CA)", bie_abbr))

    def _mlp_cmap(metric: str, vals: np.ndarray) -> tuple[str, Any, float, float]:
        finite = vals[np.isfinite(vals)]
        if metric == "kl_div":
            vmin = 0.0
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax += 1.0
            return "viridis", None, vmin, vmax
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        limit = limit or 1.0
        return "RdBu_r", 0, -limit, limit

    for name, title, values in plot_specs:
        global_mean   = np.array([nanmean_or_nan(values[:, l]) for l in layers])
        global_median = np.array([nanmedian_or_nan(values[:, l]) for l in layers])

        # Line plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(layers, global_mean,   marker="o", linewidth=2, markersize=4,
                label="Mean across abbreviations")
        ax.plot(layers, global_median, marker="s", linewidth=2, markersize=4,
                linestyle="--", label="Median across abbreviations")
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.legend(); ax.grid(alpha=0.25)
        ax.set_xlabel("Layer"); ax.set_ylabel(title); ax.set_title(title)
        ax.set_xticks(layers)
        fig.tight_layout()
        line_path = prefix.with_name(prefix.name + f"_{name}_line.png")
        fig.savefig(line_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(line_path)

        # Abbreviation×layer heatmap
        base_metric = name.split("_", 1)[1] if "_" in name else name
        cmap, center, vmin, vmax = _mlp_cmap(base_metric, values)
        fig_h = max(8.0, 0.22 * len(valid_abbr))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        sns.heatmap(values, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax, xticklabels=layers, yticklabels=valid_abbr, cbar_kws={"label": title})
        ax.set_xlabel("Layer"); ax.set_ylabel("Abbreviation")
        ax.set_title(title + " by abbreviation")
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        fig.tight_layout()
        hm_path = prefix.with_name(prefix.name + f"_{name}_abbreviation_layer_heatmap.png")
        fig.savefig(hm_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(hm_path)

    return output_paths


## VALIDATION

def validate_rows(rows: list[dict[str, str]]) -> tuple[int, dict[int, str]]:
    valid = 0
    failures: dict[int, str] = {}
    for i, row in enumerate(rows):
        try:
            primary     = row["Primary_Expression"].strip()
            alternative = row["Alternative_Expression"].strip()
            if not primary or not alternative:
                raise ValueError("missing primary or alternative expression")
            if not find_spans(row["Sentence_A"], primary):
                raise ValueError(f"primary {primary!r} not in Sentence_A")
            if not find_spans(row["Sentence_C"], alternative):
                raise ValueError(f"alternative {alternative!r} not in Sentence_C")
            valid += 1
        except ValueError as exc:
            failures[i] = str(exc)
    return valid, failures

## MODEL

def load_model(model_id: str, device_map: str, dtype: torch.dtype) -> LanguageModel:
    token = os.environ.get("HF_TOKEN_LLAMA")
    kwargs: dict[str, Any] = {"device_map": device_map, "torch_dtype": dtype}
    if token:
        kwargs["token"] = token
    model = LanguageModel(model_id, **kwargs)
    (model.eval if hasattr(model, "eval") else model.model.eval)()
    return model




def run(args: argparse.Namespace) -> None:
    global N_LAYERS, N_HEADS, D_HEAD

    if args.start_row < 0 or args.max_rows < 0:
        raise ValueError("--start-row and --max-rows must be non-negative")
    if args.patch_batch_size < 1:
        raise ValueError("--patch-batch-size must be positive")

    rows, abbreviations = read_dataset(args.dataset)
    valid_count, val_failures = validate_rows(rows)
    print(
        f"Validation: {valid_count}/{len(rows)} rows usable; "
        f"{len(val_failures)} failures"
    )
    for ri, reason in list(val_failures.items())[:10]:
        print(f"  row {ri}: {reason}")
    if args.dry_run:
        return

    cfg       = MODEL_CONFIGS[args.model]
    model_id  = str(cfg["model_id"])
    N_LAYERS  = int(cfg["n_layers"])
    N_HEADS   = int(cfg["n_heads"])
    D_HEAD    = int(cfg["d_head"])
    cache_dtype = getattr(torch, args.cache_dtype)

    result_shape = ((N_LAYERS, N_HEADS) if args.intervention == "attention_head" else (N_LAYERS,))
    output_prefix  = args.output_prefix or Path(f"outputs/ac_{args.intervention}_{args.model}")
    accumulator_path = args.accumulator or output_prefix.with_suffix(".npz")

    metadata = {
        "experiment":      "ac_patching",
        "dataset_sha256":  dataset_fingerprint(args.dataset),
        "model_id":        model_id,
        "intervention":    args.intervention,
        "answer_reduction": args.answer_reduction,
        "cache_dtype":     args.cache_dtype,
        "min_total_effect": args.min_total_effect,
        "n_layers":        N_LAYERS,
        "n_heads":         N_HEADS,
        "d_head":          D_HEAD,
        "abbreviations":   abbreviations,
    }

    if args.plot_only and not accumulator_path.exists():
        raise FileNotFoundError(f"--plot-only: {accumulator_path} not found")

    acc = load_accumulator(accumulator_path, len(rows), len(abbreviations), metadata, result_shape)

    # Reset transient failures so they are retried
    for status_key in ("baseline_status", "status_ac", "status_ca", "status_aa", "status_cc"):
        for ri in np.flatnonzero(acc[status_key] == 2):
            if int(ri) not in val_failures:
                acc[status_key][ri] = 0
                acc["skip_reasons"].pop(str(int(ri)), None)

    abbr_to_idx = {a: i for i, a in enumerate(abbreviations)}

    if args.plot_only:
        write_summaries(output_prefix, args.intervention, abbreviations, acc, include_bootstrap=True)
        plots = (plot_attention_heatmaps(output_prefix, acc) if args.intervention == "attention_head" else plot_mlp_results(output_prefix, abbreviations, acc))
        for p in plots:
            print(f"Plot: {p}")
        return

    model = load_model(model_id, args.device_map, cache_dtype)

    stop = (len(rows) if args.max_rows == 0  else min(len(rows), args.start_row + args.max_rows))
    done_bl = done_ac = done_ca = done_aa = done_cc = done_bie = 0

    for ri in range(args.start_row, stop):
        needs_bl  = acc["baseline_status"][ri] == 0
        needs_ac  = acc["status_ac"][ri]        == 0
        needs_ca  = acc["status_ca"][ri]        == 0
        needs_aa  = acc["status_aa"][ri]        == 0
        needs_cc  = acc["status_cc"][ri]        == 0
        # BIE is accumulated once both directions are done (per exact row pairing)
        needs_bie = (
            acc["bie_status"][ri] == 0
            and acc["status_ac"][ri] == 1
            and acc["status_ca"][ri] == 1
        )

        if not (needs_bl or needs_ac or needs_ca or needs_aa or needs_cc or needs_bie):
            continue

        row          = rows[ri]
        abbreviation = row["ABBR"].strip()
        ai           = abbr_to_idx[abbreviation]

        
        if needs_bie and not (needs_bl or needs_ac or needs_ca or needs_aa or needs_cc):
            stored_ac = acc["row_ac_ie"][ri]
            stored_ca = acc["row_ca_ie"][ri]
            if not np.any(np.isnan(stored_ac)) and not np.any(np.isnan(stored_ca)):
                acc["bie_sums"][ai]   += stored_ac - stored_ca
                acc["bie_counts"][ai] += 1
                acc["bie_status"][ri]  = 1
                done_bie += 1
            else:
                
                acc["bie_status"][ri] = 2
            save_accumulator(accumulator_path, acc)
            write_summaries(output_prefix, args.intervention, abbreviations, acc)
            continue

        print(f"[{ri + 1}/{len(rows)}] {abbreviation}")

        try:
            if ri in val_failures:
                raise ValueError(val_failures[ri])

            context_a, context_c, src_a_idx, src_c_idx = locate_ac_tokens(
                model.tokenizer, row
            )
            primary     = row["Primary_Expression"].strip()
            alternative = row["Alternative_Expression"].strip()

            if needs_bl:
                a_pl = score_answer(model, context_a, primary,     args.answer_reduction)
                a_al = score_answer(model, context_a, alternative, args.answer_reduction)
                c_pl = score_answer(model, context_c, primary,     args.answer_reduction)
                c_al = score_answer(model, context_c, alternative, args.answer_reduction)
                baseline = np.array([a_pl, a_al, c_pl, c_al], dtype=np.float64)

         
            if needs_bl:
                bl_a_ald = float(baseline[0] - baseline[1])
                bl_c_ald = float(baseline[2] - baseline[3])
            else:
                stored = acc["row_baselines"][ri]
                if np.any(np.isnan(stored)):
                    raise ValueError(
                        f"Per-row baseline not stored for row {ri}; "
                        "reset baseline_status[ri]=0 to recompute"
                    )
                bl_a_ald = float(stored[0] - stored[1])
                bl_c_ald = float(stored[2] - stored[3])

          
            if needs_ac or needs_cc:
                _, _, pos_c = answer_encoding(model.tokenizer, context_c, primary)
                clean_c_logits = capture_clean_logits(model, context_c, pos_c[0])

            if needs_ca or needs_aa:
                _, _, pos_a = answer_encoding(model.tokenizer, context_a, primary)
                clean_a_logits = capture_clean_logits(model, context_a, pos_a[0])

            def _accum(pfx: str, ald: np.ndarray, pl: np.ndarray, al: np.ndarray,  kl: np.ndarray, ie: np.ndarray,  norm: np.ndarray) -> None:
                acc[f"{pfx}_answer_logprob_diff_sums"][ai] += ald
                acc[f"{pfx}_primary_lp_sums"][ai]          += pl
                acc[f"{pfx}_alt_lp_sums"][ai]              += al
                acc[f"{pfx}_kl_div_sums"][ai]              += kl
                acc[f"{pfx}_ie_sums"][ai]                  += ie
                acc[f"{pfx}_ie_counts"][ai]                += 1
                acc[f"{pfx}_counts"][ai]                   += 1
                finite = np.isfinite(norm)
                acc[f"{pfx}_normalized_sums"][ai][finite]   += norm[finite]
                acc[f"{pfx}_normalized_counts"][ai][finite] += 1

            def _ie_and_norm(ald: np.ndarray, clean_baseline: float, total: float) -> tuple[np.ndarray, np.ndarray]:
                ie = ald - clean_baseline
                norm = (ie / total if np.isfinite(total) and abs(total) >= args.min_total_effect else np.full_like(ald, np.nan))
                return ie, norm

            total_ac = bl_a_ald - bl_c_ald   # denominator for A→C and A→A
            total_ca = bl_c_ald - bl_a_ald   # denominator for C→A and C→C

            if needs_ac:
                ac_ald, ac_pl, ac_al, ac_kl = evaluate_patches_direction(
                    model, args.intervention,
                    context_a, context_c, primary, alternative,
                    src_a_idx, src_c_idx, clean_c_logits,
                    cache_dtype, args.answer_reduction, args.patch_batch_size,
                )
                ac_ie, ac_norm = _ie_and_norm(ac_ald, bl_c_ald, total_ac)

            
            if needs_ca:
                ca_ald, ca_pl, ca_al, ca_kl = evaluate_patches_direction(
                    model, args.intervention,
                    context_c, context_a, primary, alternative,
                    src_c_idx, src_a_idx, clean_a_logits,
                    cache_dtype, args.answer_reduction, args.patch_batch_size,
                )
                ca_ie, ca_norm = _ie_and_norm(ca_ald, bl_a_ald, total_ca)

            
            if needs_aa:
                aa_ald, aa_pl, aa_al, aa_kl = evaluate_patches_direction(
                    model, args.intervention,
                    context_a, context_a, primary, alternative,
                    src_a_idx, src_a_idx, clean_a_logits,
                    cache_dtype, args.answer_reduction, args.patch_batch_size,
                )
                aa_ie, aa_norm = _ie_and_norm(aa_ald, bl_a_ald, total_ac)

            
            if needs_cc:
                cc_ald, cc_pl, cc_al, cc_kl = evaluate_patches_direction(
                    model, args.intervention,
                    context_c, context_c, primary, alternative,
                    src_c_idx, src_c_idx, clean_c_logits,
                    cache_dtype, args.answer_reduction, args.patch_batch_size,
                )
                cc_ie, cc_norm = _ie_and_norm(cc_ald, bl_c_ald, total_ca)

            
            if needs_bl:
                acc["baseline_sums"][ai]   += baseline
                acc["baseline_counts"][ai] += 1
                acc["row_baselines"][ri]    = baseline   # per-row storage
                acc["baseline_status"][ri]  = 1
                done_bl += 1

            if needs_ac:
                _accum("ac", ac_ald, ac_pl, ac_al, ac_kl, ac_ie, ac_norm)
                acc["row_ac_ie"][ri] = ac_ie             # per-row storage for BIE
                acc["status_ac"][ri] = 1
                done_ac += 1

            if needs_ca:
                _accum("ca", ca_ald, ca_pl, ca_al, ca_kl, ca_ie, ca_norm)
                acc["row_ca_ie"][ri] = ca_ie             # per-row storage for BIE
                acc["status_ca"][ri] = 1
                done_ca += 1

            if needs_aa:
                _accum("aa", aa_ald, aa_pl, aa_al, aa_kl, aa_ie, aa_norm)
                acc["status_aa"][ri] = 1
                done_aa += 1

            if needs_cc:
                _accum("cc", cc_ald, cc_pl, cc_al, cc_kl, cc_ie, cc_norm)
                acc["status_cc"][ri] = 1
                done_cc += 1

            
            if acc["status_ac"][ri] == 1 and acc["status_ca"][ri] == 1 \
                    and acc["bie_status"][ri] == 0:
                r_ac = acc["row_ac_ie"][ri]
                r_ca = acc["row_ca_ie"][ri]
                if not np.any(np.isnan(r_ac)) and not np.any(np.isnan(r_ca)):
                    acc["bie_sums"][ai]   += r_ac - r_ca
                    acc["bie_counts"][ai] += 1
                    acc["bie_status"][ri]  = 1
                    done_bie += 1
                else:
                    acc["bie_status"][ri] = 2

        except ValueError as exc:
            print(f"  skipped: {exc}")
            for skey, flag in (
                ("baseline_status", needs_bl), ("status_ac", needs_ac),
                ("status_ca", needs_ca), ("status_aa", needs_aa),
                ("status_cc", needs_cc),
            ):
                if flag:
                    acc[skey][ri] = 2
            acc["skip_reasons"][str(ri)] = str(exc)

        except (RuntimeError, IndexError) as exc:
            print(f"  runtime error (will retry): {exc}")
            for skey, flag in (
                ("baseline_status", needs_bl), ("status_ac", needs_ac),
                ("status_ca", needs_ca), ("status_aa", needs_aa),
                ("status_cc", needs_cc),
            ):
                if flag:
                    acc[skey][ri] = 0
            acc["skip_reasons"].pop(str(ri), None)

        save_accumulator(accumulator_path, acc)
        write_summaries(output_prefix, args.intervention, abbreviations, acc)

    write_summaries(output_prefix, args.intervention, abbreviations, acc, include_bootstrap=True)
    plots = (plot_attention_heatmaps(output_prefix, acc) if args.intervention == "attention_head" else plot_mlp_results(output_prefix, abbreviations, acc))
    for p in plots:
        print(f"Plot: {p}")

    print(
        f"\nCompleted this run — baselines: {done_bl}, "
        f"A→C: {done_ac}, C→A: {done_ca}, "
        f"A→A: {done_aa}, C→C: {done_cc}, BIE pairs: {done_bie}"
    )
    print(f"Accumulator : {accumulator_path}")
    print(f"Summaries   : {output_prefix}_*.csv")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
