

# This script is going to be used to do activation patching on the adam dataset called adam_dataset_patching_filled.csv
# Sentence A : "... abdominal aortic aneurysm"
# Sentence B : "... AAA"
# Sentence C : "... aromatic amino acids"
# I want to patch the activation of "aneurysm" in Sentence A to "AAA" into sentence B and then ask the model what does "AAA" mean, abdominal aortic aneurysm or aromatic amino acids?
# I also patch the final token of Sentence C's alternative expression into the same Sentence B abbreviation token and compare the two indirect effects.
# I want to measure the logit difference between choosing the correct answer and the incorrect answer before and after the patching.
# I also want to measure Total Indirect Effect (TIE) before and after the patching.
# I want to do attention head patching and then separately do mlp patching. 
# Each abbreviation e.g. AAA has five different sentence triplets, so I want to do the patching five times for each abbreviation and then average the results. 


from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nnsight import LanguageModel


MODEL_CONFIGS = {
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
BASELINE_NAMES = ("sentence_a", "sentence_b", "sentence_c")
REQUIRED_COLUMNS = {
    "ABBR",
    "Sentence_A",
    "Sentence_B",
    "Sentence_C",
    "Primary_Expression",
    "Alternative_Expression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("adam_dataset_patching_filled2.csv"))
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), default="llama3b", help="model to patch (default: llama3b)")
    parser.add_argument("--intervention", choices=("attention_head", "mlp"), required=True, help="intervention to apply (default: attention_head)")
    parser.add_argument("--accumulator", type=Path, help="checkpoint .npz (default: outputs/<intervention>_<model>.npz)")
    parser.add_argument("--output-prefix", type=Path, help="summary path prefix (default: outputs/<intervention>_<model>)")
    parser.add_argument("--start-row", type=int, default=0, help="first zero-based CSV data row considered in this run")
    parser.add_argument("--max-rows", type=int, default=0, help="maximum rows processed this run; 0 means all remaining rows")
    parser.add_argument("--patch-batch-size", type=int, default=24, help="patch conditions evaluated in one model batch")
    parser.add_argument("--answer-reduction", choices=("mean", "sum"), default="mean", help="reduce full-answer token log probabilities (default: mean)")
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16", help="cache activations in this dtype (default: bfloat16)")
    parser.add_argument("--device-map", default="auto", help='Hugging Face device map, for example "auto" or "cpu"')
    parser.add_argument("--min-total-effect", type=float, default=1e-6, help="minimum |Sentence_A difference - Sentence_B difference|")
    parser.add_argument("--dry-run", action="store_true", help="validate rows without loading the model")
    parser.add_argument("--plot-only", action="store_true", help="regenerate summaries and plots from an existing accumulator")
    return parser.parse_args()


def read_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError("Missing dataset columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    abbreviations = list(dict.fromkeys(row["ABBR"].strip() for row in rows))
    if not rows or any(not abbreviation for abbreviation in abbreviations):
        raise ValueError("The dataset is empty or contains a blank ABBR")
    return rows, abbreviations


def dataset_fingerprint(path: Path) -> str:
    """identifies dataset in case the accumulator loads a different dataset"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_context(sentence: str, abbreviation: str) -> str:
    return (
        f"{sentence}\n\n"
        f'Question: In the sentence above, what does abbreviationstand for?\n'
        "Answer:"
    )


def find_spans(text: str, value: str) -> list[tuple[int, int]]:
    return [
        match.span()
        for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE)
    ]


def choose_abbreviation_span(
    sentence_a: str,
    sentence_b: str,
    abbreviation: str,
    primary_expression: str,
) -> tuple[int, int]:
    """Choose the ABBR occurrence whose replacement best reconstructs Sentence_A."""
    pattern = re.compile(
        rf"(?<!\w){re.escape(abbreviation)}(?!\w)", flags=re.IGNORECASE
    )
    candidates = list(pattern.finditer(sentence_b))
    if not candidates:
        raise ValueError(f"standalone abbreviation {abbreviation!r} not in Sentence_B")

    ranked: list[tuple[float, tuple[int, int]]] = []
    for match in candidates:
        reconstructed = (
            sentence_b[: match.start()]
            + primary_expression
            + sentence_b[match.end() :]
        )
        similarity = SequenceMatcher(None, reconstructed, sentence_a).ratio()
        ranked.append((similarity, match.span()))
    ranked.sort(reverse=True)
    return ranked[0][1]


def token_index_for_span(
    tokenizer: Any,
    text: str,
    span: tuple[int, int],
    *,
    use_last_token: bool,
) -> int:
    encoded = tokenizer(
        text, add_special_tokens=True, return_offsets_mapping=True
    )
    overlapping = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if start < span[1] and end > span[0]
    ]
    if not overlapping:
        raise ValueError(f"No token overlaps character span {span}")
    return overlapping[-1] if use_last_token else overlapping[0]


def locate_patch_tokens(
    tokenizer: Any, row: dict[str, str]
) -> tuple[str, str, int, int]:
    abbreviation = row["ABBR"].strip()
    primary = row["Primary_Expression"].strip()
    sentence_a = row["Sentence_A"]
    sentence_b = row["Sentence_B"]
    primary_spans = find_spans(sentence_a, primary)
    if not primary_spans:
        raise ValueError(f"Primary expression {primary!r} not in Sentence_A")

    abbreviation_span = choose_abbreviation_span(
        sentence_a, sentence_b, abbreviation, primary
    )
    context_a = build_context(sentence_a, abbreviation)
    context_b = build_context(sentence_b, abbreviation)
    source_index = token_index_for_span(
        tokenizer, context_a, primary_spans[0], use_last_token=True
    )
    target_index = token_index_for_span(
        tokenizer, context_b, abbreviation_span, use_last_token=False
    )
    return context_a, context_b, source_index, target_index


def locate_source_token(
    tokenizer: Any,
    sentence: str,
    expression: str,
    abbreviation: str,
) -> tuple[str, int]:
    """Locate the final expression token in a clean source sentence."""
    expression_spans = find_spans(sentence, expression)
    if not expression_spans:
        raise ValueError(f"Expression {expression!r} not found in source sentence")
    context = build_context(sentence, abbreviation)
    source_index = token_index_for_span(
        tokenizer, context, expression_spans[0], use_last_token=True
    )
    return context, source_index


def answer_encoding(
    tokenizer: Any, context: str, answer: str
) -> tuple[str, list[int], list[int]]:
    full_text = context + " " + answer.strip()
    context_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("Answer boundary changed context tokenization")

    answer_ids = full_ids[len(context_ids) :]
    if not answer_ids:
        raise ValueError(f"Answer {answer!r} produced no tokens")
    positions = list(range(len(context_ids) - 1, len(full_ids) - 1))
    return full_text, answer_ids, positions


def to_numpy(value: Any) -> np.ndarray:
    value = value.value if hasattr(value, "value") else value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray):
        return value
    return value.detach().float().cpu().numpy()


def score_answer(
    model: LanguageModel,
    context: str,
    answer: str,
    reduction: str,
) -> float:
    full_text, answer_ids, positions = answer_encoding(
        model.tokenizer, context, answer
    )
    targets = torch.tensor(answer_ids, dtype=torch.long)
 
    with model.trace(full_text):
        logits = model.lm_head.output[0, positions, :]
        targets = targets.to(logits.device)
        token_scores = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        score = (
            token_scores.mean() if reduction == "mean" else token_scores.sum()
        ).save()
    return float(to_numpy(score).item())


def baseline_logit_difference(
    model: LanguageModel,
    context: str,
    primary: str,
    alternative: str,
    reduction: str,
) -> float:
    return score_answer(model, context, primary, reduction) - score_answer(
        model, context, alternative, reduction
    )


def attention_module(model: LanguageModel, layer: int) -> Any:
    return model.model.layers[layer].self_attn.o_proj


def mlp_module(model: LanguageModel, layer: int) -> Any:
    return model.model.layers[layer].mlp.down_proj


def capture_clean_activations(
    model: LanguageModel,
    intervention: str,
    context_a: str,
    source_index: int,
    cache_dtype: torch.dtype,
) -> list[torch.Tensor]:
    saved = []
    with model.trace(context_a):
        for layer in range(N_LAYERS):
            if intervention == "attention_head":
                activation = attention_module(model, layer).input
                value = activation[0, source_index, :].reshape(N_HEADS, D_HEAD)
            else:
                activation = mlp_module(model, layer).input
                value = activation[0, source_index, :]
            saved.append(value.save())

    return [
        torch.from_numpy(to_numpy(value)).to(cache_dtype)
        for value in saved
    ]


def patched_answer_scores(
    model: LanguageModel,
    intervention: str,
    context_b: str,
    answer: str,
    target_index: int,
    conditions: list[tuple[int, int | None]],
    clean_activations: list[torch.Tensor],
    reduction: str,
) -> np.ndarray:
    full_text, answer_ids, positions = answer_encoding(
        model.tokenizer, context_b, answer
    )
    batch_size = len(conditions)
    targets = torch.tensor(answer_ids, dtype=torch.long)

    with model.trace([full_text] * batch_size):
        for batch_index, (layer, head) in enumerate(conditions):
            if intervention == "attention_head":
                start = int(head) * D_HEAD
                end = start + D_HEAD
                activation = attention_module(model, layer).input
                activation[batch_index, target_index, start:end] = (
                    clean_activations[layer][int(head)].to(
                        device=activation.device, dtype=activation.dtype
                    )
                )
            else:
                activation = mlp_module(model, layer).input
                activation[batch_index, target_index, :] = (
                    clean_activations[layer].to(
                        device=activation.device, dtype=activation.dtype
                    )
                )

        logits = model.lm_head.output[:, positions, :]
        targets = targets.to(logits.device)
        expanded_targets = targets.unsqueeze(0).expand(batch_size, -1)
        token_scores = torch.log_softmax(logits, dim=-1).gather(
            -1, expanded_targets.unsqueeze(-1)
        ).squeeze(-1)
        scores = (
            token_scores.mean(dim=-1)
            if reduction == "mean"
            else token_scores.sum(dim=-1)
        ).save()

    return to_numpy(scores)


def all_conditions(intervention: str) -> list[tuple[int, int | None]]:
    if intervention == "attention_head":
        return [
            (layer, head)
            for layer in range(N_LAYERS)
            for head in range(N_HEADS)
        ]
    return [(layer, None) for layer in range(N_LAYERS)]


def evaluate_patches(
    model: LanguageModel,
    intervention: str,
    context_b: str,
    primary: str,
    alternative: str,
    target_index: int,
    clean_activations: list[torch.Tensor],
    reduction: str,
    batch_size: int,
) -> np.ndarray:
    conditions = all_conditions(intervention)
    patched_differences = np.empty(len(conditions), dtype=np.float64)
    for start in range(0, len(conditions), batch_size):
        batch = conditions[start : start + batch_size]
        primary_scores = patched_answer_scores(
            model, intervention, context_b, primary, target_index,
            batch, clean_activations, reduction
        )
        alternative_scores = patched_answer_scores(
            model, intervention, context_b, alternative, target_index,
            batch, clean_activations, reduction
        )
        patched_differences[start : start + len(batch)] = (
            primary_scores - alternative_scores
        )

    shape = (
        (N_LAYERS, N_HEADS)
        if intervention == "attention_head"
        else (N_LAYERS,)
    )
    return patched_differences.reshape(shape)


def new_accumulator(
    row_count: int,
    abbreviation_count: int,
    metadata: dict[str, Any],
    result_shape: tuple[int, ...],
) -> dict[str, Any]:
    shape = (abbreviation_count,) + result_shape
    return {
        "metadata": metadata,
        "status": np.zeros(row_count, dtype=np.uint8),
        "baseline_sums": np.zeros((abbreviation_count, 3), dtype=np.float64),
        "baseline_counts": np.zeros(abbreviation_count, dtype=np.int64),
        "patched_sums": np.zeros(shape, dtype=np.float64),
        "indirect_sums": np.zeros(shape, dtype=np.float64),
        "normalized_sums": np.zeros(shape, dtype=np.float64),
        "effect_counts": np.zeros(shape, dtype=np.int64),
        "normalized_counts": np.zeros(shape, dtype=np.int64),
        "c_status": np.zeros(row_count, dtype=np.uint8),
        "c_patched_sums": np.zeros(shape, dtype=np.float64),
        "c_indirect_sums": np.zeros(shape, dtype=np.float64),
        "c_normalized_sums": np.zeros(shape, dtype=np.float64),
        "c_effect_counts": np.zeros(shape, dtype=np.int64),
        "c_normalized_counts": np.zeros(shape, dtype=np.int64),
        "skip_reasons": {},
    }


def load_accumulator(
    path: Path,
    row_count: int,
    abbreviation_count: int,
    metadata: dict[str, Any],
    result_shape: tuple[int, ...],
) -> dict[str, Any]:
    if not path.exists():
        return new_accumulator(
            row_count, abbreviation_count, metadata, result_shape
        )

    with np.load(path, allow_pickle=False) as data:
        saved_metadata = json.loads(str(data["metadata"].item()))
        if saved_metadata != metadata:
            raise ValueError(
                f"{path} belongs to a different dataset or experiment configuration"
            )
        accumulator = {
            key: data[key].copy()
            for key in (
                "status", "baseline_sums", "baseline_counts", "patched_sums",
                "indirect_sums", "normalized_sums", "effect_counts",
                "normalized_counts",
            )
        }
        shape = (abbreviation_count,) + result_shape
        accumulator["c_status"] = (
            data["c_status"].copy()
            if "c_status" in data
            else np.zeros(row_count, dtype=np.uint8)
        )
        for key, dtype in (
            ("c_patched_sums", np.float64),
            ("c_indirect_sums", np.float64),
            ("c_normalized_sums", np.float64),
            ("c_effect_counts", np.int64),
            ("c_normalized_counts", np.int64),
        ):
            accumulator[key] = (
                data[key].copy()
                if key in data
                else np.zeros(shape, dtype=dtype)
            )
        accumulator["metadata"] = saved_metadata
        accumulator["skip_reasons"] = json.loads(
            str(data["skip_reasons"].item())
        )
    return accumulator


def save_accumulator(path: Path, accumulator: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.array(json.dumps(accumulator["metadata"], sort_keys=True)),
            status=accumulator["status"],
            baseline_sums=accumulator["baseline_sums"],
            baseline_counts=accumulator["baseline_counts"],
            patched_sums=accumulator["patched_sums"],
            indirect_sums=accumulator["indirect_sums"],
            normalized_sums=accumulator["normalized_sums"],
            effect_counts=accumulator["effect_counts"],
            normalized_counts=accumulator["normalized_counts"],
            c_status=accumulator["c_status"],
            c_patched_sums=accumulator["c_patched_sums"],
            c_indirect_sums=accumulator["c_indirect_sums"],
            c_normalized_sums=accumulator["c_normalized_sums"],
            c_effect_counts=accumulator["c_effect_counts"],
            c_normalized_counts=accumulator["c_normalized_counts"],
            skip_reasons=np.array(json.dumps(accumulator["skip_reasons"])),
        )
    os.replace(temporary, path)


def safe_average(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts != 0,
    )


def nanmean_or_nan(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def write_summaries(
    prefix: Path,
    intervention: str,
    abbreviations: list[str],
    accumulator: dict[str, Any],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    baseline_means = safe_average(
        accumulator["baseline_sums"],
        accumulator["baseline_counts"][:, None],
    )
    patched_means = safe_average(
        accumulator["patched_sums"], accumulator["effect_counts"]
    )
    indirect_means = safe_average(
        accumulator["indirect_sums"], accumulator["effect_counts"]
    )
    normalized_means = safe_average(
        accumulator["normalized_sums"], accumulator["normalized_counts"]
    )
    c_patched_means = safe_average(
        accumulator["c_patched_sums"], accumulator["c_effect_counts"]
    )
    c_indirect_means = safe_average(
        accumulator["c_indirect_sums"], accumulator["c_effect_counts"]
    )
    c_normalized_means = safe_average(
        accumulator["c_normalized_sums"],
        accumulator["c_normalized_counts"],
    )
    paired_counts = (
        (accumulator["effect_counts"] == accumulator["c_effect_counts"])
        & (accumulator["effect_counts"] > 0)
    )
    indirect_difference = np.where(
        paired_counts,
        indirect_means - c_indirect_means,
        np.nan,
    )

    by_abbreviation = prefix.with_name(prefix.name + "_by_abbreviation.csv")
    with by_abbreviation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        location_columns = (
            ["layer", "head"] if intervention == "attention_head" else ["layer"]
        )
        writer.writerow(
            ["ABBR", *location_columns, "patched_logit_difference",
             "indirect_effect", "normalized_restoration",
             "c_to_b_patched_logit_difference", "c_to_b_indirect_effect",
             "c_to_b_normalized_restoration",
             "indirect_effect_difference_a_minus_c",
             "count", "c_to_b_count"]
        )
        for abbreviation_index, abbreviation in enumerate(abbreviations):
            for layer, head in all_conditions(intervention):
                index = (
                    (abbreviation_index, layer, int(head))
                    if head is not None
                    else (abbreviation_index, layer)
                )
                location = [layer, head] if head is not None else [layer]
                writer.writerow(
                    [
                        abbreviation, *location, patched_means[index],
                        indirect_means[index], normalized_means[index],
                        c_patched_means[index], c_indirect_means[index],
                        c_normalized_means[index], indirect_difference[index],
                        accumulator["effect_counts"][index],
                        accumulator["c_effect_counts"][index],
                    ]
                )

    valid_abbreviations = accumulator["baseline_counts"] > 0
    global_path = prefix.with_name(prefix.name + "_global.csv")
    with global_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        location_columns = (
            ["layer", "head"] if intervention == "attention_head" else ["layer"]
        )
        writer.writerow(
            [*location_columns, "patched_logit_difference",
             "indirect_effect", "normalized_restoration",
             "c_to_b_patched_logit_difference", "c_to_b_indirect_effect",
             "c_to_b_normalized_restoration",
             "indirect_effect_difference_a_minus_c"]
        )
        for layer, head in all_conditions(intervention):
            cell = (
                (slice(None), layer, int(head))
                if head is not None
                else (slice(None), layer)
            )
            location = [layer, head] if head is not None else [layer]
            writer.writerow(
                [
                    *location,
                    nanmean_or_nan(patched_means[cell][valid_abbreviations]),
                    nanmean_or_nan(indirect_means[cell][valid_abbreviations]),
                    nanmean_or_nan(normalized_means[cell][valid_abbreviations]),
                    nanmean_or_nan(c_patched_means[cell][valid_abbreviations]),
                    nanmean_or_nan(c_indirect_means[cell][valid_abbreviations]),
                    nanmean_or_nan(c_normalized_means[cell][valid_abbreviations]),
                    nanmean_or_nan(
                        indirect_difference[cell][valid_abbreviations]
                    ),
                ]
            )

    baseline_path = prefix.with_name(prefix.name + "_baselines.csv")
    with baseline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["ABBR", *BASELINE_NAMES, "total_effect_a_minus_b", "row_count"]
        )
        for index, abbreviation in enumerate(abbreviations):
            writer.writerow(
                [
                    abbreviation, *baseline_means[index],
                    baseline_means[index, 0] - baseline_means[index, 1],
                    accumulator["baseline_counts"][index],
                ]
            )

    status_path = prefix.with_name(prefix.name + "_status.json")
    status_path.write_text(
        json.dumps(
            {
                "processed_rows": int(np.sum(accumulator["status"] == 1)),
                "skipped_rows": int(np.sum(accumulator["status"] == 2)),
                "pending_rows": int(np.sum(accumulator["status"] == 0)),
                "c_to_b_processed_rows": int(
                    np.sum(accumulator["c_status"] == 1)
                ),
                "c_to_b_skipped_rows": int(
                    np.sum(accumulator["c_status"] == 2)
                ),
                "c_to_b_pending_rows": int(
                    np.sum(accumulator["c_status"] == 0)
                ),
                "skip_reasons": accumulator["skip_reasons"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def macro_average_attention(
    sums: np.ndarray,
    counts: np.ndarray,
    valid_abbreviations: np.ndarray,
) -> np.ndarray:
    abbreviation_means = safe_average(sums, counts)
    values = abbreviation_means[valid_abbreviations]
    finite = np.isfinite(values)
    return np.divide(
        np.nansum(values, axis=0),
        finite.sum(axis=0),
        out=np.full(values.shape[1:], np.nan, dtype=np.float64),
        where=finite.sum(axis=0) != 0,
    )


def plot_attention_heatmaps(
    prefix: Path,
    accumulator: dict[str, Any],
) -> list[Path]:
    """Plot global layer-by-head heatmaps from the accumulated results."""
    valid_abbreviations = accumulator["baseline_counts"] > 0
    if not np.any(valid_abbreviations):
        print("No processed results are available for heatmaps")
        return []

    a_indirect = safe_average(
        accumulator["indirect_sums"], accumulator["effect_counts"]
    )
    c_indirect = safe_average(
        accumulator["c_indirect_sums"], accumulator["c_effect_counts"]
    )
    paired = (
        (accumulator["effect_counts"] == accumulator["c_effect_counts"])
        & (accumulator["effect_counts"] > 0)
    )
    metrics = {
        "normalized_restoration": macro_average_attention(
            accumulator["normalized_sums"],
            accumulator["normalized_counts"],
            valid_abbreviations,
        ),
        "patched_logit_difference": macro_average_attention(
            accumulator["patched_sums"],
            accumulator["effect_counts"],
            valid_abbreviations,
        ),
        "indirect_effect": macro_average_attention(
            accumulator["indirect_sums"],
            accumulator["effect_counts"],
            valid_abbreviations,
        ),
        "c_to_b_indirect_effect": macro_average_attention(
            accumulator["c_indirect_sums"],
            accumulator["c_effect_counts"],
            valid_abbreviations,
        ),
        "indirect_effect_difference_a_minus_c": macro_average_attention(
            np.where(paired, a_indirect - c_indirect, 0.0),
            paired.astype(np.int64),
            valid_abbreviations,
        ),
    }
    titles = {
        "normalized_restoration": "Attention-head normalized restoration",
        "patched_logit_difference": "Attention-head patched logit difference",
        "indirect_effect": "Attention-head indirect effect",
        "c_to_b_indirect_effect": "Attention-head C→B indirect effect",
        "indirect_effect_difference_a_minus_c": (
            "Attention-head IE difference: A→B minus C→B"
        ),
    }

    output_paths: list[Path] = []
    for metric_name, values in metrics.items():
        finite_values = values[np.isfinite(values)]
        if metric_name == "patched_logit_difference":
            cmap = "viridis"
            center = None
            vmin = float(np.min(finite_values)) if finite_values.size else 0.0
            vmax = float(np.max(finite_values)) if finite_values.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            cmap = "RdBu_r"
            center = 0
            limit = (
                float(np.max(np.abs(finite_values)))
                if finite_values.size
                else 1.0
            )
            if limit == 0:
                limit = 1.0
            vmin, vmax = -limit, limit

        figure, axis = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            values,
            ax=axis,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=np.arange(N_HEADS),
            yticklabels=np.arange(N_LAYERS),
            cbar_kws={"label": metric_name.replace("_", " ").title()},
        )
        axis.set_xlabel("Attention head")
        axis.set_ylabel("Layer")
        axis.set_title(titles[metric_name])
        axis.tick_params(axis="y", rotation=0)
        figure.tight_layout()

        output_path = prefix.with_name(prefix.name + f"_{metric_name}_heatmap.png")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def plot_mlp_results(
    prefix: Path,
    abbreviations: list[str],
    accumulator: dict[str, Any],
) -> list[Path]:
    """Plot global MLP layer lines and abbreviation-by-layer heatmaps."""
    valid_mask = accumulator["baseline_counts"] > 0
    if not np.any(valid_mask):
        print("No processed MLP results are available for plots")
        return []

    valid_abbreviations = np.asarray(abbreviations)[valid_mask]
    a_indirect = safe_average(
        accumulator["indirect_sums"], accumulator["effect_counts"]
    )
    c_indirect = safe_average(
        accumulator["c_indirect_sums"], accumulator["c_effect_counts"]
    )
    paired = (
        (accumulator["effect_counts"] == accumulator["c_effect_counts"])
        & (accumulator["effect_counts"] > 0)
    )
    metrics = {
        "normalized_restoration": safe_average(
            accumulator["normalized_sums"],
            accumulator["normalized_counts"],
        )[valid_mask],
        "patched_logit_difference": safe_average(
            accumulator["patched_sums"],
            accumulator["effect_counts"],
        )[valid_mask],
        "indirect_effect": safe_average(
            accumulator["indirect_sums"],
            accumulator["effect_counts"],
        )[valid_mask],
        "c_to_b_indirect_effect": c_indirect[valid_mask],
        "indirect_effect_difference_a_minus_c": np.where(
            paired, a_indirect - c_indirect, np.nan
        )[valid_mask],
    }
    titles = {
        "normalized_restoration": "MLP normalized restoration",
        "patched_logit_difference": "MLP patched logit difference",
        "indirect_effect": "MLP indirect effect",
        "c_to_b_indirect_effect": "MLP C→B indirect effect",
        "indirect_effect_difference_a_minus_c": (
            "MLP IE difference: A→B minus C→B"
        ),
    }
    baseline_means = safe_average(
        accumulator["baseline_sums"],
        accumulator["baseline_counts"][:, None],
    )[valid_mask]

    output_paths: list[Path] = []
    layers = np.arange(N_LAYERS)
    for metric_name, values in metrics.items():
        global_mean = np.array(
            [nanmean_or_nan(values[:, layer]) for layer in layers]
        )

        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(layers, global_mean, marker="o", linewidth=2, markersize=4)
        axis.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        if metric_name == "patched_logit_difference":
            sentence_a_mean = nanmean_or_nan(baseline_means[:, 0])
            sentence_b_mean = nanmean_or_nan(baseline_means[:, 1])
            axis.axhline(
                sentence_a_mean,
                color="green",
                linestyle=":",
                label="Sentence A baseline",
            )
            axis.axhline(
                sentence_b_mean,
                color="orange",
                linestyle=":",
                label="Sentence B baseline",
            )
            axis.legend()
        axis.set_xlabel("Layer")
        axis.set_ylabel(metric_name.replace("_", " ").title())
        axis.set_title(titles[metric_name])
        axis.set_xticks(layers)
        axis.grid(alpha=0.25)
        figure.tight_layout()

        line_path = prefix.with_name(prefix.name + f"_{metric_name}_line.png")
        figure.savefig(line_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(line_path)

        finite_values = values[np.isfinite(values)]
        if metric_name == "patched_logit_difference":
            cmap = "viridis"
            center = None
            vmin = float(np.min(finite_values)) if finite_values.size else 0.0
            vmax = float(np.max(finite_values)) if finite_values.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            cmap = "RdBu_r"
            center = 0
            limit = (
                float(np.max(np.abs(finite_values)))
                if finite_values.size
                else 1.0
            )
            if limit == 0:
                limit = 1.0
            vmin, vmax = -limit, limit

        figure_height = max(8.0, 0.22 * len(valid_abbreviations))
        figure, axis = plt.subplots(figsize=(14, figure_height))
        sns.heatmap(
            values,
            ax=axis,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=layers,
            yticklabels=valid_abbreviations,
            cbar_kws={"label": metric_name.replace("_", " ").title()},
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Abbreviation")
        axis.set_title(titles[metric_name] + " by abbreviation")
        axis.tick_params(axis="y", rotation=0, labelsize=7)
        figure.tight_layout()

        heatmap_path = prefix.with_name(
            prefix.name + f"_{metric_name}_abbreviation_layer_heatmap.png"
        )
        figure.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(heatmap_path)

    return output_paths


def validate_rows(rows: list[dict[str, str]]) -> tuple[int, dict[int, str]]:
    valid = 0
    failures: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        try:
            primary = row["Primary_Expression"].strip()
            alternative = row["Alternative_Expression"].strip()
            if not primary or not alternative:
                raise ValueError("missing primary or alternative expression")
            if not find_spans(row["Sentence_A"], primary):
                raise ValueError("primary expression not present in Sentence_A")
            choose_abbreviation_span(
                row["Sentence_A"], row["Sentence_B"], row["ABBR"].strip(), primary
            )
            valid += 1
        except ValueError as error:
            failures[row_index] = str(error)
    return valid, failures


def load_model(
    model_id: str, device_map: str, cache_dtype: torch.dtype
) -> LanguageModel:
    token = os.environ.get("HF_TOKEN_LLAMA")
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": cache_dtype,
    }
    if token:
        kwargs["token"] = token
    model = LanguageModel(model_id, **kwargs)
    if hasattr(model, "eval"):
        model.eval()
    else:
        model.model.eval()
    return model


def run(args: argparse.Namespace) -> None:
    global N_LAYERS, N_HEADS, D_HEAD

    if args.start_row < 0 or args.max_rows < 0:
        raise ValueError("--start-row and --max-rows must be non-negative")
    if args.patch_batch_size < 1:
        raise ValueError("--patch-batch-size must be positive")

    rows, abbreviations = read_dataset(args.dataset)
    valid_count, validation_failures = validate_rows(rows)
    print(
        f"Dataset validation: {valid_count}/{len(rows)} rows usable; "
        f"{len(validation_failures)} malformed rows"
    )
    for row_index, reason in list(validation_failures.items())[:10]:
        print(f"  row {row_index}: {reason}")
    if args.dry_run:
        return

    model_config = MODEL_CONFIGS[args.model]
    model_id = str(model_config["model_id"])
    N_LAYERS = int(model_config["n_layers"])
    N_HEADS = int(model_config["n_heads"])
    D_HEAD = int(model_config["d_head"])
    cache_dtype = getattr(torch, args.cache_dtype)
    result_shape = (
        (N_LAYERS, N_HEADS)
        if args.intervention == "attention_head"
        else (N_LAYERS,)
    )
    output_prefix = args.output_prefix or Path(
        f"outputs/{args.intervention}_{args.model}"
    )
    accumulator_path = args.accumulator or output_prefix.with_suffix(".npz")
    metadata = {
        "dataset_sha256": dataset_fingerprint(args.dataset),
        "model_id": model_id,
        "intervention": args.intervention,
        "answer_reduction": args.answer_reduction,
        "cache_dtype": args.cache_dtype,
        "min_total_effect": args.min_total_effect,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
        "abbreviations": abbreviations,
    }
    if args.plot_only and not accumulator_path.exists():
        raise FileNotFoundError(
            f"Cannot use --plot-only because {accumulator_path} does not exist"
        )
    accumulator = load_accumulator(
        accumulator_path, len(rows), len(abbreviations), metadata, result_shape
    )
    # Older runs marked every runtime failure as permanently skipped. Reset
    # those rows so transient failures (for example, a device mismatch) retry.
    for row_index in np.flatnonzero(accumulator["status"] == 2):
        if int(row_index) not in validation_failures:
            accumulator["status"][row_index] = 0
            accumulator["skip_reasons"].pop(str(int(row_index)), None)
    abbreviation_to_index = {
        abbreviation: index for index, abbreviation in enumerate(abbreviations)
    }
    if args.plot_only:
        write_summaries(
            output_prefix, args.intervention, abbreviations, accumulator
        )
        plots = (
            plot_attention_heatmaps(output_prefix, accumulator)
            if args.intervention == "attention_head"
            else plot_mlp_results(output_prefix, abbreviations, accumulator)
        )
        for plot in plots:
            print(f"Plot: {plot}")
        return

    model = load_model(model_id, args.device_map, cache_dtype)

    stop = (
        len(rows)
        if args.max_rows == 0
        else min(len(rows), args.start_row + args.max_rows)
    )
    completed_a_this_run = 0
    completed_c_this_run = 0
    for row_index in range(args.start_row, stop):
        needs_a = accumulator["status"][row_index] == 0
        needs_c = accumulator["c_status"][row_index] == 0
        if not needs_a and not needs_c:
            continue
        row = rows[row_index]
        abbreviation = row["ABBR"].strip()
        abbreviation_index = abbreviation_to_index[abbreviation]
        print(f"[{row_index + 1}/{len(rows)}] {abbreviation}")

        try:
            if row_index in validation_failures:
                raise ValueError(validation_failures[row_index])
            context_a, context_b, source_a_index, target_index = locate_patch_tokens(
                model.tokenizer, row
            )
            primary = row["Primary_Expression"].strip()
            alternative = row["Alternative_Expression"].strip()
            context_c, source_c_index = locate_source_token(
                model.tokenizer,
                row["Sentence_C"],
                alternative,
                abbreviation,
            )

            if needs_a:
                baseline = np.array([
                    baseline_logit_difference(
                        model, context, primary, alternative, args.answer_reduction
                    )
                    for context in (context_a, context_b, context_c)
                ], dtype=np.float64)
            else:
                baseline = np.array([
                    np.nan,
                    baseline_logit_difference(
                        model, context_b, primary, alternative,
                        args.answer_reduction
                    ),
                    baseline_logit_difference(
                        model, context_c, primary, alternative,
                        args.answer_reduction
                    ),
                ], dtype=np.float64)

            if needs_a:
                clean_a = capture_clean_activations(
                    model, args.intervention, context_a,
                    source_a_index, cache_dtype
                )
                patched_a = evaluate_patches(
                    model, args.intervention, context_b, primary, alternative,
                    target_index, clean_a, args.answer_reduction,
                    args.patch_batch_size
                )
                indirect_a = patched_a - baseline[1]
                total_effect_a = baseline[0] - baseline[1]
                normalized_a = (
                    indirect_a / total_effect_a
                    if abs(total_effect_a) >= args.min_total_effect
                    else np.full_like(indirect_a, np.nan)
                )

            if needs_c:
                clean_c = capture_clean_activations(
                    model, args.intervention, context_c,
                    source_c_index, cache_dtype
                )
                patched_c = evaluate_patches(
                    model, args.intervention, context_b, primary, alternative,
                    target_index, clean_c, args.answer_reduction,
                    args.patch_batch_size
                )
                indirect_c = patched_c - baseline[1]
                total_effect_c = baseline[2] - baseline[1]
                normalized_c = (
                    indirect_c / total_effect_c
                    if abs(total_effect_c) >= args.min_total_effect
                    else np.full_like(indirect_c, np.nan)
                )

            if needs_a:
                accumulator["baseline_sums"][abbreviation_index] += baseline
                accumulator["baseline_counts"][abbreviation_index] += 1
                accumulator["patched_sums"][abbreviation_index] += patched_a
                accumulator["indirect_sums"][abbreviation_index] += indirect_a
                accumulator["effect_counts"][abbreviation_index] += 1
                finite_a = np.isfinite(normalized_a)
                accumulator["normalized_sums"][abbreviation_index][finite_a] += (
                    normalized_a[finite_a]
                )
                accumulator["normalized_counts"][abbreviation_index][finite_a] += 1
                accumulator["status"][row_index] = 1
                completed_a_this_run += 1

            if needs_c:
                accumulator["c_patched_sums"][abbreviation_index] += patched_c
                accumulator["c_indirect_sums"][abbreviation_index] += indirect_c
                accumulator["c_effect_counts"][abbreviation_index] += 1
                finite_c = np.isfinite(normalized_c)
                accumulator["c_normalized_sums"][abbreviation_index][finite_c] += (
                    normalized_c[finite_c]
                )
                accumulator["c_normalized_counts"][abbreviation_index][finite_c] += 1
                accumulator["c_status"][row_index] = 1
                completed_c_this_run += 1
        except ValueError as error:
            print(f"  skipped: {error}")
            if needs_a:
                accumulator["status"][row_index] = 2
            if needs_c:
                accumulator["c_status"][row_index] = 2
            accumulator["skip_reasons"][str(row_index)] = str(error)
        except (RuntimeError, IndexError) as error:
            print(f"  runtime failure; row will be retried: {error}")
            if needs_a:
                accumulator["status"][row_index] = 0
            if needs_c:
                accumulator["c_status"][row_index] = 0
            accumulator["skip_reasons"].pop(str(row_index), None)

        save_accumulator(accumulator_path, accumulator)
        write_summaries(
            output_prefix, args.intervention, abbreviations, accumulator
        )

    if args.intervention == "attention_head":
        plots = plot_attention_heatmaps(output_prefix, accumulator)
    else:
        plots = plot_mlp_results(output_prefix, abbreviations, accumulator)
    for plot in plots:
        print(f"Plot: {plot}")

    print(f"Processed {completed_a_this_run} new A→B rows")
    print(f"Processed {completed_c_this_run} new C→B rows")
    print(f"Accumulator: {accumulator_path}")
    print(f"Summaries: {output_prefix}_*.csv")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

