
"""Attention-head and MLP activation patching on the ADAM dataset.

Sentence_A is the clean primary-expansion context, Sentence_B is the
abbreviation context, and Sentence_C is the alternative-expansion context.
The final token of Primary_Expression in Sentence_A is patched into the
abbreviation token in Sentence_B.
"""

# This script is going to be used to do activation patching on the adam dataset called adam_dataset_patching_filled.csv
# Sentence A : "... abdominal aortic aneurysm"
# Sentence B : "... AAA"
# Sentence C : "... aromatic amino acids"
# I want to patch the activation of "aneurysm" in Sentence A to "AAA" into sentence B and then ask the model what does "AAA" mean, abdominal aortic aneurysm or aromatic amino acids?
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

# Set from the selected model at the start of run().
N_LAYERS = 28
N_HEADS = 24
D_HEAD = 128
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
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("adam_dataset_patching_filled.csv"),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="llama3b",
        help="model to patch (default: llama3b)",
    )
    parser.add_argument(
        "--intervention", choices=("attention_head", "mlp"), required=True
    )
    parser.add_argument(
        "--accumulator",
        type=Path,
        help="checkpoint .npz (default: outputs/<intervention>_<model>.npz)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="summary path prefix (default: outputs/<intervention>_<model>)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="first zero-based CSV data row considered in this run",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="maximum rows processed this run; 0 means all remaining rows",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=24,
        help="patch conditions evaluated in one model batch",
    )
    parser.add_argument(
        "--answer-reduction",
        choices=("mean", "sum"),
        default="mean",
        help="reduce full-answer token log probabilities (default: mean)",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Hugging Face device map, for example "auto" or "cpu"',
    )
    parser.add_argument(
        "--min-total-effect",
        type=float,
        default=1e-6,
        help="minimum |Sentence_A difference - Sentence_B difference|",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate rows without loading the model",
    )
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_context(sentence: str, abbreviation: str) -> str:
    return (
        f"{sentence}\n\n"
        f'Question: In the sentence above, what does "{abbreviation}" stand for?\n'
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
                    clean_activations[layer][int(head)].to(activation.dtype)
                )
            else:
                activation = mlp_module(model, layer).input
                activation[batch_index, target_index, :] = (
                    clean_activations[layer].to(activation.dtype)
                )

        logits = model.lm_head.output[:, positions, :]
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

    by_abbreviation = prefix.with_name(prefix.name + "_by_abbreviation.csv")
    with by_abbreviation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        location_columns = (
            ["layer", "head"] if intervention == "attention_head" else ["layer"]
        )
        writer.writerow(
            ["ABBR", *location_columns, "patched_logit_difference",
             "indirect_effect", "normalized_restoration", "count"]
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
                        accumulator["effect_counts"][index],
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
             "indirect_effect", "normalized_restoration"]
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
                    np.nanmean(patched_means[cell][valid_abbreviations]),
                    np.nanmean(indirect_means[cell][valid_abbreviations]),
                    np.nanmean(normalized_means[cell][valid_abbreviations]),
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
                "skip_reasons": accumulator["skip_reasons"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
    accumulator = load_accumulator(
        accumulator_path, len(rows), len(abbreviations), metadata, result_shape
    )
    abbreviation_to_index = {
        abbreviation: index for index, abbreviation in enumerate(abbreviations)
    }
    model = load_model(model_id, args.device_map, cache_dtype)

    stop = (
        len(rows)
        if args.max_rows == 0
        else min(len(rows), args.start_row + args.max_rows)
    )
    completed_this_run = 0
    for row_index in range(args.start_row, stop):
        if accumulator["status"][row_index] != 0:
            continue
        row = rows[row_index]
        abbreviation = row["ABBR"].strip()
        abbreviation_index = abbreviation_to_index[abbreviation]
        print(f"[{row_index + 1}/{len(rows)}] {abbreviation}")

        try:
            if row_index in validation_failures:
                raise ValueError(validation_failures[row_index])
            context_a, context_b, source_index, target_index = locate_patch_tokens(
                model.tokenizer, row
            )
            context_c = build_context(row["Sentence_C"], abbreviation)
            primary = row["Primary_Expression"].strip()
            alternative = row["Alternative_Expression"].strip()
            baseline = np.array(
                [
                    baseline_logit_difference(
                        model, context, primary, alternative, args.answer_reduction
                    )
                    for context in (context_a, context_b, context_c)
                ],
                dtype=np.float64,
            )
            clean_activations = capture_clean_activations(
                model, args.intervention, context_a, source_index, cache_dtype
            )
            patched = evaluate_patches(
                model, args.intervention, context_b, primary, alternative,
                target_index, clean_activations, args.answer_reduction,
                args.patch_batch_size
            )
            indirect = patched - baseline[1]
            total_effect = baseline[0] - baseline[1]
            normalized = (
                indirect / total_effect
                if abs(total_effect) >= args.min_total_effect
                else np.full_like(indirect, np.nan)
            )

            accumulator["baseline_sums"][abbreviation_index] += baseline
            accumulator["baseline_counts"][abbreviation_index] += 1
            accumulator["patched_sums"][abbreviation_index] += patched
            accumulator["indirect_sums"][abbreviation_index] += indirect
            accumulator["effect_counts"][abbreviation_index] += 1
            finite = np.isfinite(normalized)
            accumulator["normalized_sums"][abbreviation_index][finite] += (
                normalized[finite]
            )
            accumulator["normalized_counts"][abbreviation_index][finite] += 1
            accumulator["status"][row_index] = 1
            completed_this_run += 1
        except (ValueError, RuntimeError, IndexError) as error:
            print(f"  skipped: {error}")
            accumulator["status"][row_index] = 2
            accumulator["skip_reasons"][str(row_index)] = str(error)

        save_accumulator(accumulator_path, accumulator)
        write_summaries(
            output_prefix, args.intervention, abbreviations, accumulator
        )

    print(f"Processed {completed_this_run} new rows")
    print(f"Accumulator: {accumulator_path}")
    print(f"Summaries: {output_prefix}_*.csv")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

