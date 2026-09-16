#!/usr/bin/env python3
"""Compute and plot example-level diagnostics for selected ADAM components."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

import adam_data_patching as patching


MLP_LAYERS = [0, 14, 31]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("adam_dataset_patching_filled.csv"),
    )
    parser.add_argument(
        "--model",
        choices=tuple(patching.MODEL_CONFIGS),
        default="llama8b",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="number of rows to process; 0 means all remaining rows",
    )
    parser.add_argument(
        "--answer-reduction",
        choices=("mean", "sum"),
        default="mean",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-total-effect", type=float, default=1e-6)
    parser.add_argument(
        "--attention-global",
        type=Path,
        help="global attention CSV used to select the strongest negative head",
    )
    parser.add_argument(
        "--attention-status",
        type=Path,
        help="attention status JSON used to check whether selection is final",
    )
    parser.add_argument(
        "--allow-partial-selection",
        action="store_true",
        help="allow head selection from an unfinished attention run",
    )
    parser.add_argument(
        "--num-negative-heads",
        type=int,
        default=2,
        help="number of most-negative heads selected automatically (default: 2)",
    )
    parser.add_argument(
        "--num-positive-heads",
        type=int,
        default=2,
        help="number of most-positive heads selected automatically (default: 2)",
    )
    parser.add_argument("--accumulator", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="plot an existing diagnostics accumulator without loading a model",
    )
    return parser.parse_args()


def select_extreme_heads(
    path: Path,
    num_negative: int,
    num_positive: int,
) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Attention summary not found: {path}")

    candidates: list[tuple[float, int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["normalized_restoration"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                candidates.append((value, int(row["layer"]), int(row["head"])))

    required = num_negative + num_positive
    if len(candidates) < required:
        raise ValueError(
            f"Need {required} finite values in {path}, found {len(candidates)}"
        )

    candidates.sort()
    negative = candidates[:num_negative]
    positive = (
        list(reversed(candidates[-num_positive:]))
        if num_positive
        else []
    )
    return [
        (layer, head)
        for _, layer, head in [*negative, *positive]
    ]


def check_attention_complete(path: Path, allow_partial: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Attention status not found: {path}")
    status = json.loads(path.read_text(encoding="utf-8"))
    pending = int(status.get("pending_rows", 0))
    if pending and not allow_partial:
        raise ValueError(
            f"The attention run still has {pending} pending rows. Finish it "
            "before selecting the strongest negative head, or pass "
            "--allow-partial-selection for an exploratory result."
        )


def new_accumulator(
    row_count: int,
    attention_heads: list[tuple[int, int]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "status": np.zeros(row_count, dtype=np.uint8),
        "baselines": np.full((row_count, 3), np.nan, dtype=np.float64),
        "attention_patched": np.full(
            (row_count, len(attention_heads)), np.nan, dtype=np.float64
        ),
        "mlp_patched": np.full(
            (row_count, len(MLP_LAYERS)), np.nan, dtype=np.float64
        ),
        "skip_reasons": {},
    }


def load_accumulator(
    path: Path,
    row_count: int,
    attention_heads: list[tuple[int, int]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return new_accumulator(row_count, attention_heads, metadata)

    with np.load(path, allow_pickle=False) as data:
        saved_metadata = json.loads(str(data["metadata"].item()))
        if saved_metadata != metadata:
            raise ValueError(
                f"{path} belongs to a different diagnostic configuration"
            )
        return {
            "metadata": saved_metadata,
            "status": data["status"].copy(),
            "baselines": data["baselines"].copy(),
            "attention_patched": data["attention_patched"].copy(),
            "mlp_patched": data["mlp_patched"].copy(),
            "skip_reasons": json.loads(str(data["skip_reasons"].item())),
        }


def save_accumulator(path: Path, accumulator: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.array(json.dumps(accumulator["metadata"], sort_keys=True)),
            status=accumulator["status"],
            baselines=accumulator["baselines"],
            attention_patched=accumulator["attention_patched"],
            mlp_patched=accumulator["mlp_patched"],
            skip_reasons=np.array(json.dumps(accumulator["skip_reasons"])),
        )
    os.replace(temporary, path)


def selected_patched_differences(
    model: Any,
    intervention: str,
    conditions: list[tuple[int, int | None]],
    context_a: str,
    context_b: str,
    source_index: int,
    target_index: int,
    primary: str,
    alternative: str,
    reduction: str,
    cache_dtype: torch.dtype,
) -> np.ndarray:
    clean = patching.capture_clean_activations(
        model, intervention, context_a, source_index, cache_dtype
    )
    execution_order = sorted(
        range(len(conditions)),
        key=lambda index: (
            conditions[index][0],
            -1 if conditions[index][1] is None else conditions[index][1],
        ),
    )
    ordered_conditions = [conditions[index] for index in execution_order]
    primary_scores = patching.patched_answer_scores(
        model,
        intervention,
        context_b,
        primary,
        target_index,
        ordered_conditions,
        clean,
        reduction,
    )
    alternative_scores = patching.patched_answer_scores(
        model,
        intervention,
        context_b,
        alternative,
        target_index,
        ordered_conditions,
        clean,
        reduction,
    )
    ordered_differences = primary_scores - alternative_scores
    differences = np.empty_like(ordered_differences)
    for ordered_index, original_index in enumerate(execution_order):
        differences[original_index] = ordered_differences[ordered_index]
    return differences


def evaluate_row(
    model: Any,
    row: dict[str, str],
    attention_heads: list[tuple[int, int]],
    reduction: str,
    cache_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abbreviation = row["ABBR"].strip()
    primary = row["Primary_Expression"].strip()
    alternative = row["Alternative_Expression"].strip()
    context_a, context_b, source_index, target_index = (
        patching.locate_patch_tokens(model.tokenizer, row)
    )
    context_c = patching.build_context(row["Sentence_C"], abbreviation)
    baselines = np.array(
        [
            patching.baseline_logit_difference(
                model, context, primary, alternative, reduction
            )
            for context in (context_a, context_b, context_c)
        ]
    )
    attention = selected_patched_differences(
        model,
        "attention_head",
        [(layer, head) for layer, head in attention_heads],
        context_a,
        context_b,
        source_index,
        target_index,
        primary,
        alternative,
        reduction,
        cache_dtype,
    )
    mlp = selected_patched_differences(
        model,
        "mlp",
        [(layer, None) for layer in MLP_LAYERS],
        context_a,
        context_b,
        source_index,
        target_index,
        primary,
        alternative,
        reduction,
        cache_dtype,
    )
    return baselines, attention, mlp


def example_numbers(rows: list[dict[str, str]]) -> np.ndarray:
    counts: dict[str, int] = {}
    numbers = np.zeros(len(rows), dtype=int)
    for index, row in enumerate(rows):
        abbreviation = row["ABBR"].strip()
        counts[abbreviation] = counts.get(abbreviation, 0) + 1
        numbers[index] = counts[abbreviation]
    return numbers


def write_diagnostic_csv(
    path: Path,
    rows: list[dict[str, str]],
    attention_heads: list[tuple[int, int]],
    accumulator: dict[str, Any],
    min_total_effect: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numbers = example_numbers(rows)
    component_names = [
        *(f"L{layer}H{head}" for layer, head in attention_heads),
        *(f"MLP_L{layer}" for layer in MLP_LAYERS),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        metric_columns = [
            f"{component}_{metric}"
            for component in component_names
            for metric in (
                "patched_logit_difference",
                "indirect_effect",
                "normalized_restoration",
            )
        ]
        writer.writerow(
            [
                "row_index",
                "ABBR",
                "example",
                "D_A",
                "D_B",
                "D_C",
                "D_A_minus_D_B",
                *metric_columns,
            ]
        )
        for row_index in np.flatnonzero(accumulator["status"] == 1):
            baselines = accumulator["baselines"][row_index]
            patched = np.concatenate(
                (
                    accumulator["attention_patched"][row_index],
                    accumulator["mlp_patched"][row_index],
                )
            )
            indirect = patched - baselines[1]
            denominator = baselines[0] - baselines[1]
            normalized = (
                indirect / denominator
                if abs(denominator) >= min_total_effect
                else np.full_like(indirect, np.nan)
            )
            metrics = [
                value
                for component_values in zip(patched, indirect, normalized)
                for value in component_values
            ]
            writer.writerow(
                [
                    int(row_index),
                    rows[row_index]["ABBR"],
                    numbers[row_index],
                    *baselines,
                    denominator,
                    *metrics,
                ]
            )


def plot_total_effect_diagnostics(
    prefix: Path,
    rows: list[dict[str, str]],
    accumulator: dict[str, Any],
    min_total_effect: float,
) -> list[Path]:
    processed = np.flatnonzero(accumulator["status"] == 1)
    if processed.size == 0:
        return []
    effects = (
        accumulator["baselines"][processed, 0]
        - accumulator["baselines"][processed, 1]
    )
    small_fraction = float(np.mean(np.abs(effects) < min_total_effect))

    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(effects, bins=40, kde=True, ax=axis)
    sns.rugplot(effects, ax=axis, height=0.05, alpha=0.35)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.axvline(
        min_total_effect, color="red", linestyle=":", linewidth=1
    )
    axis.axvline(
        -min_total_effect, color="red", linestyle=":", linewidth=1
    )
    axis.set_xlabel(r"$D_A-D_B$")
    axis.set_ylabel("Number of examples")
    axis.set_title(
        rf"Distribution of $D_A-D_B$ ({small_fraction:.1%} below threshold)"
    )
    figure.tight_layout()
    distribution_path = prefix.with_name(
        prefix.name + "_total_effect_distribution.png"
    )
    figure.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    abbreviations = np.array([rows[index]["ABBR"] for index in processed])
    unique = np.unique(abbreviations)
    means = {
        abbreviation: float(np.mean(effects[abbreviations == abbreviation]))
        for abbreviation in unique
    }
    ordered = sorted(unique, key=means.get)
    positions = {abbreviation: index for index, abbreviation in enumerate(ordered)}
    numbers = example_numbers(rows)

    figure, axis = plt.subplots(figsize=(max(18, len(ordered) * 0.28), 7))
    for row_index, effect, abbreviation in zip(processed, effects, abbreviations):
        jitter = (numbers[row_index] - 3) * 0.07
        axis.scatter(
            positions[abbreviation] + jitter,
            effect,
            s=18,
            alpha=0.7,
        )
    axis.plot(
        np.arange(len(ordered)),
        [means[abbreviation] for abbreviation in ordered],
        color="black",
        linewidth=1,
        marker="_",
        label="Abbreviation mean",
    )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(ordered)))
    axis.set_xticklabels(ordered, rotation=90, fontsize=7)
    axis.set_xlabel("Abbreviation, ordered by mean")
    axis.set_ylabel(r"$D_A-D_B$")
    axis.set_title(r"Example-level $D_A-D_B$ by abbreviation")
    axis.legend()
    figure.tight_layout()
    abbreviation_path = prefix.with_name(
        prefix.name + "_total_effect_by_abbreviation.png"
    )
    figure.savefig(abbreviation_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return [distribution_path, abbreviation_path]


def plot_component_effects(
    prefix: Path,
    rows: list[dict[str, str]],
    attention_heads: list[tuple[int, int]],
    accumulator: dict[str, Any],
    min_total_effect: float,
) -> list[Path]:
    processed = np.flatnonzero(accumulator["status"] == 1)
    if processed.size == 0:
        return []

    component_names = [
        *(f"L{layer}H{head}" for layer, head in attention_heads),
        *(f"MLP L{layer}" for layer in MLP_LAYERS),
    ]
    patched = np.concatenate(
        (
            accumulator["attention_patched"][processed],
            accumulator["mlp_patched"][processed],
        ),
        axis=1,
    )
    baselines = accumulator["baselines"][processed]
    indirect = patched - baselines[:, 1, None]
    denominators = baselines[:, 0] - baselines[:, 1]
    normalized = np.divide(
        indirect,
        denominators[:, None],
        out=np.full_like(indirect, np.nan),
        where=np.abs(denominators[:, None]) >= min_total_effect,
    )
    metrics = {
        "patched_logit_difference": patched,
        "indirect_effect": indirect,
        "normalized_restoration": normalized,
    }
    abbreviations = np.array([rows[index]["ABBR"] for index in processed])
    unique = np.unique(abbreviations)
    numbers = example_numbers(rows)
    output_paths: list[Path] = []

    for metric_name, values in metrics.items():
        ordering_score = {
            abbreviation: float(
                np.nanmean(values[abbreviations == abbreviation])
            )
            for abbreviation in unique
        }
        ordered = sorted(unique, key=ordering_score.get)
        positions = {
            abbreviation: index for index, abbreviation in enumerate(ordered)
        }
        figure, axes = plt.subplots(
            len(component_names),
            1,
            figsize=(max(20, len(ordered) * 0.30), 3.5 * len(component_names)),
            sharex=True,
        )
        for component_index, (axis, component) in enumerate(
            zip(axes, component_names)
        ):
            for local_index, row_index in enumerate(processed):
                abbreviation = abbreviations[local_index]
                jitter = (numbers[row_index] - 3) * 0.07
                axis.scatter(
                    positions[abbreviation] + jitter,
                    values[local_index, component_index],
                    s=13,
                    alpha=0.65,
                )
            component_means = [
                np.nanmean(
                    values[
                        abbreviations == abbreviation,
                        component_index,
                    ]
                )
                for abbreviation in ordered
            ]
            axis.plot(
                np.arange(len(ordered)),
                component_means,
                color="black",
                linewidth=1,
            )
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_ylabel(component)
            axis.grid(axis="y", alpha=0.2)

        axes[-1].set_xticks(np.arange(len(ordered)))
        axes[-1].set_xticklabels(ordered, rotation=90, fontsize=7)
        axes[-1].set_xlabel("Abbreviation")
        figure.suptitle(
            metric_name.replace("_", " ").title()
            + " for selected attention heads and MLP layers"
        )
        figure.tight_layout()
        output_path = prefix.with_name(
            prefix.name + f"_selected_components_{metric_name}.png"
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def create_plots(
    prefix: Path,
    rows: list[dict[str, str]],
    attention_heads: list[tuple[int, int]],
    accumulator: dict[str, Any],
    min_total_effect: float,
) -> list[Path]:
    write_diagnostic_csv(
        prefix.with_name(prefix.name + "_example_results.csv"),
        rows,
        attention_heads,
        accumulator,
        min_total_effect,
    )
    return [
        *plot_total_effect_diagnostics(
            prefix, rows, accumulator, min_total_effect
        ),
        *plot_component_effects(
            prefix,
            rows,
            attention_heads,
            accumulator,
            min_total_effect,
        ),
    ]


def run(args: argparse.Namespace) -> None:
    if args.start_row < 0 or args.max_rows < 0:
        raise ValueError("--start-row and --max-rows must be non-negative")
    if args.num_negative_heads < 0 or args.num_positive_heads < 0:
        raise ValueError("Head selection counts must be non-negative")
    if args.num_negative_heads + args.num_positive_heads == 0:
        raise ValueError("Select at least one positive or negative head")

    model_config = patching.MODEL_CONFIGS[args.model]
    patching.N_LAYERS = int(model_config["n_layers"])
    patching.N_HEADS = int(model_config["n_heads"])
    patching.D_HEAD = int(model_config["d_head"])
    model_id = str(model_config["model_id"])

    attention_global = args.attention_global or Path(
        f"outputs/attention_head_{args.model}_global.csv"
    )
    attention_status = args.attention_status or Path(
        f"outputs/attention_head_{args.model}_status.json"
    )
    output_prefix = args.output_prefix or Path(
        f"outputs/diagnostics_{args.model}"
    )
    accumulator_path = args.accumulator or output_prefix.with_suffix(".npz")

    rows, _ = patching.read_dataset(args.dataset)
    _, validation_failures = patching.validate_rows(rows)

    if accumulator_path.exists():
        with np.load(accumulator_path, allow_pickle=False) as data:
            existing_metadata = json.loads(str(data["metadata"].item()))
        attention_heads = [
            tuple(item) for item in existing_metadata["attention_heads"]
        ]
    else:
        check_attention_complete(
            attention_status, args.allow_partial_selection
        )
        attention_heads = select_extreme_heads(
            attention_global,
            args.num_negative_heads,
            args.num_positive_heads,
        )
    print(
        "Selected attention heads: "
        + ", ".join(f"L{layer}H{head}" for layer, head in attention_heads)
    )

    metadata = {
        "dataset_sha256": patching.dataset_fingerprint(args.dataset),
        "model_id": model_id,
        "answer_reduction": args.answer_reduction,
        "cache_dtype": args.cache_dtype,
        "min_total_effect": args.min_total_effect,
        "attention_heads": attention_heads,
        "mlp_layers": MLP_LAYERS,
    }
    accumulator = load_accumulator(
        accumulator_path, len(rows), attention_heads, metadata
    )

    if args.plot_only:
        plots = create_plots(
            output_prefix,
            rows,
            attention_heads,
            accumulator,
            args.min_total_effect,
        )
        for plot in plots:
            print(f"Plot: {plot}")
        return

    cache_dtype = getattr(torch, args.cache_dtype)
    model = patching.load_model(model_id, args.device_map, cache_dtype)
    stop = (
        len(rows)
        if args.max_rows == 0
        else min(len(rows), args.start_row + args.max_rows)
    )
    completed = 0
    for row_index in range(args.start_row, stop):
        if accumulator["status"][row_index] != 0:
            continue
        print(f"[{row_index + 1}/{len(rows)}] {rows[row_index]['ABBR']}")
        try:
            if row_index in validation_failures:
                raise ValueError(validation_failures[row_index])
            baselines, attention, mlp = evaluate_row(
                model,
                rows[row_index],
                attention_heads,
                args.answer_reduction,
                cache_dtype,
            )
            accumulator["baselines"][row_index] = baselines
            accumulator["attention_patched"][row_index] = attention
            accumulator["mlp_patched"][row_index] = mlp
            accumulator["status"][row_index] = 1
            completed += 1
        except ValueError as error:
            accumulator["status"][row_index] = 2
            accumulator["skip_reasons"][str(row_index)] = str(error)
            print(f"  skipped: {error}")
        except (RuntimeError, IndexError) as error:
            print(f"  runtime failure; row remains pending: {error}")
        save_accumulator(accumulator_path, accumulator)

    plots = create_plots(
        output_prefix,
        rows,
        attention_heads,
        accumulator,
        args.min_total_effect,
    )
    for plot in plots:
        print(f"Plot: {plot}")
    print(f"Processed {completed} new rows")
    print(f"Accumulator: {accumulator_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
