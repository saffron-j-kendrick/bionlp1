
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


MODEL_ID = "meta-llama/Meta-Llama-3-8B"
N_LAYERS = 32
N_HEADS = 32
D_HEAD = 128
REQUIRED_COLUMNS = {"ABBR", "Sentence_A", "Sentence_C", "target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("adam_dataset_patching_filled2_small.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/adam_causal_projection_llama8b"),
    )
    parser.add_argument("--accumulator", type=Path)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="zero means all rows after --start-row",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=16,
        help="number of source heads patched in one forward pass",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_spans(text: str, value: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(re.escape(value), text, re.IGNORECASE)]


def token_index_for_span(
    tokenizer: Any,
    text: str,
    span: tuple[int, int],
) -> int:
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    overlapping = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if start < span[1] and end > span[0]
    ]
    if not overlapping:
        raise ValueError(f"No token overlaps character span {span} in {text!r}")
    # Represent a multi-token target by its final subtoken.
    return overlapping[-1]


def locate_target_tokens(
    tokenizer: Any,
    row: dict[str, str],
) -> tuple[str, str, int, int]:
    sentence_a = row["Sentence_A"].strip()
    sentence_c = row["Sentence_C"].strip()
    target = row["target"].strip()
    if not sentence_a or not sentence_c or not target:
        raise ValueError("Sentence_A, Sentence_C, and target must be non-empty")

    spans_a = find_spans(sentence_a, target)
    spans_c = find_spans(sentence_c, target)
    if not spans_a:
        raise ValueError(f"target {target!r} not found in Sentence_A")
    if not spans_c:
        raise ValueError(f"target {target!r} not found in Sentence_C")

    return (
        sentence_a,
        sentence_c,
        token_index_for_span(tokenizer, sentence_a, spans_a[0]),
        token_index_for_span(tokenizer, sentence_c, spans_c[0]),
    )


def to_numpy(value: Any) -> np.ndarray:
    value = value.value if hasattr(value, "value") else value
    if isinstance(value, np.ndarray):
        return value
    return value.detach().float().cpu().numpy()


def attention_output_input(model: LanguageModel, layer: int) -> Any:
    """Input to o_proj, arranged as concatenated per-head value outputs."""
    return model.model.layers[layer].self_attn.o_proj.input


def capture_clean_states(
    model: LanguageModel,
    sentence: str,
    target_index: int,
    cache_dtype: torch.dtype,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Capture clean head outputs and post-layer residuals at the target."""
    saved_heads: list[Any] = []
    saved_residuals: list[Any] = []
    with model.trace(sentence):
        for layer in range(N_LAYERS):
            heads = attention_output_input(model, layer)
            saved_heads.append(
                heads[0, target_index, :].reshape(N_HEADS, D_HEAD).save()
            )
            saved_residuals.append(
                model.model.layers[layer].output[0, target_index, :].save()
            )

    clean_heads = [
        torch.from_numpy(to_numpy(value)).to(cache_dtype) for value in saved_heads
    ]
    clean_residuals = np.stack(
        [to_numpy(value).astype(np.float32) for value in saved_residuals]
    )
    return clean_heads, clean_residuals


def patch_head(
    model: LanguageModel,
    source_layer: int,
    source_head: int,
    batch_index: int,
    target_index: int,
    clean_source_heads: list[torch.Tensor],
) -> None:
    activation = attention_output_input(model, source_layer)
    start = source_head * D_HEAD
    end = start + D_HEAD
    activation[batch_index, target_index, start:end] = clean_source_heads[
        source_layer
    ][source_head].to(device=activation.device, dtype=activation.dtype)


def transmission_batch(
    model: LanguageModel,
    target_sentence: str,
    target_index: int,
    conditions: list[tuple[int, int]],
    clean_source_heads: list[torch.Tensor],
    clean_target_residuals: np.ndarray,
    directions: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return transmission, delta norm, direction norm, and dot product."""
    saved_residuals: list[Any] = []
    with model.trace([target_sentence] * len(conditions)):
        for observation_layer in range(N_LAYERS):
            # NNsight Envoys must be accessed in model execution order. Apply
            # this layer's interventions before requesting this layer's output;
            # patching all layers first and then returning to layer 0 raises
            # MissedProviderError.
            for batch_index, (source_layer, source_head) in enumerate(conditions):
                if source_layer == observation_layer:
                    patch_head(
                        model,
                        source_layer,
                        source_head,
                        batch_index,
                        target_index,
                        clean_source_heads,
                    )
            saved_residuals.append(
                model.model.layers[observation_layer]
                .output[:, target_index, :]
                .save()
            )

    shape = (len(conditions), N_LAYERS)
    scores = np.empty(shape, dtype=np.float32)
    delta_norms = np.empty(shape, dtype=np.float32)
    dot_products = np.empty(shape, dtype=np.float32)
    direction_norm_by_layer = np.linalg.norm(directions, axis=1)
    direction_norms = np.broadcast_to(
        direction_norm_by_layer,
        shape,
    ).copy()
    denominators = direction_norm_by_layer**2 + epsilon
    for observation_layer, saved in enumerate(saved_residuals):
        patched = to_numpy(saved).astype(np.float32)
        delta = patched - clean_target_residuals[observation_layer]
        dots = delta @ directions[observation_layer]
        delta_norms[:, observation_layer] = np.linalg.norm(delta, axis=1)
        dot_products[:, observation_layer] = dots
        scores[:, observation_layer] = dots / denominators[observation_layer]

    # Layers before the patched layer cannot be downstream of that patch.
    for batch_index, (source_layer, _) in enumerate(conditions):
        scores[batch_index, :source_layer] = np.nan
        delta_norms[batch_index, :source_layer] = np.nan
        direction_norms[batch_index, :source_layer] = np.nan
        dot_products[batch_index, :source_layer] = np.nan
    return scores, delta_norms, direction_norms, dot_products


def evaluate_direction(
    model: LanguageModel,
    target_sentence: str,
    target_index: int,
    clean_source_heads: list[torch.Tensor],
    clean_target_residuals: np.ndarray,
    directions: np.ndarray,
    epsilon: float,
    patch_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conditions = [
        (layer, head)
        for layer in range(N_LAYERS)
        for head in range(N_HEADS)
    ]
    shape = (N_LAYERS, N_HEADS, N_LAYERS)
    results = [
        np.full(shape, np.nan, dtype=np.float32)
        for _ in range(4)
    ]
    for start in range(0, len(conditions), patch_batch_size):
        batch = conditions[start : start + patch_batch_size]
        batch_metrics = transmission_batch(
            model,
            target_sentence,
            target_index,
            batch,
            clean_source_heads,
            clean_target_residuals,
            directions,
            epsilon,
        )
        for batch_index, (source_layer, source_head) in enumerate(batch):
            for result, batch_metric in zip(results, batch_metrics):
                result[source_layer, source_head] = batch_metric[batch_index]
    return tuple(results)


def accumulator_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "bidirectional_causal_projection",
        "model_id": MODEL_ID,
        "dataset_sha256": dataset_fingerprint(args.dataset),
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
        "epsilon": args.epsilon,
        "target_column": "target",
        "reverse_direction": "r_C - r_A",
    }


def new_accumulator(row_count: int, metadata: dict[str, Any]) -> dict[str, Any]:
    shape = (2, N_LAYERS, N_HEADS, N_LAYERS)
    return {
        "sums": np.zeros(shape, dtype=np.float64),
        "counts": np.zeros(shape, dtype=np.int64),
        "status": np.zeros(row_count, dtype=np.int8),
        "diagnostic_sums": np.zeros((3,) + shape, dtype=np.float64),
        "diagnostic_counts": np.zeros((3,) + shape, dtype=np.int64),
        "diagnostic_status": np.zeros(row_count, dtype=np.int8),
        "failures": {},
        "metadata": metadata,
    }


def load_accumulator(
    path: Path,
    row_count: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return new_accumulator(row_count, metadata)
    with np.load(path, allow_pickle=False) as data:
        stored_metadata = json.loads(str(data["metadata"].item()))
        if stored_metadata != metadata:
            raise ValueError(
                f"{path} belongs to a different dataset or configuration"
            )
        status = data["status"]
        if status.shape != (row_count,):
            raise ValueError(f"{path} has the wrong row count")
        diagnostic_shape = (3, 2, N_LAYERS, N_HEADS, N_LAYERS)
        has_diagnostics = {
            "diagnostic_sums",
            "diagnostic_counts",
            "diagnostic_status",
        }.issubset(data.files)
        return {
            "sums": data["sums"],
            "counts": data["counts"],
            "status": status,
            "diagnostic_sums": (
                data["diagnostic_sums"]
                if has_diagnostics
                else np.zeros(diagnostic_shape, dtype=np.float64)
            ),
            "diagnostic_counts": (
                data["diagnostic_counts"]
                if has_diagnostics
                else np.zeros(diagnostic_shape, dtype=np.int64)
            ),
            "diagnostic_status": (
                data["diagnostic_status"]
                if has_diagnostics
                else np.zeros(row_count, dtype=np.int8)
            ),
            "failures": json.loads(str(data["failures"].item())),
            "metadata": stored_metadata,
        }


def save_accumulator(path: Path, accumulator: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        sums=accumulator["sums"],
        counts=accumulator["counts"],
        status=accumulator["status"],
        diagnostic_sums=accumulator["diagnostic_sums"],
        diagnostic_counts=accumulator["diagnostic_counts"],
        diagnostic_status=accumulator["diagnostic_status"],
        failures=np.array(json.dumps(accumulator["failures"])),
        metadata=np.array(json.dumps(accumulator["metadata"], sort_keys=True)),
    )
    temporary.replace(path)


def safe_average(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def write_summary(prefix: Path, accumulator: dict[str, Any]) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output = prefix.with_name(prefix.name + "_transmission.csv")
    means = safe_average(accumulator["sums"], accumulator["counts"])
    residual = (means[0] + means[1]) / 2.0
    diagnostic_means = safe_average(
        accumulator["diagnostic_sums"],
        accumulator["diagnostic_counts"],
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_layer",
                "source_head",
                "observation_layer",
                "T_AC",
                "T_CA",
                "T_residual",
                "delta_norm_AC",
                "delta_norm_CA",
                "direction_norm_AC",
                "direction_norm_CA",
                "dot_product_AC",
                "dot_product_CA",
                "pair_count",
            ]
        )
        for source_layer in range(N_LAYERS):
            for source_head in range(N_HEADS):
                for observation_layer in range(source_layer, N_LAYERS):
                    writer.writerow(
                        [
                            source_layer + 1,
                            source_head,
                            observation_layer + 1,
                            means[0, source_layer, source_head, observation_layer],
                            means[1, source_layer, source_head, observation_layer],
                            residual[source_layer, source_head, observation_layer],
                            diagnostic_means[
                                0, 0, source_layer, source_head, observation_layer
                            ],
                            diagnostic_means[
                                0, 1, source_layer, source_head, observation_layer
                            ],
                            diagnostic_means[
                                1, 0, source_layer, source_head, observation_layer
                            ],
                            diagnostic_means[
                                1, 1, source_layer, source_head, observation_layer
                            ],
                            diagnostic_means[
                                2, 0, source_layer, source_head, observation_layer
                            ],
                            diagnostic_means[
                                2, 1, source_layer, source_head, observation_layer
                            ],
                            min(
                                accumulator["counts"][
                                    0, source_layer, source_head, observation_layer
                                ],
                                accumulator["counts"][
                                    1, source_layer, source_head, observation_layer
                                ],
                            ),
                        ]
                    )
    return output


def compute_head_scores(
    accumulator: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mover and injection maps for AC, CA, and their mean.

    Both returned arrays have shape [direction, source_layer, source_head],
    where direction indices 0, 1, and 2 denote A→C, C→A, and bidirectional.
    """
    means = safe_average(accumulator["sums"], accumulator["counts"])
    transmission = np.stack(
        [means[0], means[1], (means[0] + means[1]) / 2.0]
    )
    mover = np.full((3, N_LAYERS, N_HEADS), np.nan, dtype=np.float64)
    injection = np.full_like(mover, np.nan)

    for source_layer in range(N_LAYERS):
        injection[:, source_layer, :] = transmission[
            :, source_layer, :, source_layer
        ]
        if source_layer == N_LAYERS - 1:
            # D_i is empty for a head in the final layer.
            continue
        downstream = transmission[
            :, source_layer, :, source_layer + 1 :
        ]
        mover[:, source_layer, :] = np.mean(
            np.maximum(0.0, downstream),
            axis=-1,
        )
    return mover, injection


def write_head_scores(prefix: Path, accumulator: dict[str, Any]) -> Path:
    """Write one mover/injection row per source attention head."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output = prefix.with_name(prefix.name + "_head_scores.csv")
    mover, injection = compute_head_scores(accumulator)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_layer",
                "source_head",
                "downstream_layer_count",
                "M_plus_AC",
                "M_plus_CA",
                "M_plus_residual",
                "I_AC",
                "I_CA",
                "I_residual",
            ]
        )
        for source_layer in range(N_LAYERS):
            for source_head in range(N_HEADS):
                writer.writerow(
                    [
                        source_layer + 1,
                        source_head,
                        N_LAYERS - source_layer - 1,
                        mover[0, source_layer, source_head],
                        mover[1, source_layer, source_head],
                        mover[2, source_layer, source_head],
                        injection[0, source_layer, source_head],
                        injection[1, source_layer, source_head],
                        injection[2, source_layer, source_head],
                    ]
                )
    return output


def plot_head_scores(prefix: Path, accumulator: dict[str, Any]) -> Path:
    """Plot bidirectional positive-mover and injection scores by head."""
    mover, injection = compute_head_scores(accumulator)
    mover_residual = mover[2]
    injection_residual = injection[2]

    figure, axes = plt.subplots(1, 2, figsize=(18, 10))
    mover_image = axes[0].imshow(
        mover_residual,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0].set_title(r"Positive mover score $M_i^+$")
    axes[0].set_xlabel("Attention head")
    axes[0].set_ylabel("Source layer")
    axes[0].set_xticks(np.arange(N_HEADS))
    axes[0].set_yticks(np.arange(N_LAYERS))
    axes[0].set_yticklabels(np.arange(1, N_LAYERS + 1))
    figure.colorbar(mover_image, ax=axes[0], label=r"$M_i^+$")

    finite_injection = np.abs(injection_residual[np.isfinite(injection_residual)])
    injection_limit = (
        float(np.percentile(finite_injection, 99))
        if finite_injection.size
        else 1.0
    )
    if injection_limit == 0.0:
        injection_limit = 1.0
    injection_image = axes[1].imshow(
        injection_residual,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-injection_limit,
        vmax=injection_limit,
        interpolation="nearest",
    )
    axes[1].set_title(r"Injection score $I_i=T_{i,l}$")
    axes[1].set_xlabel("Attention head")
    axes[1].set_ylabel("Source layer")
    axes[1].set_xticks(np.arange(N_HEADS))
    axes[1].set_yticks(np.arange(N_LAYERS))
    axes[1].set_yticklabels(np.arange(1, N_LAYERS + 1))
    figure.colorbar(injection_image, ax=axes[1], label=r"$I_i$")

    figure.suptitle("Source-head semantic movement", fontsize=15)
    figure.tight_layout()
    output = prefix.with_name(prefix.name + "_head_scores.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_transmission(prefix: Path, accumulator: dict[str, Any]) -> Path:
    """Plot transmission by source layer and downstream layer distance."""
    sums = accumulator["sums"]
    counts = accumulator["counts"]
    means = safe_average(sums, counts)
    residual_map = (means[0] + means[1]) / 2.0

    layers = np.arange(1, N_LAYERS + 1)
    figure, axes = plt.subplots(1, 2, figsize=(17, 6))

    # Each curve follows a fixed source layer, averaged over its 32 heads.
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, N_LAYERS))
    for source_layer in range(N_LAYERS):
        source_line = np.nanmean(
            residual_map[source_layer, :, source_layer:],
            axis=0,
        )
        axes[0].plot(
            layers[source_layer:],
            source_line,
            color=colors[source_layer],
            linewidth=1.0,
            alpha=0.8,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("Observation layer")
    axes[0].set_ylabel("T residual")
    axes[0].set_title("Transmission from each source layer")
    axes[0].set_xticks(layers)
    axes[0].grid(alpha=0.2)
    scalar_mappable = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=1, vmax=N_LAYERS),
    )
    colorbar = figure.colorbar(scalar_mappable, ax=axes[0], pad=0.02)
    colorbar.set_label("Source layer")

    # Pool patches by distance from their source layer. This avoids comparing a
    # changing set of causally eligible source layers at each absolute layer.
    distances = np.arange(N_LAYERS)
    distance_lines = np.full((2, N_LAYERS), np.nan, dtype=np.float64)
    for direction in range(2):
        for distance in distances:
            distance_sum = 0.0
            distance_count = 0
            for source_layer in range(N_LAYERS - distance):
                observation_layer = source_layer + distance
                distance_sum += sums[
                    direction, source_layer, :, observation_layer
                ].sum()
                distance_count += counts[
                    direction, source_layer, :, observation_layer
                ].sum()
            if distance_count:
                distance_lines[direction, distance] = (
                    distance_sum / distance_count
                )
    distance_residual = (distance_lines[0] + distance_lines[1]) / 2.0

    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].plot(distances, distance_lines[0], marker="o", label="A→C")
    axes[1].plot(distances, distance_lines[1], marker="s", label="C→A")
    axes[1].plot(
        distances,
        distance_residual,
        marker="D",
        markersize=3,
        linewidth=2.5,
        label="T residual",
    )
    axes[1].set_xlabel("Layers downstream from patch (0 = source layer)")
    axes[1].set_ylabel("Transmission score")
    axes[1].set_title("Mean transmission by downstream distance")
    axes[1].set_xticks(distances)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()

    output = prefix.with_name(prefix.name + "_transmission_by_layer.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_diagnostics(
    prefix: Path,
    accumulator: dict[str, Any],
) -> list[Path]:
    """Plot the unnormalised terms underlying transmission by absolute layer."""
    sums = accumulator["diagnostic_sums"]
    counts = accumulator["diagnostic_counts"]
    if not np.any(counts):
        print("No diagnostic results are available to plot.")
        return []

    # Pool over causally eligible source layers and heads for each observation
    # layer. Metric order: ||delta r||, ||d||, delta r dot d.
    pooled = np.full((3, 2, N_LAYERS), np.nan, dtype=np.float64)
    for metric in range(3):
        for direction in range(2):
            layer_sums = sums[metric, direction].sum(axis=(0, 1))
            layer_counts = counts[metric, direction].sum(axis=(0, 1))
            pooled[metric, direction] = safe_average(
                layer_sums,
                layer_counts,
            )

    layers = np.arange(1, N_LAYERS + 1)
    output_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.axvspan(20, 21, color="gold", alpha=0.15, label="Layers 20–21")
    axis.plot(
        layers,
        pooled[0, 0],
        marker="o",
        markersize=3,
        label=r"$\|\Delta r_l\|$ A→C",
    )
    axis.plot(
        layers,
        pooled[0, 1],
        marker="s",
        markersize=3,
        label=r"$\|\Delta r_l\|$ C→A",
    )
    direction_norm = np.nanmean(pooled[1], axis=0)
    axis.plot(
        layers,
        direction_norm,
        marker="D",
        markersize=3,
        linewidth=2.5,
        label=r"$\|d_l\|$ clean A–C direction",
    )
    axis.set_xlabel("Observation layer")
    axis.set_ylabel("L2 norm")
    axis.set_title("Residual-change and semantic-direction scales")
    axis.set_xticks(layers)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    norms_path = prefix.with_name(prefix.name + "_diagnostic_norms.png")
    norms_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(norms_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(norms_path)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.axvspan(20, 21, color="gold", alpha=0.15, label="Layers 20–21")
    axis.plot(
        layers,
        pooled[2, 0],
        marker="o",
        markersize=3,
        label=r"$\Delta r_l \cdot d_l$ A→C",
    )
    axis.plot(
        layers,
        pooled[2, 1],
        marker="s",
        markersize=3,
        label=r"$\Delta r_l \cdot d_l$ C→A",
    )
    axis.plot(
        layers,
        np.nanmean(pooled[2], axis=0),
        marker="D",
        markersize=3,
        linewidth=2.5,
        label="Bidirectional mean",
    )
    axis.set_xlabel("Observation layer")
    axis.set_ylabel("Dot product")
    axis.set_title("Source-aligned residual propagation")
    axis.set_xticks(layers)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    dot_path = prefix.with_name(prefix.name + "_diagnostic_dot_product.png")
    figure.savefig(dot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(dot_path)
    return output_paths


def load_model(device_map: str, dtype: torch.dtype) -> LanguageModel:
    token = os.environ.get("HF_TOKEN_LLAMA")
    kwargs: dict[str, Any] = {"device_map": device_map, "torch_dtype": dtype}
    if token:
        kwargs["token"] = token
    model = LanguageModel(MODEL_ID, **kwargs)
    (model.eval if hasattr(model, "eval") else model.model.eval)()
    return model


def validate_rows(rows: list[dict[str, str]]) -> dict[int, str]:
    failures: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        target = row["target"].strip()
        if not target:
            failures[row_index] = "blank target"
        elif not find_spans(row["Sentence_A"], target):
            failures[row_index] = f"target {target!r} not in Sentence_A"
        elif not find_spans(row["Sentence_C"], target):
            failures[row_index] = f"target {target!r} not in Sentence_C"
    return failures


def run(args: argparse.Namespace) -> None:
    if args.start_row < 0 or args.max_rows < 0:
        raise ValueError("--start-row and --max-rows must be non-negative")
    if args.patch_batch_size < 1:
        raise ValueError("--patch-batch-size must be positive")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")

    rows = read_dataset(args.dataset)
    validation_failures = validate_rows(rows)
    print(
        f"Validation: {len(rows) - len(validation_failures)}/{len(rows)} "
        "rows usable"
    )
    for row_index, reason in list(validation_failures.items())[:10]:
        print(f"  row {row_index}: {reason}")
    if args.dry_run:
        return

    accumulator_path = (
        args.accumulator or args.output_prefix.with_suffix(".npz")
    )
    metadata = accumulator_metadata(args)
    accumulator = load_accumulator(accumulator_path, len(rows), metadata)

    if args.plot_only:
        summary = write_summary(args.output_prefix, accumulator)
        head_scores = write_head_scores(args.output_prefix, accumulator)
        plot = plot_transmission(args.output_prefix, accumulator)
        head_plot = plot_head_scores(args.output_prefix, accumulator)
        diagnostic_plots = plot_diagnostics(args.output_prefix, accumulator)
        print(
            f"Summary: {summary}\n"
            f"Head scores: {head_scores}\n"
            f"Plot: {plot}\n"
            f"Head-score plot: {head_plot}"
        )
        for diagnostic_plot in diagnostic_plots:
            print(f"Diagnostic plot: {diagnostic_plot}")
        return

    cache_dtype = getattr(torch, args.cache_dtype)
    model = load_model(args.device_map, cache_dtype)
    stop = (
        len(rows)
        if args.max_rows == 0
        else min(len(rows), args.start_row + args.max_rows)
    )

    completed_this_run = 0
    for row_index in range(args.start_row, stop):
        needs_transmission = accumulator["status"][row_index] != 1
        needs_diagnostics = accumulator["diagnostic_status"][row_index] != 1
        if not (needs_transmission or needs_diagnostics):
            continue
        if row_index in validation_failures:
            if needs_transmission:
                accumulator["status"][row_index] = 2
            if needs_diagnostics:
                accumulator["diagnostic_status"][row_index] = 2
            accumulator["failures"][str(row_index)] = validation_failures[row_index]
            continue

        row = rows[row_index]
        print(
            f"[{row_index + 1}/{len(rows)}] "
            f"{row['ABBR'].strip()} — {row['target'].strip()}"
        )
        try:
            sentence_a, sentence_c, index_a, index_c = locate_target_tokens(
                model.tokenizer, row
            )
            heads_a, residuals_a = capture_clean_states(
                model, sentence_a, index_a, cache_dtype
            )
            heads_c, residuals_c = capture_clean_states(
                model, sentence_c, index_c, cache_dtype
            )

            direction_ac = residuals_a - residuals_c
            direction_ca = -direction_ac
            metrics_ac = evaluate_direction(
                model,
                sentence_c,
                index_c,
                heads_a,
                residuals_c,
                direction_ac,
                args.epsilon,
                args.patch_batch_size,
            )
            metrics_ca = evaluate_direction(
                model,
                sentence_a,
                index_a,
                heads_c,
                residuals_a,
                direction_ca,
                args.epsilon,
                args.patch_batch_size,
            )

            if needs_transmission:
                for direction_index, metrics in enumerate(
                    (metrics_ac, metrics_ca)
                ):
                    scores = metrics[0]
                    finite = np.isfinite(scores)
                    accumulator["sums"][direction_index][finite] += scores[finite]
                    accumulator["counts"][direction_index][finite] += 1
                accumulator["status"][row_index] = 1

            if needs_diagnostics:
                for direction_index, metrics in enumerate(
                    (metrics_ac, metrics_ca)
                ):
                    for metric_index, values in enumerate(metrics[1:]):
                        finite = np.isfinite(values)
                        target_sums = accumulator["diagnostic_sums"][
                            metric_index, direction_index
                        ]
                        target_counts = accumulator["diagnostic_counts"][
                            metric_index, direction_index
                        ]
                        target_sums[finite] += values[finite]
                        target_counts[finite] += 1
                accumulator["diagnostic_status"][row_index] = 1

            accumulator["failures"].pop(str(row_index), None)
            completed_this_run += 1
        except ValueError as error:
            print(f"  skipped: {error}")
            if needs_transmission:
                accumulator["status"][row_index] = 2
            if needs_diagnostics:
                accumulator["diagnostic_status"][row_index] = 2
            accumulator["failures"][str(row_index)] = str(error)
        except (RuntimeError, IndexError) as error:
            # Leave status pending so an interrupted/OOM row can be retried.
            print(f"  runtime error (will retry): {error}")

        save_accumulator(accumulator_path, accumulator)
        write_summary(args.output_prefix, accumulator)

    save_accumulator(accumulator_path, accumulator)
    summary = write_summary(args.output_prefix, accumulator)
    head_scores = write_head_scores(args.output_prefix, accumulator)
    plot = plot_transmission(args.output_prefix, accumulator)
    head_plot = plot_head_scores(args.output_prefix, accumulator)
    diagnostic_plots = plot_diagnostics(args.output_prefix, accumulator)
    print(
        f"Completed rows this run: {completed_this_run}\n"
        f"Accumulator: {accumulator_path}\n"
        f"Summary: {summary}\n"
        f"Head scores: {head_scores}\n"
        f"Plot: {plot}\n"
        f"Head-score plot: {head_plot}"
    )
    for diagnostic_plot in diagnostic_plots:
        print(f"Diagnostic plot: {diagnostic_plot}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
