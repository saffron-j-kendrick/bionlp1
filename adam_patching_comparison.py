## IMPORTS

from __future__ import annotations
import argparse
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import adam_ac_patching as _base
from pathlib import Path
from typing import Any

## POSITION

POSITION_CONFIGS: dict[str,dict[str,Any]] = {
    "start": {
        "dataset" : Path("adam_dataset_patching_filled_start.csv"),
        "label" : "Start",
        "colour" : "steelblue",
    },
    "end": {
        "dataset" : Path("adam_dataset_patching_filled_end.csv"),
        "label" : "End",
        "colour" : "darkorange",
    }
}

def _output_prefix(position: str, intervention: str, model: str) -> Path:
    return Path(f"outputs/ac_{position}_{intervention}_{model}")


## ARGUMENTS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--position", choices=("start", "end"), help="which subset to process (not needed with --compare-only)")
    p.add_argument("--compare-only", action="store_true", help="load both runs and produce comparison plots")
    p.add_argument("--model", choices=tuple(_base.MODEL_CONFIGS), default="llama8b")
    p.add_argument("--intervention", choices=("attention_head", "mlp"), required=True)
    p.add_argument("--accumulator", type=Path)
    p.add_argument("--output-prefix", type=Path, dest="output_prefix_override")
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--answer-reduction", choicecs=("mean", "sum"), default="mean")
    p.add_argument("--cache-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--min-total-effect", type=float, default=1e-6)
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    if not args.compare_only and args.position is None:
        p.error("--position (start, end) is required unless --compare-only is selected")
    
    return args


## GLOBAL CSV

_NON_METRIC_COLS = frozenset({"layer", "head", "bootstrap_resamples"})

def load_global_csv(csv_path: Path, intervention:str, n_layers: int, n_heads: int) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    metric_cols = [c for c in fieldnames if c not in _NON_METRIC_COLS]

    result: dict[str, np.ndarray] = {}

    if intervention=="attention_head":
        for col in metric_cols:
            mat = np.full((n_layers, n_heads), np.nan, dtype=np.float64)
            for row in rows:
                try:
                    mat[int(row["layer"]), int(row["head"])] = float(row[col])
                except (ValueError, KeyError):
                    pass
            result[col] = mat
    else:
        for col in metric_cols:
            arr = np.full(n_layers, np.nan, dtype=np.float64)
            for row in rows:
                try:
                    arr[int(row["layer"])] = float(row[col])
                except (ValueError, KeyError):
                    pass
            result[col] = arr
    return result

## METRICS

COMPARE_METRICS: list[tuple[str, str]] = [
    ("ac_ie_mean", "A->C Indirect Effect"),
    ("ca_ie_mean", "C->A Indirect Effect"),
    ("bidirectional_ie_mean", "Bidrectional IE"),
    ("ac_normalised_mean", "A->C Normalised Restoration"),
    ("ca_normalised_mean", "C->A Normalised Restoration"),
    ("ac_kl_div_mean", "A->C KL Divergence"),
    ("ca_kl_div_mean", "C->A KL Divergence"),
    ("aa_ie_mean", "A->A IE"),
    ("cc_ie_mean", "C->C IE") ]


def _cmap_setup(col: str, values: np.ndarray) -> tuple[str, Any, float, float]:
    finite=values[np.isfinite(values)]

    if "kl_div" in col:
        vmin = 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        vmax = vmax or 1.0
        return "viridis", None, vmin, vmax
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = limit or 1.0

    return "RdBu_r", 0, -limit, limit

## COMPARISON PLOTS

def compare_attention_heatmaps(start_data: dict[str, np.ndarray], end_data: dict[str, np.ndarray], n_layers:int, n_heads:int, out_dir:Path) -> list[Path]:
    """Creates a three panel figure"""

    out_dir.mkdir(parents=True, exist_ok = True)
    output_paths: list[Path] = []
    layers = list(range(n_layers))
    heads = list(range(n_heads))

    for col, label in COMPARE_METRICS:
        if col not in start_data or col not in end_data:
            continue
        s = start_data[col]   # (layers, heads)
        e = end_data[col]

        diff = e - s


        all_vals = np.concatenate([s[np.isfinite(s)], e[np.isfinite(e)]])
        if all_vals.size == 0:
            continue

        cmap_se, center_se, vmin_se, vmax_se = _cmap_setup(col, all_vals)

        d_finite = diff[np.isfinite(diff)]
        lim_d = float(np.max(np.abs(d_finite))) if d_finite.size else 1.0
        lim_d = lim_d or 1.0

        fig, axes = plt.subplots(1, 3, figsize=(42, 10))
        panels = [
            (axes[0], s, f"start - {label}", cmap_se, center_se, vmin_se, vmax_se),
            (axes[1], e, f"end - {label}", cmap_se, center_se, vmin_se, vmax_se),
            (axes[2], diff, f"difference (end-start)", "RdBu_r", 0, -lim_d, lim_d)
        ]
        for ax, values, title, cmap, center, vmin, vmax in panels:
            sns.heatmap(values, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax, xticklabels=heads, yticklabels=layers, cbar_kws={"label": label})
            ax.set_title(title, fontsize=13, pad = 8)
            ax.set_xlabel("Attention head")
            ax.set_ylabel("Layer")
            ax.tick_params(axis="y", rotation=0)

        fig.suptitle(f"Position comparison - {label}", fontsize = 15, y=1.01)
        fig.tight_layout()
        out = out_dir / f"position_cmp_{col}_heatmap.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out)
        print(f"{out.name}")
    return output_paths

def compare_mlp_lines(start_data: dict[str, np.ndarray], end_data: dict[str, np.ndarray], n_layers:int, out_dir:Path) -> list[Path]:
    """Creates a two panel figure"""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths:list[Path] =[]
    layers = np.arange(n_layers)

    for col, label in COMPARE_METRICS:
        if col not in start_data or col not in end_data:
            continue
        s = start_data[col]
        e = end_data[col]
        diff = e-s

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,5))

        ax1.plot(layers, s, marker="o", linewidth = 2, markersize=5, label="Start", color=POSITION_CONFIGS["start"]["colour"])
        ax1.plot(layers, e, marker="s", linewidth = 2, markersize = 5, label = "End", color = POSITION_CONFIGS["end"]["colour"])
        ax1.axhline(0, color="black", linewidth = 0.8, linestyle = "--", alpha=0.4)
        ax1.set_title(f"{label}: Start vs End", fontsize = 12)
        ax1.set_xlabel=("Layer")
        ax1.set_ylabel=(label)
        ax1.legend()
        ax1.grid(alpha=0.25)
        ax1.set_xticks(layers)

        colours = ["firebrick" if v>0 else "steelblue" for v in diff]

        ax2.bar(layers, diff, color = colours, alpha=0.8)
        ax2.axhline(0, color = "black", linewidth = 0.8, linestyle="--", alpha=0.4)
        ax2.set_title(f"{label}: End - Start", fontsize = 12)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Δ (End - Start)")
        ax2.grid(alpha=0.25)
        ax2.set_xticks(layers)

        fig.suptitle(f"Position comparison - {label}", fontsize = 14)
        fig.tight_layout()
        out = out_dir / f"position_cmp_{col}_line.png"
        fig.savefig(out, dpi = 200, bbox_inches = "tight")
        plt.close(fig)
        output_paths.append(out)
        print(f"{out.name}")

    return output_paths

def compare_summary_heatmap(start_data: dict[str, np.ndarray], end_data: dict[str, np.ndarray], intervention:str, n_layers:int, n_heads:int, out_dir:Path) -> list[Path]:

    out_dir.mkdir(parents=True, exist_ok=True)
    available = [(col, lbl) for col, lbl in COMPARE_METRICS if col in start_data and col in end_data]
    if not available:
        return []

    n_metrics = len(available)
    layers = np.arange(n_layers)

    def _layer_mean(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.nanmean(arr, axis = 1) #(N_LAYERS,)
        return arr

    s_mat = np.full((n_metrics, n_layers), np.nan)
    e_mat = np.full ((n_metrics, n_layers), np.nan)
    diff_mat = np.full((n_metrics, n_layers), np.nan)

    ylabels = []
    for i, (col, lbl) in enumerate(available):
        s_mat[i] = _layer_mean(start_data[col])
        e_mat[i] = _layer_mean((end_data[col]))
        diff_mat[i] = e_mat[i] - s_mat[i]
        ylabels.append(lbl)
    
    fig, axes = plt.subplots(1, 3, figsize=(max(12, n_layers * 0.55) * 3, max(5, n_metrics * 0.55)))

    for ax, mat, title in [
        (axes[0], s_mat, "Start"),
        (axes[1], e_mat, "End"),
        (axes[2], diff_mat, "Difference (End-Start)")
    ]:
        finite = mat[np.isfinite(mat)]
        if title.startswith("Diff"):
            lim = float(np.max(np.abs(finite))) if finite.size else 1.0
            lim = lim or 1.0
            sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, xticklabels=layers.tolist(), yticklabels=ylabels, cbar_kws={"label": "Δ value"})
        else:
            lim = float(np.max(np.abs(finite))) if finite.size else 1.0
            lim = lim or 1.0
            sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin= -lim, vmax = lim, xticklabels=layers.tolist(), yticklabels=ylabels, cbar_kws = {"label": "value"})
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Layer")
        ax.tick_params(axis = "y", rotation=0, labelsize=9)
    fig.suptitle(f"All metrics position comparison [{intervention}]", fontsize = 14, y=1.02)
    fig.tight_layout()
    out= out_dir / f"position_cmp_summary_heatmap.png"
    fig.savefig(out, dpi = 200, bbox_inches="tight")
    plt.close(fig)
    print(f"{out.name}")
    return [out]

## COMPARISON CSV

def write_comparison_csv(start_data: dict[str, np.ndarray], end_data: dict[str, np.ndarray], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "start_mean", "start_max_abs", "end_mean", "end_median", "end_max_abs", "diff_mean", "diff_median", "diff_max_abs"])

        for col, label in COMPARE_METRICS:
            if col not in start_data or col not in end_data:
                continue
            s = start_data[col]
            e = end_data[col]
            d = e - s
            sf = s[np.isfinite(s)]
            ef = e[np.isfinite(e)]
            df = d[np.isfinite(d)]

            _m = lambda a : float(np.mean(a)) if a.size else float("nan") # noqa E731
            _med = lambda a : float(np.median(a)) if a.size else float("nan")
            _mx = lambda a : float(np.max(np.abs(a))) if a.size else float("nan")
            w.writerow([label, _m(sf), _med(sf), _mx(sf), _m(ef), _med(ef), _mx(ef), _m(df), _med(df), _mx(df)])
    print(f"Comparison CSV : {out_path}")

## MAIN

def main() -> None:
    args = parse_args()

    cfg = _base.MODEL_CONFIGS[args.model]
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])

    if args.compare_only:

        p_start = _output_prefix("start", args.intervention, args.model)
        p_end = _output_prefix("end", args.intervention, args.model)
        csv_s = p_start.with_name(p_start.name + "_global.csv")
        csv_e = p_end.with_name(p_end.name + "_global.csv")

        missing = [str(p) for p in (csv_s, csv_e) if not p.exists()]
        if missing:
            sys.exit("Cannot compare the results")
        print(f"Loading {csv_s.name} ...")

        start_data = load_global_csv(csv_s, args.intervention, n_layers, n_heads)

        print(f"Loading {csv_e.name} ...")
        end_data = load_global_csv(csv_e, args.intervention, n_layers, n_heads)

        out_dir = Path(f"outputs/position_comparison_{args.intervention}_{args.model}")
        out_dir.mkdir(parents = True, exist_ok = True)

        if args.intervention == "attention_head":
            plots = compare_attention_heatmaps(start_data, end_data, n_layers, n_heads, out_dir)
        else:
            plots = compare_mlp_lines(start_data, end_data, n_layers, out_dir)

        plots+= compare_summary_heatmap(start_data, end_data, args.intervention, n_layers, n_heads, out_dir)

        write_comparison_csv(start_data, end_data, out_dir / f"position_comparison_summary.csv")

        print(f"\n{len(plots)} plots written to {out_dir}/")
        return
    
    pos_cfg = POSITION_CONFIGS[args.position]
    dataset = pos_cfg["dataset"]

    if not dataset.exists():
        sys.exit(f"Dataset not found: {dataset}")

    pfx = args.output_prefix_override or _output_prefix(args.position, args.intervention, args.model)

    acc_path = args.accumulator or pfx.with_suffix(".npz")

    print(f"Position : {args.position} ({pos_cfg["label"]})")
    print(f"Dataset : {dataset}")
    print(f"Output : {pfx}_*")
    print()

    import argparse as _ap
    base_args = _ap.Namespace(dataset=dataset, model=args.model, intervention=args.intervention, accumulator=acc_path, output_prefix=pfx, start_row=args.start_row, max_rows = args.max_rows, patch_batch_size=args.patch_bath_size, answer_reduction = args.answer_reduction, cache_dtype=args.cache_dtype, device_map=args.device_map, min_total_effect=args.min_total_effect, dry_run=args.dry_run, plot_only = False )
    _base.run(base_args)


if __name__ == "__main__":
    main()
        