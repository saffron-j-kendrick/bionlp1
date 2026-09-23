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

