#!/usr/bin/env python3
"""Fill expression columns in the ADAM activation-patching dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "ABBR",
    "Sentence_A",
    "Sentence_B",
    "Sentence_C",
    "Primary_Expression",
    "Alternative_Expression",
}
MAPPING_COLUMNS = {"ABBR", "Primary_Long_Form", "Alternative_Long_Form"}


def load_mapping(mapping_path: Path) -> dict[str, tuple[str, str]]:
    """Load canonical primary and alternative expressions by abbreviation."""
    with mapping_path.open(newline="", encoding="utf-8-sig") as mapping_file:
        reader = csv.DictReader(mapping_file)
        if reader.fieldnames is None:
            raise ValueError(f"{mapping_path} has no header row")

        missing_columns = MAPPING_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Mapping file is missing required columns: {missing}")

        mapping: dict[str, tuple[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            abbreviation = row["ABBR"].strip()
            expressions = (
                row["Primary_Long_Form"].strip(),
                row["Alternative_Long_Form"].strip(),
            )
            if not abbreviation or not all(expressions):
                raise ValueError(
                    f"Incomplete mapping at {mapping_path}:{line_number}"
                )
            if abbreviation in mapping and mapping[abbreviation] != expressions:
                raise ValueError(f"Conflicting mappings for {abbreviation}")
            mapping[abbreviation] = expressions

    return mapping


def fill_expressions(
    input_path: Path,
    mapping_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")

        rows = list(reader)
        fieldnames = reader.fieldnames

    mapping = load_mapping(mapping_path)
    missing_abbreviations = sorted(
        {row["ABBR"].strip() for row in rows}.difference(mapping)
    )
    if missing_abbreviations:
        raise ValueError(
            "No mapping found for: " + ", ".join(missing_abbreviations)
        )

    primary_filled = 0
    alternative_filled = 0
    for row in rows:
        abbreviation = row["ABBR"].strip()
        primary, alternative = mapping[abbreviation]

        if overwrite or not row["Primary_Expression"].strip():
            row["Primary_Expression"] = primary
            primary_filled += 1

        if overwrite or not row["Alternative_Expression"].strip():
            row["Alternative_Expression"] = alternative
            alternative_filled += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return primary_filled, alternative_filled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill Primary_Expression and Alternative_Expression using the "
            "canonical long forms in adam_dataset_results.csv."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("adam_dataset_patching.csv"),
        help="input CSV (default: adam_dataset_patching.csv)",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=Path,
        help=(
            "CSV containing ABBR, Primary_Long_Form, and Alternative_Long_Form "
            "(default: adam_dataset_results.csv beside the input)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output CSV (default: <input stem>_filled.csv)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input CSV instead of creating a filled copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace expression values that are already populated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_place and args.output:
        raise SystemExit("--in-place and --output cannot be used together")

    output_path = (
        args.input
        if args.in_place
        else args.output
        or args.input.with_name(f"{args.input.stem}_filled{args.input.suffix}")
    )
    mapping_path = args.mapping or args.input.with_name("adam_dataset_results.csv")

    primary_count, alternative_count = fill_expressions(
        args.input, mapping_path, output_path, overwrite=args.overwrite
    )
    print(f"Wrote {output_path}")
    print(f"Filled {primary_count} primary expressions")
    print(f"Filled {alternative_count} alternative expressions")


if __name__ == "__main__":
    main()
