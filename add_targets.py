## IMPORTS

import re
import pandas as pd

## CONSTANT

INPUT_SENTENCES_CSV = "adam_sentences_filtered.csv"
INPUT_DATASET_CSV   = "adam_dataset_results.csv"
OUTPUT_CSV          = "adam_sentences_filtered_with_targets.csv"


## FUNCTIONS

def _alpha_words(text: str) -> list[str]:
    """Return all alphabetic tokens (lower-cased) found in text."""
    return [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)]


def _connector_word(sentence: str) -> str:
    """
    Return the first of 'and', 'therefore', 'but' that does not already appear
    as a whole word in sentence.
    """
    present = set(_alpha_words(sentence))
    for candidate in ("and", "therefore", "but"):
        if candidate not in present:
            return candidate
    return "therefore"  # extremely unlikely that all three are present


def find_target(sentence_a: str, anchor: str) -> tuple[str, bool]:
    """
    Find the target word and whether a connector word needs to be appended.
    Returns:
    - target_word
    - needs_connector=True means the caller should append target_word to the
        sentence group before saving.
    """
    match = re.search(re.escape(anchor), sentence_a, re.IGNORECASE)
    if not match:
        # Anchor not found — fall back to a connector word
        return _connector_word(sentence_a), True

    # Words that appear strictly BEFORE the anchor
    before = set(_alpha_words(sentence_a[: match.start()]))

    # Words that are PART OF the anchor (to avoid matching them in the anchor itself)
    anchor_words = set(_alpha_words(anchor))

    # Candidate words strictly AFTER the anchor
    after_text = sentence_a[match.end():]
    candidates = re.findall(r'\b[a-zA-Z]+\b', after_text)

    for word in candidates:
        word_lower = word.lower()

        # Skip stop words that are part of the anchor text itself
        if word_lower in anchor_words:
            continue

        if word_lower not in before:
            # This occurrence is the first instance in the sentence → valid target
            return word, False
        # else: word appeared before the anchor → not the first instance → try next

    # Nothing valid found — we need to add a connector
    return _connector_word(sentence_a), True


def process_row(row: pd.Series, primary_lf: str, alt_lf: str) -> pd.Series:
    row = row.copy()

    # target for sentence A, B, C
    target, needs_connector = find_target(str(row["Sentence_A"]), primary_lf)
    if needs_connector:
        row["Sentence_A"] = str(row["Sentence_A"]).rstrip().rstrip(".").rstrip() + " " + target + "."
        row["Sentence_B"] = str(row["Sentence_B"]).rstrip().rstrip(".").rstrip() + " " + target + "."
        row["Sentence_C"] = str(row["Sentence_C"]).rstrip().rstrip(".").rstrip() + " " + target + "."
    row["target"] = target

    # target prime for sentence A', B', C'
    target_prime, needs_connector_prime = find_target(str(row["Sentence_A_Prime"]), alt_lf)
    if needs_connector_prime:
        row["Sentence_A_Prime"] = str(row["Sentence_A_Prime"]).rstrip().rstrip(".").rstrip() + " " + target_prime + "."
        row["Sentence_B_Prime"] = str(row["Sentence_B_Prime"]).rstrip().rstrip(".").rstrip() + " " + target_prime + "."
        row["Sentence_C_Prime"] = str(row["Sentence_C_Prime"]).rstrip().rstrip(".").rstrip() + " " + target_prime + "."
    row["target_prime"] = target_prime

    return row

## MAIN

def main():
    df_sentences = pd.read_csv(INPUT_SENTENCES_CSV, keep_default_na=False)
    df_dataset   = pd.read_csv(INPUT_DATASET_CSV,   keep_default_na=False)

    print(f"Loaded {len(df_sentences)} rows from {INPUT_SENTENCES_CSV}")

    # Build ABBR → (primary_lf, alt_lf) lookup from the dataset CSV
    lf_map: dict[str, tuple[str, str]] = {
        str(r["ABBR"]).strip(): (
            str(r["Primary_Long_Form"]).strip(),
            str(r["Alternative_Long_Form"]).strip(),
        )
        for _, r in df_dataset.iterrows()
    }

    updated_rows = []
    skipped = 0
    for _, row in df_sentences.iterrows():
        abbr = str(row["ABBR"]).strip()
        if abbr not in lf_map:
            print(f"  [WARN] '{abbr}' not in dataset CSV — skipping target assignment")
            row = row.copy()
            row["target"]       = ""
            row["target_prime"] = ""
            updated_rows.append(row)
            skipped += 1
            continue

        primary_lf, alt_lf = lf_map[abbr]
        updated_rows.append(process_row(row, primary_lf, alt_lf))

    df_out = pd.DataFrame(updated_rows)

    # Ensure target columns appear at the end
    cols = [c for c in df_out.columns if c not in ("target", "target_prime")]
    df_out = df_out[cols + ["target", "target_prime"]]

    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone!  {len(df_out)} rows written to {OUTPUT_CSV}")
    if skipped:
        print(f"  ({skipped} rows skipped — ABBR not found in dataset CSV)")
    print()
    print(df_out[["ABBR", "target", "target_prime"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
