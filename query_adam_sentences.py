"""
For each abbreviation in adam_dataset_results.csv:
  1. Query ADAM for 20 sentences containing the primary long form.
  2. Filter to sentences that contain only the target abbreviation; keep first 5.
     Skip the abbreviation if fewer than 5 valid sentences are found.
  3. For each of the 5 sentences build Sentence A / B / C.
  4. Repeat steps 1-2 for the alternative long form to get 5 more sentences.
  5. For each of those 5 sentences build Sentence A' / B' / C'.
  6. Write one output row per sentence pair (5 rows per abbreviation).

Output columns:
  ABBR | Sentence_A | Sentence_B | Sentence_C |
  Sentence_A_Prime | Sentence_B_Prime | Sentence_C_Prime

Saved to adam_sentences_filtered.csv
"""

import csv
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

ADAM_URL   = "https://arrowsmith.psych.uic.edu/cgi-bin/arrowsmith_uic/adam.cgi"
INPUT_CSV  = "adam_dataset_results.csv"
OUTPUT_CSV = "adam_sentences_filtered.csv"
N_FETCH    = 20   # sentences to request from ADAM per long form
N_KEEP     = 5    # sentences to keep after filtering
DELAY      = 1.2  # seconds between HTTP requests



ACRONYM_PATTERN = r"[\(\[]\W*([A-Z\-]{2,})s?\W*[\)\]]"

def extract_acronyms(sentence: str) -> set[str]:
    """Return the set of all bracketed acronyms found in a sentence."""
    return {m.group(1) for m in re.finditer(ACRONYM_PATTERN, sentence)}


def get_pid_lid(abbr: str, long_form: str) -> tuple[str, str] | tuple[None, None]:
    """
    POST to ADAM for `abbr` and return (pid, lid) from the first row whose
    cell-0 exactly matches `abbr` and whose first <a> link text in cell 2
    matches `long_form` (case-insensitive).
    """
    try:
        resp = requests.post(ADAM_URL, data={"t": abbr, "Submit": "Submit"}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Main query failed for {abbr}: {e}")
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        return None, None

    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 5 or cells[0].get_text(strip=True) != abbr:
            continue
        lf_link = cells[2].find("a")
        if not lf_link:
            continue
        if lf_link.get_text(strip=True).lower() == long_form.lower():
            href = lf_link.get("href", "")
            pid_m = re.search(r'pid=([^&]+)', href)
            lid_m = re.search(r'lid=([^&]+)', href)
            if pid_m and lid_m:
                return pid_m.group(1), lid_m.group(1)

    return None, None


def fetch_sentences(pid: str, lid: str, n: int = N_FETCH) -> list[str]:
    """
    Fetch the sentences page for the given pid/lid and return up to `n`
    sentence strings.
    """
    url = f"{ADAM_URL}?pid={pid}&lid={lid}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Sentences fetch failed ({pid}/{lid}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    sentences = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        try:
            int(cells[0].get_text(strip=True))
        except ValueError:
            continue
        sentences.append(cells[2].get_text(" ", strip=True))
        if len(sentences) >= n:
            break

    return sentences


def filter_sentences(sentences: list[str], abbr: str, keep: int = N_KEEP) -> list[str]:
    """
    Keep only sentences that contain exactly the target abbreviation and no
    other bracketed acronyms.  Return the first `keep` such sentences, or an
    empty list if fewer than `keep` are found.
    """
    valid = []
    for s in sentences:
        acronyms = extract_acronyms(s)
        if acronyms == {abbr}:
            valid.append(s)
        if len(valid) == keep:
            break
    return valid if len(valid) == keep else []


# ── Sentence transformations ──────────────────────────────────────────────────

def _remove_abbr_bracket(sentence: str, abbr: str) -> str:
    """Remove '(ABBR)' and any surrounding whitespace from the sentence."""
    pattern = r'\s*[\(\[]\W*' + re.escape(abbr) + r's?\W*[\)\]]'
    return re.sub(pattern, '', sentence).strip()


def _unbracket_abbr(sentence: str, abbr: str) -> str:
    """Turn '(ABBR)' into 'ABBR' — strip the brackets but keep the text."""
    pattern = r'[\(\[]\W*(' + re.escape(abbr) + r's?)\W*[\)\]]'
    return re.sub(pattern, r'\1', sentence)


def _remove_long_form(sentence: str, long_form: str) -> str:
    """Remove the long form text (case-insensitive) and collapse extra spaces."""
    result = re.sub(re.escape(long_form), '', sentence, flags=re.IGNORECASE)
    return re.sub(r'  +', ' ', result).strip()


def _replace_long_form(sentence: str, source: str, target: str) -> str:
    """Replace source long form with target long form (case-insensitive)."""
    result = re.sub(re.escape(source), target, sentence, flags=re.IGNORECASE)
    return re.sub(r'  +', ' ', result).strip()


def make_sentence_a(sentence: str, abbr: str) -> str:
    """Remove '(ABBR)' and its brackets."""
    return _remove_abbr_bracket(sentence, abbr)


def make_sentence_b(sentence: str, abbr: str, primary_lf: str) -> str:
    """Remove the primary long form; strip brackets from the abbreviation."""
    s = _remove_long_form(sentence, primary_lf)
    return _unbracket_abbr(s, abbr)


def make_sentence_c(sentence: str, abbr: str, primary_lf: str, alt_lf: str) -> str:
    """Replace primary long form with alternative long form; remove '(ABBR)'."""
    s = _replace_long_form(sentence, primary_lf, alt_lf)
    return _remove_abbr_bracket(s, abbr)


def make_sentence_a_prime(sentence: str, abbr: str) -> str:
    """Remove '(ABBR)' and its brackets (applied to alternative sentences)."""
    return _remove_abbr_bracket(sentence, abbr)


def make_sentence_b_prime(sentence: str, abbr: str, alt_lf: str) -> str:
    """Remove the alternative long form; strip brackets from the abbreviation."""
    s = _remove_long_form(sentence, alt_lf)
    return _unbracket_abbr(s, abbr)


def make_sentence_c_prime(sentence: str, abbr: str, alt_lf: str, primary_lf: str) -> str:
    """Replace alternative long form with primary long form; remove '(ABBR)'."""
    s = _replace_long_form(sentence, alt_lf, primary_lf)
    return _remove_abbr_bracket(s, abbr)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df_meta = pd.read_csv(INPUT_CSV, keep_default_na=False)
    print(f"Loaded {len(df_meta)} abbreviations from {INPUT_CSV}")

    # Resume support: skip abbreviations already written to the output file
    already_done: set[str] = set()
    if pd.io.common.file_exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
        already_done = set(df_existing["ABBR"].unique())
        print(f"Resuming — {len(already_done)} abbreviations already in {OUTPUT_CSV}")

    cols = [
        "ABBR",
        "Sentence_A", "Sentence_B", "Sentence_C",
        "Sentence_A_Prime", "Sentence_B_Prime", "Sentence_C_Prime",
    ]
    write_header = not pd.io.common.file_exists(OUTPUT_CSV)
    out_fh = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=cols)
    if write_header:
        writer.writeheader()

    total = len(df_meta)
    for i, (_, row) in enumerate(df_meta.iterrows(), 1):
        abbr       = str(row["ABBR"]).strip()
        primary_lf = str(row["Primary_Long_Form"]).strip()
        alt_lf     = str(row["Alternative_Long_Form"]).strip()

        if abbr in already_done:
            print(f"[{i}/{total}] {abbr}  — already done, skipping")
            continue

        print(f"[{i}/{total}] {abbr}  — querying primary: '{primary_lf}'")

        # ── Primary long form sentences ───────────────────────────────────────
        pid, lid = get_pid_lid(abbr, primary_lf)
        time.sleep(DELAY)
        if pid is None:
            print(f"  [SKIP] No pid/lid found for primary long form of {abbr}")
            continue

        raw_primary = fetch_sentences(pid, lid, n=N_FETCH)
        time.sleep(DELAY)
        primary_sents = filter_sentences(raw_primary, abbr, keep=N_KEEP)
        if not primary_sents:
            print(f"  [SKIP] Fewer than {N_KEEP} valid primary sentences for {abbr}")
            continue

        # ── Alternative long form sentences ───────────────────────────────────
        print(f"  — querying alternative: '{alt_lf}'")
        pid_alt, lid_alt = get_pid_lid(abbr, alt_lf)
        time.sleep(DELAY)
        if pid_alt is None:
            print(f"  [SKIP] No pid/lid found for alternative long form of {abbr}")
            continue

        raw_alt = fetch_sentences(pid_alt, lid_alt, n=N_FETCH)
        time.sleep(DELAY)
        alt_sents = filter_sentences(raw_alt, abbr, keep=N_KEEP)
        if not alt_sents:
            print(f"  [SKIP] Fewer than {N_KEEP} valid alternative sentences for {abbr}")
            continue

        # ── Build one row per sentence pair ───────────────────────────────────
        for p_sent, a_sent in zip(primary_sents, alt_sents):
            writer.writerow({
                "ABBR":             abbr,
                "Sentence_A":       make_sentence_a(p_sent, abbr),
                "Sentence_B":       make_sentence_b(p_sent, abbr, primary_lf),
                "Sentence_C":       make_sentence_c(p_sent, abbr, primary_lf, alt_lf),
                "Sentence_A_Prime": make_sentence_a_prime(a_sent, abbr),
                "Sentence_B_Prime": make_sentence_b_prime(a_sent, abbr, alt_lf),
                "Sentence_C_Prime": make_sentence_c_prime(a_sent, abbr, alt_lf, primary_lf),
            })
        out_fh.flush()
        print(f"  — wrote {N_KEEP} rows for {abbr}")

    out_fh.close()

    df_out = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
    print(f"\nDone!  {len(df_out)} rows in {OUTPUT_CSV}")
    print(f"Abbreviations with sentences: {df_out['ABBR'].nunique()}")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
