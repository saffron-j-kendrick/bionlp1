"""
For each abbreviation in adam_results.csv where Num_Variants >= 10:
  1. Re-query the ADAM main page to extract the pid and lid from the first
     exact-match row (these are embedded in the hyperlinks).
  2. Fetch the corresponding sentences page.
  3. Extract the first 10 sentences (with their PubMed IDs).

Output dataframe columns:
  ABBR | Long_Form | Sentence_Num | PMID | Sentence

Saved to adam_sentences.csv
"""

import csv
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

ADAM_URL  = "https://arrowsmith.psych.uic.edu/cgi-bin/arrowsmith_uic/adam.cgi"
INPUT_CSV = "adam_results.csv"
OUTPUT_CSV = "adam_sentences.csv"
N_SENTENCES = 10
DELAY = 1.2   # seconds between requests


# ── helpers ──────────────────────────────────────────────────────────────────

def get_pid_lid(abbr: str) -> tuple[str, str] | tuple[None, None]:
    """
    Query the ADAM main page for `abbr` and return (pid, lid) from the
    first row whose Abbreviation cell exactly matches `abbr`.

    The long-form link looks like:
      <a href="adam.cgi?pid=P001147&lid=S069418" ...>arterial blood gas</a>
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
        if len(cells) < 5:
            continue
        if cells[0].get_text(strip=True) != abbr:
            continue

        # Long-form cell (index 2) – first <a> holds pid and lid
        lf_link = cells[2].find("a")
        if not lf_link:
            continue

        href = lf_link.get("href", "")
        pid_match = re.search(r'pid=([^&]+)', href)
        lid_match = re.search(r'lid=([^&]+)', href)

        if pid_match and lid_match:
            return pid_match.group(1), lid_match.group(1)

    return None, None


def fetch_sentences(pid: str, lid: str, n: int = 10) -> list[dict]:
    """
    Fetch the sentences page for the given pid/lid and return up to `n`
    records, each a dict with keys: sentence_num, pmid, sentence.
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

    records = []
    for row in table.find_all("tr")[1:]:          # skip header
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        sentence_num_text = cells[0].get_text(strip=True)
        try:
            sentence_num = int(sentence_num_text)
        except ValueError:
            continue

        pmid_link = cells[1].find("a")
        pmid = pmid_link.get_text(strip=True) if pmid_link else cells[1].get_text(strip=True)

        sentence = cells[2].get_text(" ", strip=True)

        records.append({
            "sentence_num": sentence_num,
            "pmid": pmid,
            "sentence": sentence,
        })

        if len(records) >= n:
            break

    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # keep_default_na=False prevents "NA" from being parsed as NaN
    df_meta = pd.read_csv(INPUT_CSV, keep_default_na=False)
    # Only process abbreviations with enough variants
    eligible = df_meta[df_meta["Num_Variants"].astype(int) >= N_SENTENCES].copy()
    print(f"Eligible abbreviations (Num_Variants >= {N_SENTENCES}): {len(eligible)} / {len(df_meta)}")

    # Resume support: skip abbreviations already present in the output file
    already_done: set[str] = set()
    if pd.io.common.file_exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
        already_done = set(df_existing["ABBR"].unique())
        print(f"Resuming — {len(already_done)} abbreviations already in {OUTPUT_CSV}")

    total = len(eligible)
    cols = ["ABBR", "Long_Form", "Sentence_Num", "PMID", "Sentence"]

    # Open output file in append mode; write header only if starting fresh
    write_header = not pd.io.common.file_exists(OUTPUT_CSV)
    out_fh = open(OUTPUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=cols)
    if write_header:
        writer.writeheader()

    processed = 0
    for i, (_, row) in enumerate(eligible.iterrows(), 1):
        abbr = str(row["ABBR"]).strip()
        lf   = row["Long_Form"]

        if abbr in already_done:
            print(f"[{i}/{total}] {abbr}  —  already done, skipping")
            continue

        print(f"[{i}/{total}] {abbr}  —  getting pid/lid …", end=" ", flush=True)
        pid, lid = get_pid_lid(abbr)
        time.sleep(DELAY)

        if pid is None:
            print("SKIP (no pid/lid found)")
            continue

        print(f"pid={pid} lid={lid}  —  fetching sentences …", end=" ", flush=True)
        sentences = fetch_sentences(pid, lid, n=N_SENTENCES)
        time.sleep(DELAY)

        if not sentences:
            print("SKIP (no sentences)")
            continue

        for s in sentences:
            writer.writerow({
                "ABBR":         abbr,
                "Long_Form":    lf,
                "Sentence_Num": s["sentence_num"],
                "PMID":         s["pmid"],
                "Sentence":     s["sentence"],
            })
        out_fh.flush()  # ensure rows are written to disk after each abbreviation

        print(f"got {len(sentences)} sentences")
        processed += 1

    out_fh.close()

    df_out = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
    print(f"\nDone!  {len(df_out)} sentence rows in {OUTPUT_CSV}")
    print(f"Abbreviations with sentences: {df_out['ABBR'].nunique()}")
    print(df_out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
