"""
Query the ADAM (Another Database of Abbreviations in MEDLINE) database for each
abbreviation in sf_list.txt and collect the first record whose Abbreviation column
exactly matches the query term.

Output columns: ABBR, Long_Form, Num_Variants, Long_Form_Score, Count
"""

import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

ADAM_URL = "https://arrowsmith.psych.uic.edu/cgi-bin/arrowsmith_uic/adam.cgi"
SF_FILE = "sf_list.txt"
OUTPUT_CSV = "adam_results.csv"
DELAY = 1.0  # seconds between requests 


def parse_count_from_text(text: str) -> int:
    """Extract the first integer in parentheses from a text string.

    Example: 'arterial blood gas (99)'  →  99
    """
    match = re.search(r'\((\d+)\)', text)
    return int(match.group(1)) if match else 0


def query_adam(abbr: str) -> dict | None:
    """
    POST a query to ADAM and return the first row whose Abbreviation column
    exactly matches `abbr` (case-sensitive).

    Column mapping:
      Cell 0 – Abbreviation
      Cell 1 – Variants (may contain multiple <a> links with counts)
      Cell 2 – Long-forms and variants (may contain multiple <a> links with counts)
      Cell 3 – Long-form Score
      Cell 4 – Count (total)

    Num_Variants = count appended to the *first* long-form link in Cell 2.
    Long_Form    = text of that first link (without the trailing parenthetical count).

    Returns a dict or None if no exact match is found.
    """
    try:
        resp = requests.post(ADAM_URL, data={"t": abbr, "Submit": "Submit"}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Request failed for {abbr}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        print(f"  [WARN]  No table found for {abbr}")
        return None

    rows = table.find_all("tr")
    # Skip the header row (index 0)
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        abbr_col = cells[0].get_text(strip=True)
        if abbr_col != abbr:
            # Not an exact match for our query term — skip
            continue

        # ── Long-form cell (cell 2) ──────────────────────────────────────────
        # May contain multiple <a> tags; we want only the first one.
        lf_cell = cells[2]
        first_lf_link = lf_cell.find("a")
        if first_lf_link:
            long_form = first_lf_link.get_text(strip=True)
        else:
            long_form = lf_cell.get_text(strip=True)

        # The count for this specific long-form variant follows the <a> tag as
        # bare text, e.g. " (4868)".  We grab it from the cell's full text by
        # looking at the text immediately after the first </a>.
        lf_cell_text = lf_cell.get_text(" ", strip=True)  # "long form (N) other form (M)…"
        num_variants = parse_count_from_text(lf_cell_text)   # first (N)

        # ── Score cell (cell 3) ──────────────────────────────────────────────
        score_text = cells[3].get_text(strip=True)
        try:
            score = float(score_text)
        except ValueError:
            score = None

        # ── Count cell (cell 4) ─────────────────────────────────────────────
        count_text = cells[4].get_text(strip=True)
        try:
            count = int(count_text)
        except ValueError:
            count = None

        return {
            "ABBR": abbr,
            "Long_Form": long_form,
            "Num_Variants": num_variants,
            "Long_Form_Score": score,
            "Count": count,
        }

    print(f"  [WARN]  No exact match found for {abbr}")
    return None


def main():
    with open(SF_FILE, "r") as f:
        abbreviations = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(abbreviations)} abbreviations from {SF_FILE}")

    results = []
    for i, abbr in enumerate(abbreviations, 1):
        print(f"[{i}/{len(abbreviations)}] Querying: {abbr}")
        record = query_adam(abbr)
        if record:
            results.append(record)
        time.sleep(DELAY)

    df = pd.DataFrame(results, columns=["ABBR", "Long_Form", "Num_Variants", "Long_Form_Score", "Count"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(df)} records saved to {OUTPUT_CSV}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
