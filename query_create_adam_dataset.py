## IMPORTS

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

## CONSTANTS

ADAM_URL = "https://arrowsmith.psych.uic.edu/cgi-bin/arrowsmith_uic/adam.cgi"
SF_FILE = "sf_list.txt"
OUTPUT_CSV = "adam_dataset_results.csv"
N_SENTENCES = 20
DELAY = 1.2  # seconds between requests 


## FUNCTIONS

def parse_count_from_text(text: str) -> int:
    """Extract the first integer in parentheses from a text string.

    Example: 'arterial blood gas (99)'  →  99
    """
    match = re.search(r'\((\d+)\)', text)
    return int(match.group(1)) if match else 0



def query_adam(abbr: str) -> dict | None:
    f"""
    In this function, we are querying the ADAM database for primary and altnerative long form expressions for a given abbreviation.

    1. Query the database with the given url and abbreviation.
    2. For the abbreviation, get all the row indices from cell 0 ["Abbreviation"] that exactly match the given abbreviation.
    3. Take the first index and get the primary long form expression from cell 2 ["Long-forms and variants"], get the number of variants in cell 1 ["Variants"]. Also get the Long-form Score from cell 3.
    4. Check that the number of variants for the primary long form expression is greater than 20 AND the Long-form Score is greater than 0.5 -> if not then skip this abbreviation.
    5. If the primary long form expression is valid then we want to get the alternative long form expression.
    6. Iterate through the remaining row indices from step 2. For each index, check that the long form expression in cell 2 ["Long-forms and variants"] does not overlap with the primary long form expression (check the first token). If it overlaps, move to the next index. 
    7. For the valid alternative long form expression, get the number of variants in cell 1 ["Variants"] that follow the expression name (is given in brackets). Also get the Long-form Score from cell 3. 
    8. Check that the number of variants for the alternative long form expression is greater than 20 AND the Long-form Score is greater than 0.5 -> if not then skip this row and check the next index.
    9. If there are no valid alternative long form expressions, return None in the table.
    10. Create a dictionary with the primary and alternative long form expressions, their scores, and the number of variants.

    return dictionary in a pandas dataframe with the columns: ABBR, Long_Form, Primary_Num_Variants, Alternative_Long_Form, Alternative_Num_Variants, Primary_Long_Form_Score, Alternative_Long_Form_Score and save as csv.
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
    matching_rows = []

    # Phase 1: collect all rows where cell 0 exactly matches the abbreviation.
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        if cells[0].get_text(strip=True) == abbr:
            matching_rows.append(row)

    # Need at least 2 matching rows (primary + at least one potential alternative).
    if len(matching_rows) < 2:
        print(f"  [WARN]  Fewer than 2 rows found for {abbr}. Skipping...")
        return None

    # Phase 2: extract the primary long form from the first matching row.
    primary_cells = matching_rows[0].find_all("td")
    primary_lf_cell = primary_cells[2]
    first_link = primary_lf_cell.find("a")
    if first_link:
        primary_long_form = first_link.get_text(strip=True)
    else:
        primary_long_form = primary_lf_cell.get_text(strip=True)

    primary_num_variants = parse_count_from_text(primary_lf_cell.get_text(strip=True))
    primary_long_form_score = float(primary_cells[3].get_text(strip=True))

    print(f"  [INFO]  Primary long form for {abbr}: {primary_long_form} | variants={primary_num_variants} | score={primary_long_form_score}")

    if primary_num_variants < 20 or primary_long_form_score < 0.5:
        print(f"  [WARN]  Primary long form for {abbr} is not valid. Skipping...")
        return None

    # Phase 3: find a valid alternative from the remaining matching rows.
    for alt_row in matching_rows[1:]:
        alt_cells = alt_row.find_all("td")
        if len(alt_cells) < 5:
            continue
        alt_lf_text = alt_cells[2].get_text(strip=True)
        if alt_lf_text.split()[0] == primary_long_form.split()[0]:
            continue
        alt_first_link = alt_cells[2].find("a")
        alternative_long_form = alt_first_link.get_text(strip=True) if alt_first_link else alt_lf_text
        alternative_num_variants = parse_count_from_text(alt_lf_text)
        alternative_long_form_score = float(alt_cells[3].get_text(strip=True))
        print(f"  [INFO]  Alternative long form for {abbr}: {alternative_long_form} | variants={alternative_num_variants} | score={alternative_long_form_score}")
        if alternative_num_variants < 20 or alternative_long_form_score < 0.2:
            print(f"  [WARN]  Alternative long form for {abbr} is not valid. Trying next...")
            continue
        return {
            "ABBR": abbr,
            "Primary_Long_Form": primary_long_form,
            "Primary_Num_Variants": primary_num_variants,
            "Alternative_Long_Form": alternative_long_form,
            "Alternative_Num_Variants": alternative_num_variants,
            "Primary_Long_Form_Score": primary_long_form_score,
            "Alternative_Long_Form_Score": alternative_long_form_score,
        }

    print(f"  [WARN]  No valid alternative long form found for {abbr}. Skipping...")
    return None


def main():
    with open(SF_FILE, "r") as f:
        abbreviations = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(abbreviations)} abbreviations from {SF_FILE}")
    results = []
    for i, abbreviation in tqdm(enumerate(abbreviations, 1), total=len(abbreviations), desc="Querying ADAM database"):
        print(f"[{i}/{len(abbreviations)}] Querying: {abbreviation}")
        result = query_adam(abbreviation)
        if result:
            results.append(result)
        time.sleep(DELAY)
    df = pd.DataFrame(results, columns=["ABBR", "Primary_Long_Form", "Primary_Num_Variants", "Alternative_Long_Form", "Alternative_Num_Variants", "Primary_Long_Form_Score", "Alternative_Long_Form_Score"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! {len(df)} records saved to {OUTPUT_CSV}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()