import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import json
import re
from pathlib import Path
import os


OUTPUT_FILE = Path("filtered_sentences.jsonl")
ACRONYM_PATTERN = r"[\(\[]\W*([A-Z\-]{2,})s?\W*[\)\]]"


def extract_acronyms(sentence: str) -> set[str]:
    """Find all acronyms in a sentence and return them as a set within first captured group"""
    stripped_acronyms = [match.group(1) for match in re.finditer(ACRONYM_PATTERN, sentence)]
    return set(stripped_acronyms)


def build_regex_pattern(abbreviation: str) -> str:
    return r"\(\W*{abbreviation}s?\W*\)".format(abbreviation=abbreviation)


def has_abbreviations(sentence, abbreviations):
    """
    Counts the number of abbreviations in a sentence.
    """
    for abbr in abbreviations:
        pattern = build_regex_pattern(abbr)
        logging.debug(f"Checking for abbreviation {abbr} in sentence {sentence} with pattern {pattern}")
        matches = re.findall(pattern, sentence)
        if len(matches) > 0:
            return True
    return False

@dataclass
class SentenceWithTargetAbbreviation:
    sentence: str
    target_abbreviation: str


def filter_sentences(sentence_dataset: list[SentenceWithTargetAbbreviation], abbreviations: set[str]) -> list[SentenceWithTargetAbbreviation]:
    """
    Iterates through the list of sentences and returns a list of sentences that only contain the target abbreviation.
    If a sentence contains multiple abbreviations, it is not added to the list.
    """
    valid_sentences = []
    for sentence_row in sentence_dataset:
        sentence = sentence_row.sentence
        target_abbreviation = sentence_row.target_abbreviation
        # target_pattern = build_regex_pattern(target_abbreviation)
        # if not re.search(target_pattern, sentence):
        #     logging.debug(f"Sentence {sentence_row.sentence} does not contain the target abbreviation {sentence_row.target_abbreviation}")
        #     continue

        # non_pattern_abbreviations = abbreviations - {target_abbreviation}
        # if has_abbreviations(sentence, non_pattern_abbreviations):
        #     logging.debug(f"Sentence {sentence_row.sentence} contains multiple abbreviations {non_pattern_abbreviations}")
        #     continue

        acronyms = extract_acronyms(sentence)
        if len(acronyms) > 1:
            logging.debug(f"Sentence {sentence_row.sentence} contains multiple acronyms {acronyms}")
            continue

        if target_abbreviation not in acronyms:
            logging.debug(f"Sentence {sentence_row.sentence} does not contain the target abbreviation {sentence_row.target_abbreviation}")
            continue

        logging.debug(f"Sentence {sentence_row.sentence} is valid")
        valid_sentences.append(sentence_row)
    logging.info(f"Found {len(valid_sentences)} valid sentences")
    return valid_sentences


# create a dataframe from the created jsonl file 

def create_dataframe(jsonl_file: Path) -> pd.DataFrame:
    dataframe = []
    with open(jsonl_file, "r") as f:
        for line in f:
            sentence_row = json.loads(line)
            dataframe.append(SentenceWithTargetAbbreviation(**sentence_row))
    SentenceWithTargetAbbreviationDataFrame = pd.DataFrame([sentence_row.__dict__ for sentence_row in dataframe])
    # save
    SentenceWithTargetAbbreviationDataFrame.to_csv("sentence_with_target_abbreviation.csv", index=False)
    return SentenceWithTargetAbbreviationDataFrame




def create_triple_sentence_dataframe(sentence_with_target_abbreviation_dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = []
    for index, row in sentence_with_target_abbreviation_dataframe.iterrows():
        sentence = row["sentence"]
        target_abbreviation = row["target_abbreviation"]
        dataframe.append([target_abbreviation, sentence, sentence, sentence])
    TripleSentenceDataFrame = pd.DataFrame(dataframe)
    # rename columns
    TripleSentenceDataFrame.columns = ["abbr", "sentence_a", "sentence_b", "sentence_c"]
    # save
    TripleSentenceDataFrame.to_csv("triple_sentence.csv", index=False)
    return TripleSentenceDataFrame

def keep_ten_sentences_per_abbr(triple_sentence_dataframe: pd.DataFrame) -> pd.DataFrame:
    ''' for each abbreviation, keep the first 10 sentences, so if an abbreviation has more then just ignore the extra sentences'''
    new_dataframe = pd.DataFrame()
    for abbr in triple_sentence_dataframe["abbr"].unique():
        abbr_sentences = triple_sentence_dataframe[triple_sentence_dataframe["abbr"] == abbr]
        print(f"Abbreviation {abbr} has {len(abbr_sentences)} sentences")
        if len(abbr_sentences) >= 10:
            print(f"Keeping 10 sentences for {abbr}")
            abbr_sentences = abbr_sentences.iloc[:10]
        else:
            print(f"Keeping all sentences for {abbr}")
            abbr_sentences = abbr_sentences
        new_dataframe = pd.concat([new_dataframe, abbr_sentences])
    return new_dataframe


def main():
    # # load the csv of all sentences

    # full_sentences = pd.read_csv("adam_sentences.csv")
    # raw_sentences = full_sentences["Sentence"].tolist()
    # raw_target_abbreviations = full_sentences["ABBR"].tolist()
    # sentence_dataset = [SentenceWithTargetAbbreviation(sentence, abbreviation) for sentence, abbreviation in zip(raw_sentences, raw_target_abbreviations)]

    # abbreviations = pd.read_csv("sf_list_full.txt", header=None).iloc[:, 0].tolist()
    # abbreviations = set(abbreviations)

    # valid_sentences = filter_sentences(sentence_dataset, abbreviations)
    # # save sentences list to json file

    # if OUTPUT_FILE.exists():
    #     os.remove(OUTPUT_FILE)

    # logging.info(f"Saving valid sentences to {OUTPUT_FILE}")

    # with open(OUTPUT_FILE, "a") as f:
    #     for sentence_row in valid_sentences:
    #         json_line = json.dumps(sentence_row.__dict__) + "\n"
    #         # Append the json line to the file
    #         f.write(json_line)
    #         f.flush()
    sentence_with_target_abbreviation_dataframe = create_dataframe(OUTPUT_FILE)
    triple_sentence_dataframe = create_triple_sentence_dataframe(sentence_with_target_abbreviation_dataframe)
    print(triple_sentence_dataframe.head())
    print(f"Original dataframe has {len(triple_sentence_dataframe)} rows")
    # count the number of rows per unique abbr and return the lowest count i.e. AAA has 14 rows etc and the abbr with the lowest count
    abbr_counts = triple_sentence_dataframe["abbr"].value_counts()
    lowest_count = abbr_counts.min()
    lowest_abbr = abbr_counts.idxmin()
    print(f"The lowest count is {lowest_count} for {lowest_abbr}")
    # remove any rows where the count for the abbr is less than 10
    triple_sentence_dataframe = triple_sentence_dataframe[triple_sentence_dataframe["abbr"].isin(abbr_counts[abbr_counts >= 10].index)]
    triple_sentence_dataframe = keep_ten_sentences_per_abbr(triple_sentence_dataframe)
    print(f"Removed rows where the count for the abbr is less than 10")
    print(f"Kept 10 sentences per abbreviation")
    print(f"New dataframe has {len(triple_sentence_dataframe)} rows")
    #print the max
    print(f"The max count is {triple_sentence_dataframe['abbr'].value_counts().max()}")
    # print the min
    print(f"The min count is {triple_sentence_dataframe['abbr'].value_counts().min()}")
    # print the mean
    print(f"The mean count is {triple_sentence_dataframe['abbr'].value_counts().mean()}")
    # print the median
    print(f"The median count is {triple_sentence_dataframe['abbr'].value_counts().median()}")
    # print the mode
    print(f"The mode count is {triple_sentence_dataframe['abbr'].value_counts().mode()}")
    
    # save the dataframe
    triple_sentence_dataframe.to_csv("triple_sentence_filtered.csv", index=False)


if __name__ == "__main__":
    main()