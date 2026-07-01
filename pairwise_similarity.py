import os
import numpy as np
import torch
import tqdm
import pathlib
import argparse
import inflect
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity 

adam_sentences = pd.read_csv('adam_sentences.csv', keep_default_na=False)
abbrvs = adam_sentences["ABBR"].unique()

sentence_1 = adam_sentences.iloc[2]["Sentence"]
sentence_2 = adam_sentences.iloc[3]["Sentence"]

def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def remove_abbreviations(abbr, sentence):
    sentence = re.sub(r'\([^)]*\)', '', sentence)

    if abbr in sentence:
        sentence = re.sub(abbr, '', sentence)
    return sentence


def tokenize_sentence(sentence):
    return word_tokenize(sentence)


# sentence_1 = remove_punctuation(sentence_1)
# sentence_1 = remove_abbreviations(sentence_1)
# print(sentence_1)
# sentence_2 = remove_punctuation(sentence_2)
# sentence_2 = remove_abbreviations(sentence_2)
# print(sentence_2)
# sentence_3 = "Guidelines from the AAA recommend routine hearing screenings for adults over the age of 50."
# sentence_3 = remove_punctuation(sentence_3)
# print(sentence_3)

abbr_1 = "AAA"
sentence_1 = "OBJECTIVE: abdominal aortic aneurysm represents a chronic degenerative condition associated with atherosclerosis."
sentence_2 = "METHODS: The Canadian Institute for Health Information database (a collection of all acute care hospitalizations) was reviewed to identify patients who received nonemergent repair of an AAA  between April 1, 2003 and March 31, 2004."
sentence_3 = "Guidelines from the AAA recommend routine hearing screenings for adults over the age of 50."

sentence_1 = remove_punctuation(sentence_1)
sentence_2 = remove_punctuation(sentence_2)
sentence_3 = remove_punctuation(sentence_3)

print(sentence_1)
print(sentence_2)
print(sentence_3)

model_name = "openai-community/gpt2"
dev_model_configs = {'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2')}
def load_model(name, all_hidden_states=True):
    configuration_class, model_class, tokeniser_class, weights = dev_model_configs[name]
    model, tokeniser = load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states)
    return model, tokeniser

def load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states=True):
    config = configuration_class.from_pretrained(weights, output_hidden_states=all_hidden_states)
    model = model_class.from_pretrained(weights, config=config)
        
    tokeniser = tokeniser_class.from_pretrained(weights)
    
    return model, tokeniser


model, tokenizer = load_model(model_name)
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})


inputs_1 = tokenizer(sentence_1, max_length = 512, return_tensors="pt", truncation=True, padding=True, output_hidden_states=True)
input_ids_1 = inputs_1["input_ids"]
attention_mask_1 = inputs_1["attention_mask"]
outputs_1 = model(**inputs_1)

inputs_2 = tokenizer(sentence_2, max_length = 512, return_tensors="pt", truncation=True, padding=True, output_hidden_states=True)
input_ids_2 = inputs_2["input_ids"]
attention_mask_2 = inputs_2["attention_mask"]
outputs_2 = model(**inputs_2)

inputs_3 = tokenizer(sentence_3, max_length = 512, return_tensors="pt", truncation=True, padding=True, output_hidden_states=True)
input_ids_3 = inputs_3["input_ids"]
attention_mask_3 = inputs_3["attention_mask"]
outputs_3 = model(**inputs_3)

# print(outputs_1)  #batch_size, sequence_length, hidden_size
# print(outputs_2)  #batch_size, sequence_length, hidden_size
# print(outputs_3)  #batch_size, sequence_length, hidden_size

output_1_hidden_states = torch.tensor(outputs_1.hidden_states[-1]).detach().squeeze(0) #sequence_length, hidden_size
output_2_hidden_states = torch.tensor(outputs_2.hidden_states[-1]).detach().squeeze(0) #sequence_length, hidden_size
output_3_hidden_states = torch.tensor(outputs_3.hidden_states[-1]).detach().squeeze(0) #sequence_length, hidden_size

print(output_1_hidden_states.shape)
print(output_2_hidden_states.shape)
print(output_3_hidden_states.shape)

print((cosine_similarity(output_1_hidden_states, output_2_hidden_states)).shape) #seq_length1, seq_length2
print((cosine_similarity(output_1_hidden_states, output_3_hidden_states)).shape) #seq_length1, seq_length3
print((cosine_similarity(output_2_hidden_states, output_3_hidden_states)).shape) #seq_length2, seq_length3