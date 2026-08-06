## IMPORTS 

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
from scipy.stats import pearsonr, kendalltau, spearmanr, ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns

ACL_COLUMN_WIDTH = 3.30
ACL_TEXT_WIDTH = 7.00
ACL_RSA_HEIGHT = 4.50
ACL_SINGLE_HEIGHT = 2.80

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    # Embed TrueType fonts in vector output for reliable PDF rendering.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

access_token = os.environ.get('HF_TOKEN')

if access_token is None:
    raise ValueError("HF_TOKEN is not set")


model_name_map = {'meta-llama/Llama-3.2-3B' : 'Llama', "openai-community/gpt2" : "GPT2", "tiiuae/Falcon3-7B-Base" : "Falcon", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" : "DeepSeek", "Qwen/Qwen2.5-7B" : "Qwen", "mistralai/Mistral-7B-v0.1" : "Mistral", "microsoft/biogpt" : "BioGPT", "google/multiberts-seed_3" : "MultiBERTs", "FacebookAI/roberta-base" : "RoBERTa", "dmis-lab/biobert-base-cased-v1.2" : "BioBERT", "ContactDoctor/Bio-Medical-Llama-3-8B" : "Bio-Medical-Llama", "tarun7r/Finance-Llama-8B" : "Finance-Llama", "marcev/financebert" : "FinanceBERT"}


## FUNCTIONS


def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def remove_abbreviations(abbr, sentence):
    sentence = re.sub(r'\([^)]*\)', '', sentence)

    if abbr in sentence:
        sentence = re.sub(abbr, '', sentence)
    return sentence

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in sentence.split() if word not in stop_words])

def tokenize_sentence(sentence):
    return word_tokenize(sentence)

def load_model(name, all_hidden_states=True):
    configuration_class, model_class, tokeniser_class, weights = dev_model_configs[name]
    model, tokeniser = load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states)
    return model, tokeniser

def load_model_from_classes(configuration_class, model_class, tokeniser_class, weights, all_hidden_states=True):
    config = configuration_class.from_pretrained(weights, output_hidden_states=all_hidden_states)
    model = model_class.from_pretrained(weights, config=config)
        
    tokeniser = tokeniser_class.from_pretrained(weights)
    
    return model, tokeniser

def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/a/36535397
    # 
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []   

def find_target_positions(tokeniser, sentence_ids, target_word):
    """Find the token indices in sentence_ids that correspond to target_word.

    Tries three strategies in order:
    1. Exact token-id match (no leading space).
    2. Exact token-id match with a leading space (handles GPT-2 / Mistral-style
       SentencePiece tokenisers where mid-sentence words carry a ▁ prefix).
    3. Decode-and-match fallback: slide a window over the token sequence,
       decode each span, and compare to the target string.  This handles cases
       where the in-context subword split differs from the isolated encoding
       (common with SentencePiece BPE models such as Mistral).
    Returns the indices array (same format as search_sequence_numpy) or [].
    """
    # Strategy 1: bare word
    ids = np.array(tokeniser.encode(target_word, add_special_tokens=False))
    loc = search_sequence_numpy(sentence_ids, ids.reshape(-1))
    if len(loc) > 0:
        return loc

    # Strategy 2: leading space
    ids_spaced = np.array(tokeniser.encode(' ' + target_word, add_special_tokens=False))
    loc = search_sequence_numpy(sentence_ids, ids_spaced.reshape(-1))
    if len(loc) > 0:
        return loc

    # Strategy 3: decode-and-match (case-insensitive, strips whitespace)
    target_norm = target_word.strip().lower()
    n = len(sentence_ids)
    for start in range(n):
        for end in range(start + 1, min(start + 12, n + 1)):
            span_text = tokeniser.decode(sentence_ids[start:end]).strip().lower()
            if span_text == target_norm:
                return np.array(list(range(start, end)))

    return []

def get_target_token_embeddings(model_name, model, tokeniser, input_ids, attention_mask, layers, torch_device, add_arg_dict={}, batch_size = 1, middle_dim=None, target_word=None):
    print(f'Extracting target representations from model for layers {layers}')
    # When device_map="auto" is used, accelerate manages device placement via hooks;
    # calling model.to() or moving inputs manually would conflict with those hooks.
    if not hasattr(model, 'hf_device_map'):
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)

    # Initialize token representations dynamically 
    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size))
        for _ in layers
    ]

    # Extracting representations during the forward pass
    with torch.no_grad():
        # Using batch_start to avoid shadowing 'i'
        for batch_start in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
            outputs = model(
                input_ids[batch_start:batch_start+batch_size],
                attention_mask=attention_mask[batch_start:batch_start+batch_size],
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[1:]  # Exclude embeddings
            add_arg_dict["i"] = batch_start

            # Calculate the sequence lengths for the current batch
            batch_attention = attention_mask[batch_start:batch_start+batch_size]
            sequence_lengths = batch_attention.sum(dim=1).cpu().numpy()

            for layer_idx, layer in enumerate(layers):
                token_reps = hidden_states



                # Get the layer representations
                if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
                    layer_reps = token_reps[1][layer].cpu()[:, :, :]
                elif layer == model.config.num_hidden_layers:
                    layer_reps = token_reps[0].cpu()[:, :, :]
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'tarun7r/Finance-Llama-8B', 'marcev/financebert']:
                    layer_reps = token_reps[layer].cpu()[:, :, :]
                else:
                    layer_reps = token_reps[2][layer].cpu()[:, :, :]
                    
                # Locate target word tokens using the three-strategy helper
                # (bare, leading-space, and decode-and-match fallback for
                # SentencePiece models like Mistral whose in-context splits
                # may differ from isolated encodings).
                current_batch_size = layer_reps.shape[0]
                target_token_loc_per_sent = []
                for i in range(current_batch_size):
                    sentence_ids = input_ids[batch_start + i, :].cpu().numpy().reshape(-1)
                    loc = find_target_positions(tokeniser, sentence_ids, target_word)
                    if len(loc) == 0:
                        raise ValueError(f"Target word '{target_word}' not found in sentence token ids at batch index {batch_start + i}")
                    target_token_loc_per_sent.append(loc)
                target_token_reps = np.vstack([reps[target_token_loc_per_sent[i][-1]].cpu().float().numpy() for i, reps in enumerate(layer_reps)])

                tokens_per_layer[layer_idx][batch_start:batch_start+batch_size, :] = target_token_reps
    
    return tokens_per_layer


def get_mean_token_embeddings(model_name, model, tokeniser, input_ids, attention_mask, layers, torch_device, add_arg_dict={}, batch_size = 1, middle_dim=None):
    print(f'Extracting mean epresentations from model for layers {layers}')
    # When device_map="auto" is used, accelerate manages device placement via hooks;
    # calling model.to() or moving inputs manually would conflict with those hooks.
    if not hasattr(model, 'hf_device_map'):
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)

    # Initialize token representations dynamically 
    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size))
        for _ in layers
    ]

    # Extracting representations during the forward pass
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
            outputs = model(
                input_ids[i:i+batch_size],
                attention_mask=attention_mask[i:i+batch_size],
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[1:]  # Exclude embeddings
            add_arg_dict["i"] = i

            # process each layer and get the mean token embedding
            for layer_idx, layer in enumerate(layers):
                token_reps = hidden_states

                # Get tokens where tokens aren't special tokens or pad tokens
                non_special_token_mask = lambda x: np.array(tokeniser.get_special_tokens_mask(x, already_has_special_tokens=True)) == 0
                pad_token_mask = lambda x: np.array(x.cpu() == tokeniser.pad_token_id)
                get_tokens_to_keep = lambda x: np.argwhere(non_special_token_mask(x) * (pad_token_mask(x) == False)).reshape(-1)

                # Get the mean token embedding
                if model_name in ['distilroberta-base', 'xlnet-base-cased', 'xlm-mlm-xnli15-1024']:
                    layer_reps = token_reps[1][layer].cpu()[:, :, :]
                elif layer == model.config.num_hidden_layers:
                    layer_reps = token_reps[0].cpu()[:, :, :]
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'tarun7r/Finance-Llama-8B', 'marcev/financebert']:
                    layer_reps = token_reps[layer].cpu()[:, :, :]
                else:
                    layer_reps = token_reps[2][layer].cpu()[:, :, :]
                
                tokens_per_layer[layer_idx][i:i+batch_size, :] = np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
                
                # print(tokens_per_layer.shape)

    return tokens_per_layer

## SENTENCES 


abbr_dataset = pd.read_excel("triple_sentence_test_with_primes.xlsx")
sentence_a_embeddings = []
sentence_a_primes = []
sentence_b_embeddings = []
sentence_b_primes = []
sentence_c_embeddings = []
sentence_c_primes = []
abbrs = []
targets = []
targets_primes = []

for i in range(len(abbr_dataset)):
    sentence_a_embeddings.append(abbr_dataset.iloc[i]['sentence_a'])
    sentence_b_embeddings.append(abbr_dataset.iloc[i]['sentence_b'])
    sentence_c_embeddings.append(abbr_dataset.iloc[i]['sentence_c'])
    abbrs.append(abbr_dataset.iloc[i]['abbr'])
    targets.append(abbr_dataset.iloc[i]['target'])
    sentence_a_primes.append(abbr_dataset.iloc[i]['sentence_a_prime'])
    sentence_b_primes.append(abbr_dataset.iloc[i]['sentence_b_prime'])
    sentence_c_primes.append(abbr_dataset.iloc[i]['sentence_c_prime'])
    targets_primes.append(abbr_dataset.iloc[i]['target_prime'])
 


### MODELS ###

dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B'),
                    'microsoft/biogpt' : (AutoConfig.from_pretrained("microsoft/biogpt", token = access_token), AutoModelForCausalLM.from_pretrained("microsoft/biogpt", token = access_token), AutoTokenizer.from_pretrained("microsoft/biogpt", token = access_token), 'microsoft/biogpt'),
                    'google/multiberts-seed_3' : (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForMaskedLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), 'google/multiberts-seed_3'),
                    'marcev/financebert' : (AutoConfig.from_pretrained("marcev/financebert"), AutoModelForMaskedLM.from_pretrained("marcev/financebert"), AutoTokenizer.from_pretrained("marcev/financebert"), 'marcev/financebert'),
                    'dmis-lab/biobert-base-cased-v1.2' : (AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), 'dmis-lab/biobert-base-cased-v1.2')}

#dev_model_configs = {'tarun7r/Finance-Llama-8B' : (AutoConfig.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoModelForCausalLM.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoTokenizer.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), 'tarun7r/Finance-Llama-8B')}

models = dev_model_configs.keys()

torch_device = torch.device("cuda")



for model_name in tqdm.tqdm(models):
    print('Loading {}'.format(model_name))
    model, tokeniser = load_model(model_name)

    print("Model loaded successfully")
    model.eval()
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})

    layers = range(1, model.config.num_hidden_layers + 1)

    layers = [x for x in layers if x in range(1, model.config.num_hidden_layers + 1)]

    # iterate throguh all sentences

    sentence_a_embs = []
    sentence_b_embs = []
    sentence_c_embs = []
    sentence_a_primes_embs = []
    sentence_b_primes_embs = []
    sentence_c_primes_embs = []

    
    if model_name in ['google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2', 'marcev/financebert']:
        print(f'Extracting mean token embeddings for {model_name}')
        for i in range(len(sentence_a_embeddings)):
            sent_a = sentence_a_embeddings[i]
            sent_b = sentence_b_embeddings[i]
            sent_c = sentence_c_embeddings[i]
            target = targets[i]
            sent_a_prime = sentence_a_primes[i]
            sent_b_prime = sentence_b_primes[i]
            sent_c_prime = sentence_c_primes[i]
            target_prime = targets_primes[i]

            inputs_a = tokeniser(sent_a,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_a = inputs_a["input_ids"]
            attention_mask_a = inputs_a["attention_mask"]
            embeddings_a = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_a, attention_mask_a, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_a_embs.append(embeddings_a)

            inputs_a_prime = tokeniser(sent_a_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_a_prime = inputs_a_prime["input_ids"]
            attention_mask_a_prime = inputs_a_prime["attention_mask"]
            embeddings_a_prime = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_a_prime, attention_mask_a_prime, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_a_primes_embs.append(embeddings_a_prime)

            inputs_b = tokeniser(sent_b,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_b = inputs_b["input_ids"]
            attention_mask_b = inputs_b["attention_mask"]
            embeddings_b = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_b, attention_mask_b, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_b_embs.append(embeddings_b)

            inputs_b_prime = tokeniser(sent_b_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_b_prime = inputs_b_prime["input_ids"]
            attention_mask_b_prime = inputs_b_prime["attention_mask"]
            embeddings_b_prime = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_b_prime, attention_mask_b_prime, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_b_primes_embs.append(embeddings_b_prime)

            inputs_c = tokeniser(sent_c,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_c = inputs_c["input_ids"]
            attention_mask_c = inputs_c["attention_mask"]
            embeddings_c = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_c, attention_mask_c, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_c_embs.append(embeddings_c)

            inputs_c_prime = tokeniser(sent_c_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_c_prime = inputs_c_prime["input_ids"]
            attention_mask_c_prime = inputs_c_prime["attention_mask"]
            embeddings_c_prime = get_mean_token_embeddings(model_name, model, tokeniser, input_ids_c_prime, attention_mask_c_prime, layers, torch_device, batch_size=1, middle_dim=None)
            sentence_c_primes_embs.append(embeddings_c_prime)
    else:
        print(f'Extracting target token embeddings for {model_name}')
        for i in range(len(sentence_a_embeddings)):
            sent_a = sentence_a_embeddings[i]
            sent_b = sentence_b_embeddings[i]
            sent_c = sentence_c_embeddings[i]
            target = targets[i]
            sent_a_prime = sentence_a_primes[i]
            sent_b_prime = sentence_b_primes[i]
            sent_c_prime = sentence_c_primes[i]
            target_prime = targets_primes[i]

            inputs_a = tokeniser(sent_a,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_a = inputs_a["input_ids"]
            attention_mask_a = inputs_a["attention_mask"]
            embeddings_a = get_target_token_embeddings(model_name, model, tokeniser, input_ids_a, attention_mask_a, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
            sentence_a_embs.append(embeddings_a)

            inputs_a_prime = tokeniser(sent_a_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_a_prime = inputs_a_prime["input_ids"]
            attention_mask_a_prime = inputs_a_prime["attention_mask"]
            embeddings_a_prime = get_target_token_embeddings(model_name, model, tokeniser, input_ids_a_prime, attention_mask_a_prime, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_prime)
            sentence_a_primes_embs.append(embeddings_a_prime)

            inputs_b = tokeniser(sent_b,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_b = inputs_b["input_ids"]
            attention_mask_b = inputs_b["attention_mask"]
            embeddings_b = get_target_token_embeddings(model_name, model, tokeniser, input_ids_b, attention_mask_b, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
            sentence_b_embs.append(embeddings_b)

            inputs_b_prime = tokeniser(sent_b_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_b_prime = inputs_b_prime["input_ids"]
            attention_mask_b_prime = inputs_b_prime["attention_mask"]
            embeddings_b_prime = get_target_token_embeddings(model_name, model, tokeniser, input_ids_b_prime, attention_mask_b_prime, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_prime)
            sentence_b_primes_embs.append(embeddings_b_prime)

            inputs_c = tokeniser(sent_c,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_c = inputs_c["input_ids"]
            attention_mask_c = inputs_c["attention_mask"]
            embeddings_c = get_target_token_embeddings(model_name, model, tokeniser, input_ids_c, attention_mask_c, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
            sentence_c_embs.append(embeddings_c)

            inputs_c_prime = tokeniser(sent_c_prime,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
            input_ids_c_prime = inputs_c_prime["input_ids"]
            attention_mask_c_prime = inputs_c_prime["attention_mask"]
            embeddings_c_prime = get_target_token_embeddings(model_name, model, tokeniser, input_ids_c_prime, attention_mask_c_prime, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_prime)
            sentence_c_primes_embs.append(embeddings_c_prime)
    sentence_a_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_a_embs]
    sentence_b_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_b_embs]
    sentence_c_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_c_embs]
    sentence_a_primes_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_a_primes_embs]
    sentence_b_primes_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_b_primes_embs]
    sentence_c_primes_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_c_primes_embs]

    layers = range(1, model.config.num_hidden_layers + 1)
    CosSim_12 = []
    CosSim_13 = []
    CosSim_23 = []

    CosSim_12_primes = []
    CosSim_13_primes = []
    CosSim_23_primes = []

    PearsonCorr_12 = []
    PearsonCorr_13 = []
    PearsonCorr_23 = []

    PearsonCorr_12_primes = []
    PearsonCorr_13_primes = []
    PearsonCorr_23_primes = []
   

    for layer_idx in range(len(layers)):
        sims_12, sims_13, sims_23 = [], [], []
        sims_12_primes, sims_13_primes, sims_23_primes = [], [], []
        for sent_idx in range(len(sentence_a_embs)):
            emb_a = sentence_a_embs[sent_idx][layer_idx]  # shape (1, hidden_size)
            emb_b = sentence_b_embs[sent_idx][layer_idx]
            emb_c = sentence_c_embs[sent_idx][layer_idx]
            emb_a_prime = sentence_a_primes_embs[sent_idx][layer_idx]
            emb_b_prime = sentence_b_primes_embs[sent_idx][layer_idx]
            emb_c_prime = sentence_c_primes_embs[sent_idx][layer_idx]
            sims_12.append(cosine_similarity(emb_a, emb_b)[0][0])
            sims_13.append(cosine_similarity(emb_a, emb_c)[0][0])
            sims_23.append(cosine_similarity(emb_b, emb_c)[0][0])
            sims_12_primes.append(cosine_similarity(emb_a_prime, emb_b_prime)[0][0])
            sims_13_primes.append(cosine_similarity(emb_a_prime, emb_c_prime)[0][0])
            sims_23_primes.append(cosine_similarity(emb_b_prime, emb_c_prime)[0][0])
        
    
        CosSim_12.append(np.mean(sims_12))
        CosSim_13.append(np.mean(sims_13))
        CosSim_23.append(np.mean(sims_23))
        CosSim_12_primes.append(np.mean(sims_12_primes))
        CosSim_13_primes.append(np.mean(sims_13_primes))
        CosSim_23_primes.append(np.mean(sims_23_primes))


    # subplot three lines, cossims and then primes
    model_name = model_name_map[model_name]

    fig, axes = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=True)

    axes[0].plot(CosSim_12, label='S1–S2')
    axes[0].plot(CosSim_13, label='S1–S3')
    axes[0].plot(CosSim_23, label='S2–S3')
    axes[0].set_title('Original')
    axes[0].legend(loc='best')

    axes[1].plot(CosSim_12_primes, label="S1' - S2'")
    axes[1].plot(CosSim_13_primes, label="S1' - S3'")
    axes[1].plot(CosSim_23_primes, label="S2' - S3'")
    axes[1].set_title('Primes')
    axes[1].legend(loc='best')

    fig.supxlabel('Layer')
    fig.supylabel('Average Cosine Similarity')
    fig.suptitle(f'Average Cosine Similarity for {model_name}')
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test_with_primes.png')
    fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test_with_primes.eps')
    fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test_with_primes.pdf')
    plt.show()
    plt.close(fig)

    # plt.plot(CosSim_12, label='Pair: Sentence 1 and Sentence 2')
    # plt.plot(CosSim_13, label='Pair: Sentence 1 and Sentence 3')
    # plt.plot(CosSim_23, label='Pair: Sentence 2 and Sentence 3')
    # plt.xlabel('Layer')
    # plt.ylabel('Average Cosine Similarity')
    # if model_name in ['MultiBERTs', 'RoBERTa', 'BioBERT']:
    #     plt.title(f'Average Cosine Similarity using the mean token embeddings for {model_name}')
    # else:
    #     plt.title(f'Average Cosine Similarity using the target token embeddings for {model_name}')
    # plt.legend()
    # plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test.png')
    # plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test.eps')
    # plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test.pdf')
    # plt.show()
    # plt.close()

  