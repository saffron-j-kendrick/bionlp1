## IMPORTS 

import os
import gc
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
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, kendalltau, spearmanr, ttest_1samp, ttest_rel
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


model_name_map = {
    'meta-llama/Meta-Llama-3-8B': 'Llama-3-8B',
    'aaditya/Llama3-OpenBioLLM-8B': 'OpenBioLLM-8B',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama-3-8B-Instruct',
    'ContactDoctor/Bio-Medical-Llama-3-8B': 'Bio-Medical-Llama-3-8B',
    'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12': 'BlueBERT-PubMed',
    'google-bert/bert-base-uncased': 'BERT-base-uncased',
    'microsoft/MediPhi-Instruct': 'MediPhi-Instruct',
    'microsoft/Phi-3.5-mini-instruct': 'Phi-3.5-mini-instruct',
    'BioMistral/BioMistral-7B': 'BioMistral-7B',
    'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral-7B-Instruct-v0.1',
    'meta-llama/Llama-2-7b-hf': 'Llama-2-7B',
    'epfl-llm/meditron-7b': 'Meditron-7B',
    'dmis-lab/biobert-base-cased-v1.2': 'BioBERT',
    'google-bert/bert-base-cased': 'BERT-base-cased',
    'answerdotai/ModernBERT-base': 'ModernBERT-base',
    'thomas-sounack/BioClinical-ModernBERT-base': 'BioClinical-ModernBERT-base',
}



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

REMOTE_CODE_MODELS = {
    'microsoft/MediPhi-Instruct',
    'microsoft/Phi-3.5-mini-instruct',
}


def load_model(name, all_hidden_states=True):
    configuration_class, model_class, tokeniser_class, weights = dev_model_configs[name]
    return load_model_from_classes(
        name,
        configuration_class,
        model_class,
        tokeniser_class,
        weights,
        all_hidden_states=all_hidden_states,
    )


def load_model_from_classes(name, configuration_class, model_class, tokeniser_class, weights, all_hidden_states=True):
    common_kwargs = {'token': access_token}
    if name in REMOTE_CODE_MODELS:
        common_kwargs['trust_remote_code'] = True

    config = configuration_class.from_pretrained(
        weights,
        output_hidden_states=all_hidden_states,
        **common_kwargs,
    )

    model_kwargs = dict(common_kwargs)
    # Load the large decoder-only models in half precision and let Accelerate
    # place them across available devices. Encoder-only BERT models remain fp32.
    if model_class is AutoModelForCausalLM and torch.cuda.is_available():
        model_kwargs.update({
            'torch_dtype': torch.float16,
            'device_map': 'auto',
            'low_cpu_mem_usage': True,
        })

    model = model_class.from_pretrained(weights, config=config, **model_kwargs)
    tokeniser = tokeniser_class.from_pretrained(weights, **common_kwargs)
    return model, tokeniser


def linear_cka(X, Y):
    """Linear CKA between representation matrices X and Y (n_samples x n_features).

    Measures how similar the geometry of two sets of representations is across
    the same set of samples (sentences).  A value of 1 means the two
    representation spaces are identical up to an orthogonal transformation;
    0 means they are completely unrelated.

    Formula (Kornblith et al. 2019):
        CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c = H @ X, H = I - (1/n) 11^T is the centering matrix.
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    X_c = H @ X
    Y_c = H @ Y
    numerator = np.linalg.norm(X_c.T @ Y_c, 'fro') ** 2
    denom = np.linalg.norm(X_c.T @ X_c, 'fro') * np.linalg.norm(Y_c.T @ Y_c, 'fro')
    return float(numerator / denom) if denom > 0 else 0.0


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR-adjusted p-values, preserving input order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return adjusted

    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    adjusted[valid] = q
    return adjusted

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

def get_target_token_embeddings(model_name, model, tokeniser, input_ids, attention_mask, layers, torch_device, add_arg_dict=None, batch_size=1, middle_dim=None, target_word=None):
    print(f'Extracting target representations from model for layers {layers}')
    if add_arg_dict is None:
        add_arg_dict = {}

    # When device_map="auto" is used, Accelerate manages device placement via hooks.
    if not hasattr(model, 'hf_device_map'):
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)

    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size), dtype=np.float32)
        for _ in layers
    ]

    # Build a list of candidate token-id sequences to search for.
    # BPE tokenizers produce different IDs for the same surface form depending
    # on the preceding character (space, punctuation, start-of-sequence, etc.).
    # We try several plausible surface forms so that short abbreviations like
    # 'M' that appear after '(' or ',' are still matched.
    _forms = [
        target_word,
        ' ' + target_word,
        target_word.lower(),
        ' ' + target_word.lower(),
        target_word.upper(),
        ' ' + target_word.upper(),
    ]
    _candidate_seqs = []
    seen = set()
    for form in _forms:
        ids = tuple(tokeniser.encode(form, add_special_tokens=False))
        if ids and ids not in seen:
            seen.add(ids)
            _candidate_seqs.append(np.array(ids))

    with torch.no_grad():
        for batch_start in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
            batch_input_ids = input_ids[batch_start:batch_start + batch_size]
            batch_attention_mask = attention_mask[batch_start:batch_start + batch_size]
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )
            # hidden_states[0] is the embedding output; after slicing, index 0 is layer 1.
            hidden_states = outputs.hidden_states[1:]
            add_arg_dict['i'] = batch_start

            for layer_idx, layer in enumerate(layers):
                layer_reps = hidden_states[layer - 1].detach().cpu()
                current_batch_size = layer_reps.shape[0]
                target_token_loc_per_sent = []

                for batch_idx in range(current_batch_size):
                    sentence_ids = batch_input_ids[batch_idx].detach().cpu().numpy().reshape(-1)
                    loc = []
                    for candidate in _candidate_seqs:
                        loc = search_sequence_numpy(sentence_ids, candidate.reshape(-1))
                        if len(loc) > 0:
                            break
                    if len(loc) == 0:
                        # Fall back to mean pooling over non-padding tokens and
                        # warn rather than crash — a single unmatchable
                        # abbreviation should not abort the entire run.
                        print(
                            f"Warning: target word '{target_word}' not found in "
                            f"sentence token ids at batch index {batch_start + batch_idx}. "
                            f"Falling back to mean of non-padding tokens."
                        )
                        pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else 0
                        non_pad = np.where(sentence_ids != pad_id)[0]
                        loc = non_pad if len(non_pad) > 0 else np.arange(len(sentence_ids))
                    target_token_loc_per_sent.append(loc)

                # Average all subtokens belonging to the target word so comparisons are
                # less sensitive to tokenizer-specific word segmentation.
                target_token_reps = np.vstack([
                    reps[target_token_loc_per_sent[batch_idx]].mean(dim=0).float().numpy()
                    for batch_idx, reps in enumerate(layer_reps)
                ])

                tokens_per_layer[layer_idx][
                    batch_start:batch_start + current_batch_size, :
                ] = target_token_reps

    return tokens_per_layer


def get_mean_token_embeddings(model_name, model, tokeniser, input_ids, attention_mask, layers, torch_device, add_arg_dict=None, batch_size=1, middle_dim=None):
    print(f'Extracting mean representations from model for layers {layers}')
    if add_arg_dict is None:
        add_arg_dict = {}

    
    if not hasattr(model, 'hf_device_map'):
        input_ids = input_ids.to(torch_device)
        attention_mask = attention_mask.to(torch_device)
        model.to(torch_device)

    tokens_per_layer = [
        np.zeros((input_ids.shape[0], model.config.hidden_size), dtype=np.float32)
        for _ in layers
    ]

    def get_tokens_to_keep(token_ids):
        token_ids_cpu = token_ids.detach().cpu().tolist()
        non_special = np.array(
            tokeniser.get_special_tokens_mask(token_ids_cpu, already_has_special_tokens=True)
        ) == 0
        if tokeniser.pad_token_id is None:
            is_pad = np.zeros(len(token_ids_cpu), dtype=bool)
        else:
            is_pad = np.array(token_ids_cpu) == tokeniser.pad_token_id
        return np.flatnonzero(non_special & ~is_pad)

    with torch.no_grad():
        for batch_start in tqdm.tqdm(range(0, input_ids.shape[0], batch_size)):
            batch_input_ids = input_ids[batch_start:batch_start + batch_size]
            batch_attention_mask = attention_mask[batch_start:batch_start + batch_size]
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )
            # hidden_states[0] is the embedding output; after slicing, index 0 is layer 1.
            hidden_states = outputs.hidden_states[1:]
            add_arg_dict['i'] = batch_start

            for layer_idx, layer in enumerate(layers):
                layer_reps = hidden_states[layer - 1].detach().cpu()
                pooled = []
                for batch_idx, reps in enumerate(layer_reps):
                    keep = get_tokens_to_keep(batch_input_ids[batch_idx])
                    if len(keep) == 0:
                        raise ValueError('No non-special, non-padding tokens available for mean pooling')
                    pooled.append(reps[keep].float().mean(dim=0).numpy())

                tokens_per_layer[layer_idx][
                    batch_start:batch_start + len(pooled), :
                ] = np.vstack(pooled)

    return tokens_per_layer


## SENTENCES 


abbr_dataset = pd.read_csv("adam_sentences_filtered_with_targets.csv")
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
    sentence_a_embeddings.append((abbr_dataset.iloc[i]['Sentence_A']))
    sentence_b_embeddings.append((abbr_dataset.iloc[i]['Sentence_B']))
    sentence_c_embeddings.append((abbr_dataset.iloc[i]['Sentence_C']))
    abbrs.append(abbr_dataset.iloc[i]['ABBR'])
    targets.append(abbr_dataset.iloc[i]['target'])
    sentence_a_primes.append((abbr_dataset.iloc[i]['Sentence_A_Prime']))
    sentence_b_primes.append((abbr_dataset.iloc[i]['Sentence_B_Prime']))
    sentence_c_primes.append((abbr_dataset.iloc[i]['Sentence_C_Prime']))
    targets_primes.append(abbr_dataset.iloc[i]['target_prime'])
 


### MODELS ###


dev_model_configs = {
    'meta-llama/Meta-Llama-3-8B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'meta-llama/Meta-Llama-3-8B'),
    'aaditya/Llama3-OpenBioLLM-8B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'aaditya/Llama3-OpenBioLLM-8B'),
    'meta-llama/Meta-Llama-3-8B-Instruct': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'meta-llama/Meta-Llama-3-8B-Instruct'),
    'ContactDoctor/Bio-Medical-Llama-3-8B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'ContactDoctor/Bio-Medical-Llama-3-8B'),
    'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12': (BertConfig, BertModel, BertTokenizer, 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'),
    'google-bert/bert-base-uncased': (AutoConfig, AutoModel, AutoTokenizer, 'google-bert/bert-base-uncased'),
    'microsoft/MediPhi-Instruct': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'microsoft/MediPhi-Instruct'),
    'microsoft/Phi-3.5-mini-instruct': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'microsoft/Phi-3.5-mini-instruct'),
    'BioMistral/BioMistral-7B': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'BioMistral/BioMistral-7B'),
    'mistralai/Mistral-7B-Instruct-v0.1': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'mistralai/Mistral-7B-Instruct-v0.1'),
    'meta-llama/Llama-2-7b-hf': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'meta-llama/Llama-2-7b-hf'),
    'epfl-llm/meditron-7b': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'epfl-llm/meditron-7b'),
    'dmis-lab/biobert-base-cased-v1.2': (BertConfig, BertModel, BertTokenizer, 'dmis-lab/biobert-base-cased-v1.2'),
    'google-bert/bert-base-cased': (AutoConfig, AutoModel, AutoTokenizer, 'google-bert/bert-base-cased'),
    'answerdotai/ModernBERT-base': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'answerdotai/ModernBERT-base'),
    'thomas-sounack/BioClinical-ModernBERT-base': (AutoConfig, AutoModelForCausalLM, AutoTokenizer, 'thomas-sounack/BioClinical-ModernBERT-base'),
}




ENCODER_MODELS = {
    'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
    'google-bert/bert-base-uncased',
    'dmis-lab/biobert-base-cased-v1.2',
    'google-bert/bert-base-cased',
    'answerdotai/ModernBERT-base',
    'thomas-sounack/BioClinical-ModernBERT-base',
}

models = dev_model_configs.keys()
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            model.resize_token_embeddings(len(tokeniser))

        #unpack_dict = lambda x: (x['input_ids'], x['attention_mask'])

        
    layers = range(1, model.config.num_hidden_layers + 1)

    layers = [x for x in layers if x in range(1, model.config.num_hidden_layers + 1)]

    # iterate throguh all sentences

    sentence_a_embs = []
    sentence_b_embs = []
    sentence_c_embs = []
    sentence_a_primes_embs = []
    sentence_b_primes_embs = []
    sentence_c_primes_embs = []

    
    if model_name in ENCODER_MODELS:
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
    CosSim_12_primes = []
    CosSim_12_diff = []
    CosSim_12_primes_diff = []
    CosSim_12_full_diff = []
    # Per-sentence raw arrays per layer for t-tests
    all_sims_12 = []
    all_sims_12_primes = []
    all_sim_diff_full = []
  

    for layer_idx in range(len(layers)):
        sims_12 = []
        sims_12_primes = []
        sim_diff, sim_diff_primes = [], []
        sim_diff_full = []
        for sent_idx in range(len(sentence_a_embs)):
            emb_a = sentence_a_embs[sent_idx][layer_idx]  # shape (1, hidden_size)
            emb_b = sentence_b_embs[sent_idx][layer_idx]
            emb_c = sentence_c_embs[sent_idx][layer_idx]
            emb_a_prime = sentence_a_primes_embs[sent_idx][layer_idx]
            emb_b_prime = sentence_b_primes_embs[sent_idx][layer_idx]
            emb_c_prime = sentence_c_primes_embs[sent_idx][layer_idx]
            # (b, a) - (b, c)
            sims_12.append(cosine_similarity(emb_b, emb_a)[0][0] - cosine_similarity(emb_b, emb_c)[0][0])
            sims_12_primes.append(cosine_similarity(emb_b_prime, emb_a_prime)[0][0] - cosine_similarity(emb_b_prime, emb_c_prime)[0][0])
            sim_diff.append(cosine_similarity(emb_b, emb_a)[0][0] - cosine_similarity(emb_b, emb_c)[0][0] - cosine_similarity(emb_a, emb_c)[0][0])
            sim_diff_primes.append(cosine_similarity(emb_b_prime, emb_a_prime)[0][0] - cosine_similarity(emb_a_prime, emb_c_prime)[0][0] - cosine_similarity(emb_b_prime, emb_c_prime)[0][0])
            sim_diff_full.append((cosine_similarity(emb_b, emb_a)[0][0] - cosine_similarity(emb_b, emb_c)[0][0]) + (cosine_similarity(emb_b_prime, emb_a_prime)[0][0] - cosine_similarity(emb_b_prime, emb_c_prime)[0][0]))
            # def _norm(v):
            #     n = np.linalg.norm(v)
            #     return v / n if n > 0 else v

            # euclidean_12.append(np.linalg.norm(_norm(emb_a) - _norm(emb_b)) - np.linalg.norm(_norm(emb_a) - _norm(emb_c)))
            # euclidean_12_primes.append(np.linalg.norm(_norm(emb_a_prime) - _norm(emb_b_prime)) - np.linalg.norm(_norm(emb_a_prime) - _norm(emb_c_prime)))
    
        CosSim_12.append(np.mean(sims_12))
        CosSim_12_primes.append(np.mean(sims_12_primes))
        CosSim_12_diff.append(np.mean(sim_diff))
        CosSim_12_primes_diff.append(np.mean(sim_diff_primes))
        CosSim_12_full_diff.append(np.mean(sim_diff_full))

        all_sims_12.append(list(sims_12))
        all_sims_12_primes.append(list(sims_12_primes))
        all_sim_diff_full.append(list(sim_diff_full))


    CKA_12 = []
    CKA_12_primes = []
    CKA_12_full_diff = []

    for layer_idx in range(len(layers)):
        X_a = np.vstack([sentence_a_embs[s][layer_idx] for s in range(len(sentence_a_embs))])
        X_b = np.vstack([sentence_b_embs[s][layer_idx] for s in range(len(sentence_b_embs))])
        X_c = np.vstack([sentence_c_embs[s][layer_idx] for s in range(len(sentence_c_embs))])
        X_a_p = np.vstack([sentence_a_primes_embs[s][layer_idx] for s in range(len(sentence_a_primes_embs))])
        X_b_p = np.vstack([sentence_b_primes_embs[s][layer_idx] for s in range(len(sentence_b_primes_embs))])
        X_c_p = np.vstack([sentence_c_primes_embs[s][layer_idx] for s in range(len(sentence_c_primes_embs))])

        cka_ba   = linear_cka(X_b,   X_a)
        cka_bc   = linear_cka(X_b,   X_c)
        cka_ba_p = linear_cka(X_b_p, X_a_p)
        cka_bc_p = linear_cka(X_b_p, X_c_p)

        CKA_12.append(cka_ba - cka_bc)
        CKA_12_primes.append(cka_ba_p - cka_bc_p)
        CKA_12_full_diff.append((cka_ba - cka_bc) + (cka_ba_p - cka_bc_p))

    # One-sample t-test per layer (H0: combined score = 0) followed by
    # Benjamini-Hochberg FDR correction across all layers for this model.
    layer_x = list(range(len(all_sim_diff_full)))
    ttest_p_values = []
    for l in layer_x:
        _, p = ttest_1samp(all_sim_diff_full[l], 0)
        ttest_p_values.append(p)
    ttest_q_values = benjamini_hochberg(ttest_p_values)

    def sig_stars(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        return ''

    sig_labels = [sig_stars(q) for q in ttest_q_values]

    #save
    model_name_save = model_name.replace('/', '_')
    np.save(f'data/CosSim_12_{model_name_save}_test_with_difference.npy', CosSim_12)
    np.save(f'data/CosSim_12_primes_{model_name_save}_test_with_difference.npy', CosSim_12_primes)
    np.save(f'data/CosSim_12_diff_{model_name_save}_test_with_difference.npy', CosSim_12_diff)
    np.save(f'data/CosSim_12_primes_diff_{model_name_save}_test_with_difference.npy', CosSim_12_primes_diff)
    np.save(f'data/CosSim_12_full_diff_{model_name_save}_test_with_difference.npy', CosSim_12_full_diff)
    # Raw per-sentence arrays (shape: num_layers × num_sentences) needed for t-tests
    np.save(f'data/all_sims_12_{model_name_save}_test_with_difference.npy', np.array(all_sims_12))
    np.save(f'data/all_sims_12_primes_{model_name_save}_test_with_difference.npy', np.array(all_sims_12_primes))
    np.save(f'data/all_sim_diff_full_{model_name_save}_test_with_difference.npy', np.array(all_sim_diff_full))
    np.save(f'data/ttest_p_values_{model_name_save}_test_with_difference.npy', np.asarray(ttest_p_values))
    np.save(f'data/ttest_q_values_bh_fdr_{model_name_save}_test_with_difference.npy', np.asarray(ttest_q_values))

    # CKA results
    np.save(f'data/CKA_12_{model_name_save}_test_with_difference.npy', np.array(CKA_12))
    np.save(f'data/CKA_12_primes_{model_name_save}_test_with_difference.npy', np.array(CKA_12_primes))
    np.save(f'data/CKA_12_full_diff_{model_name_save}_test_with_difference.npy', np.array(CKA_12_full_diff))
    # # np.save(f'data/Euclidean_12_{model_name_save}_test_with_primes.npy', Euclidean_12)
    # np.save(f'data/Euclidean_13_{model_name_save}_test_with_primes.npy', Euclidean_13)
    # np.save(f'data/Euclidean_23_{model_name_save}_test_with_primes.npy', Euclidean_23)
    # np.save(f'data/Euclidean_12_primes_{model_name_save}_test_with_primes.npy', Euclidean_12_primes)
    # np.save(f'data/Euclidean_13_primes_{model_name_save}_test_with_primes.npy', Euclidean_13_primes)
    # np.save(f'data/Euclidean_23_primes_{model_name_save}_test_with_primes.npy', Euclidean_23_primes)
    # subplot three lines, cossims and then primes
    display_name = model_name_map.get(model_name, model_name.split('/')[-1])

    fig, axes = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=False)

    layer_numbers = list(range(1, len(CosSim_12) + 1))
    sig_x = [layer_numbers[l] for l, lbl in enumerate(sig_labels) if lbl]
    sig_marker_handle = None
    for ax in [axes[0]]:
        if sig_x:
            h, = ax.plot(sig_x, [0.0] * len(sig_x), marker='*', linestyle='none',
                         color='red', markersize=5, label='q<0.05 (one-sample t-test, BH-FDR)', zorder=5)
            sig_marker_handle = h

    axes[0].plot(layer_numbers, CosSim_12, label='<SB–SA> - <SB–SC>', color='deepskyblue')
    axes[0].plot(layer_numbers, CosSim_12_primes, label="<SB'–SA'> - <SB'–SC'>", color='chartreuse')
    axes[0].set_title('Independent Components')
    axes[0].legend(loc='best')

    axes[1].plot(layer_numbers, CosSim_12_full_diff,
                 label="(<SB–SA>-<SB–SC>) + (<SB'–SA'>-<SB'–SC'>)", color='darkorange')
    axes[1].set_title('Combined Similarity Difference')
    axes[1].legend(loc='best')

    fig.supxlabel('Layer')
    fig.supylabel('Average Cosine Similarity Diff')
    fig.suptitle(f'Combined Cosine Similarity Difference for {display_name}')
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_twoplot.png')
    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_twoplot.eps')
    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_twoplot.pdf')
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT))

    ax = fig.gca()
    ax.plot(layer_numbers, CosSim_12_full_diff,
            label="(<SB–SA>-<SB–SC>) + (<SB'–SA'>-<SB'–SC'>)", color='darkorange')

    # # Asterisks at y=0 for one-sample t-test significance
    # sig_x = [layer_numbers[l] for l, lbl in enumerate(sig_labels) if lbl]
    # if sig_x:
    #     ax.plot(sig_x, [0.0] * len(sig_x), marker='*', linestyle='none',
    #             color='red', markersize=5, label='q<0.05 (one-sample t-test, BH-FDR)', zorder=5)

    ax.legend(loc='best')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Combined Cosine Similarity Diff')
    ax.set_title(f'Combined Cosine Similarity Difference for {display_name}')
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])
    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_oneplot.png')
    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_oneplot.eps')
    fig.savefig(f'figures/CosineSimilarityDifference_{display_name}_test_oneplot.pdf')
    plt.show()
    plt.close(fig)

    fig_cka, axes_cka = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=False)

    axes_cka[0].plot(layer_numbers, CKA_12,        label='CKA(B,A) - CKA(B,C)',         color='deepskyblue')
    axes_cka[0].plot(layer_numbers, CKA_12_primes, label="CKA(B',A') - CKA(B',C')",     color='chartreuse')
    axes_cka[0].set_title('Independent Components (CKA)')
    axes_cka[0].legend(loc='best')

    axes_cka[1].plot(layer_numbers, CKA_12_full_diff,
                     label="(CKA(B,A)-CKA(B,C)) + (CKA(B',A')-CKA(B',C'))", color='darkorange')
    axes_cka[1].set_title('Combined CKA Difference')
    axes_cka[1].legend(loc='best')

    fig_cka.supxlabel('Layer')
    fig_cka.supylabel('CKA Difference')
    fig_cka.suptitle(f'Combined CKA Difference for {display_name}')
    fig_cka.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    fig_cka.savefig(f'figures/CKADifference_{display_name}_test_twoplot.png')
    fig_cka.savefig(f'figures/CKADifference_{display_name}_test_twoplot.eps')
    fig_cka.savefig(f'figures/CKADifference_{display_name}_test_twoplot.pdf')
    plt.show()
    plt.close(fig_cka)

    fig_cka1 = plt.figure(figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT))
    ax_cka1 = fig_cka1.gca()
    ax_cka1.plot(layer_numbers, CKA_12_full_diff,
                 label="(CKA(B,A)-CKA(B,C)) + (CKA(B',A')-CKA(B',C'))", color='darkorange')
    ax_cka1.legend(loc='best')
    ax_cka1.set_xlabel('Layer')
    ax_cka1.set_ylabel('Combined CKA Difference')
    ax_cka1.set_title(f'Combined CKA Difference for {display_name}')
    fig_cka1.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])
    fig_cka1.savefig(f'figures/CKADifference_{display_name}_test_oneplot.png')
    fig_cka1.savefig(f'figures/CKADifference_{display_name}_test_oneplot.eps')
    fig_cka1.savefig(f'figures/CKADifference_{display_name}_test_oneplot.pdf')
    plt.show()
    plt.close(fig_cka1)


    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=False)

    axes_comp[0].plot(layer_numbers, CosSim_12_full_diff, color='darkorange',
                      label="(<SB–SA>-<SB–SC>) + (<SB'–SA'>-<SB'–SC'>)")
    axes_comp[0].set_title('Cosine Similarity Difference')
    axes_comp[0].set_xlabel('Layer')
    axes_comp[0].set_ylabel('Avg Cosine Sim Diff')
    axes_comp[0].legend(loc='best')

    axes_comp[1].plot(layer_numbers, CKA_12_full_diff, color='mediumpurple',
                      label="(CKA(B,A)-CKA(B,C)) + (CKA(B',A')-CKA(B',C'))")
    axes_comp[1].set_title('CKA Difference')
    axes_comp[1].set_xlabel('Layer')
    axes_comp[1].set_ylabel('CKA Diff')
    axes_comp[1].legend(loc='best')

    fig_comp.suptitle(f'CosSim vs CKA – {display_name}')
    fig_comp.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig_comp.savefig(f'figures/CosSim_vs_CKA_{display_name}_test.png')
    fig_comp.savefig(f'figures/CosSim_vs_CKA_{display_name}_test.eps')
    fig_comp.savefig(f'figures/CosSim_vs_CKA_{display_name}_test.pdf')
    plt.show()
    plt.close(fig_comp)

    # Release the current model before loading the next checkpoint.
    del model
    del tokeniser
    del sentence_a_embs, sentence_b_embs, sentence_c_embs
    del sentence_a_primes_embs, sentence_b_primes_embs, sentence_c_primes_embs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()






