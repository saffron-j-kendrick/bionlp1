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


model_name_map = {'meta-llama/Llama-3.2-3B' : 'Llama-3B', "openai-community/gpt2" : "GPT2", "tiiuae/Falcon3-7B-Base" : "Falcon", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" : "DeepSeek", "Qwen/Qwen2.5-7B" : "Qwen", "mistralai/Mistral-7B-v0.1" : "Mistral", "microsoft/biogpt" : "BioGPT", "google/multiberts-seed_3" : "MultiBERTs", "FacebookAI/roberta-base" : "RoBERTa", "dmis-lab/biobert-base-cased-v1.2" : "BioBERT", "ContactDoctor/Bio-Medical-Llama-3-8B" : "Bio-Medical-Llama", "tarun7r/Finance-Llama-8B" : "Finance-Llama", "marcev/financebert" : "FinanceBERT", "meta-llama/Meta-Llama-3-8B" : "Llama"}


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
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'tarun7r/Finance-Llama-8B', 'marcev/financebert', 'meta-llama/Meta-Llama-3-8B']:
                    layer_reps = token_reps[layer].cpu()[:, :, :]
                else:
                    layer_reps = token_reps[2][layer].cpu()[:, :, :]
                    
                # get the target token ids (encoded once, no special tokens)
                # also try with a leading space for tokenizers that encode mid-sentence words differently (e.g. GPT-2)
                target_token_ids = np.array(tokeniser.encode(target_word, add_special_tokens=False))
                target_token_ids_spaced = np.array(tokeniser.encode(' ' + target_word, add_special_tokens=False))
                current_batch_size = layer_reps.shape[0]
                target_token_loc_per_sent = []
                for i in range(current_batch_size):
                    sentence_ids = input_ids[batch_start + i, :].cpu().numpy().reshape(-1)
                    loc = search_sequence_numpy(sentence_ids, target_token_ids.reshape(-1))
                    if len(loc) == 0:
                        loc = search_sequence_numpy(sentence_ids, target_token_ids_spaced.reshape(-1))
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
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'tarun7r/Finance-Llama-8B', 'marcev/financebert', 'meta-llama/Meta-Llama-3-8B']:
                    layer_reps = token_reps[layer].cpu()[:, :, :]
                else:
                    layer_reps = token_reps[2][layer].cpu()[:, :, :]
                
                tokens_per_layer[layer_idx][i:i+batch_size, :] = np.vstack([np.mean(reps[get_tokens_to_keep(input_ids[i])].cpu().numpy(), axis=0) for i, reps in enumerate(layer_reps)])
                
                # print(tokens_per_layer.shape)

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

# dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B'),
#                     'ContactDoctor/Bio-Medical-Llama-3-8B' : (AutoConfig.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoModelForCausalLM.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), 'ContactDoctor/Bio-Medical-Llama-3-8B'),
#                     'tarun7r/Finance-Llama-8B' : (AutoConfig.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoModelForCausalLM.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoTokenizer.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), 'tarun7r/Finance-Llama-8B'),
#                     'microsoft/biogpt' : (AutoConfig.from_pretrained("microsoft/biogpt", token = access_token), AutoModelForCausalLM.from_pretrained("microsoft/biogpt", token = access_token), AutoTokenizer.from_pretrained("microsoft/biogpt", token = access_token), 'microsoft/biogpt'),
#                     'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2'),
#                     'Qwen/Qwen2.5-7B' : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), 'Qwen/Qwen2.5-7B'),
#                     'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"), 'mistralai/Mistral-7B-v0.1'),
#                     'tiiuae/Falcon3-7B-Base' : (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), 'tiiuae/Falcon3-7B-Base'),
#                     'google/multiberts-seed_3' : (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForMaskedLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), 'google/multiberts-seed_3'),
#                     'FacebookAI/roberta-base' : (AutoConfig.from_pretrained("FacebookAI/roberta-base"), AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base"), AutoTokenizer.from_pretrained("FacebookAI/roberta-base"), 'FacebookAI/roberta-base'),
#                     'dmis-lab/biobert-base-cased-v1.2' : (AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), 'dmis-lab/biobert-base-cased-v1.2')}

# dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B'),
#                      'ContactDoctor/Bio-Medical-Llama-3-8B' : (AutoConfig.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoModelForCausalLM.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), 'ContactDoctor/Bio-Medical-Llama-3-8B'),
#                      'tarun7r/Finance-Llama-8B' : (AutoConfig.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoModelForCausalLM.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoTokenizer.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), 'tarun7r/Finance-Llama-8B'),
#                     'google/multiberts-seed_3' : (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForMaskedLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), 'google/multiberts-seed_3'),
#                     'marcev/financebert' : (AutoConfig.from_pretrained("marcev/financebert"), AutoModelForMaskedLM.from_pretrained("marcev/financebert"), AutoTokenizer.from_pretrained("marcev/financebert"), 'marcev/financebert'),
#                     'dmis-lab/biobert-base-cased-v1.2' : (AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), 'dmis-lab/biobert-base-cased-v1.2')}


dev_model_configs = {'meta-llama/Meta-Llama-3-8B' : (AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token = access_token) , 'meta-llama/Meta-Llama-3-8B'),
                     'ContactDoctor/Bio-Medical-Llama-3-8B' : (AutoConfig.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoModelForCausalLM.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B", token = access_token), 'ContactDoctor/Bio-Medical-Llama-3-8B'),
                     'tarun7r/Finance-Llama-8B' : (AutoConfig.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoModelForCausalLM.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), AutoTokenizer.from_pretrained("tarun7r/Finance-Llama-8B", token = access_token), 'tarun7r/Finance-Llama-8B'),
                    }


models = dev_model_configs.keys()

# models = ['tarun7r/Finance-Llama-8B', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'meta-llama/Llama-3.2-3B']
# tokenizer = AutoTokenizer.from_pretrained("tarun7r/Finance-Llama-8B")
# model = AutoModelForCausalLM.from_pretrained("tarun7r/Finance-Llama-8B", device_map="auto")
torch_device = torch.device("cuda")



for model_name in tqdm.tqdm(models):
    print('Loading {}'.format(model_name))
    #model, tokeniser = load_model(model_name)

    if model_name in ['ContactDoctor/Bio-Medical-Llama-3-8B', 'tarun7r/Finance-Llama-8B']:
        tokeniser = AutoTokenizer.from_pretrained(model_name, token = access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token = access_token,
            torch_dtype=torch.float16,  # Half precision
            device_map="auto",          # Automatic device placement
            low_cpu_mem_usage=True,     # Efficient CPU memory usage during loading
            trust_remote_code=True
        )
    else:
        model, tokeniser = load_model(model_name)
    print("Model loaded successfully")
    model.eval()
    if tokeniser.pad_token is None:
        if tokeniser.eos_token:
            tokeniser.pad_token = tokeniser.eos_token
        else:
            tokeniser.add_special_tokens({'pad_token': '<pad>'})

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

    Euclidean_12 = []
    Euclidean_13 = []
    Euclidean_23 = []

    CosSim_12_primes = []
    CosSim_13_primes = []
    CosSim_23_primes = []
    CosSim_12_diff = []
    CosSim_12_primes_diff = []
    Euclidean_12_primes = []
    Euclidean_13_primes = []
    Euclidean_23_primes = []

    # Per-sentence raw arrays per layer for paired t-tests
    all_sims_12 = []
    all_sims_12_primes = []
  

    for layer_idx in range(len(layers)):
        sims_12, sims_13, sims_23 = [], [], []
        sims_12_primes, sims_13_primes, sims_23_primes = [], [], []
        sim_diff, sim_diff_primes = [], []
        euclidean_12, euclidean_13, euclidean_23 = [], [], []
        euclidean_12_primes, euclidean_13_primes, euclidean_23_primes = [], [], []
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

            # def _norm(v):
            #     n = np.linalg.norm(v)
            #     return v / n if n > 0 else v

            # euclidean_12.append(np.linalg.norm(_norm(emb_a) - _norm(emb_b)) - np.linalg.norm(_norm(emb_a) - _norm(emb_c)))
            # euclidean_12_primes.append(np.linalg.norm(_norm(emb_a_prime) - _norm(emb_b_prime)) - np.linalg.norm(_norm(emb_a_prime) - _norm(emb_c_prime)))
    
        CosSim_12.append(np.mean(sims_12))
        CosSim_12_primes.append(np.mean(sims_12_primes))
        CosSim_12_diff.append(np.mean(sim_diff))
        CosSim_12_primes_diff.append(np.mean(sim_diff_primes))
        # Euclidean_12.append(np.mean(euclidean_12))
        # Euclidean_12_primes.append(np.mean(euclidean_12_primes))
        all_sims_12.append(list(sims_12))
        all_sims_12_primes.append(list(sims_12_primes))

    # Paired t-test per layer: <SB–SA>-<SB-SC> vs <SB'–SA'>-<SB'–SC'>
    layer_x = list(range(len(all_sims_12)))
    ttest_p_values = []
    for l in layer_x:
        _, p = ttest_rel(all_sims_12[l], all_sims_12_primes[l])
        ttest_p_values.append(p)

    def sig_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return ''

    sig_labels = [sig_stars(p) for p in ttest_p_values]

    #save
    model_name_save = model_name.replace('/', '_')
    np.save(f'data/CosSim_12_{model_name_save}_test_with_difference.npy', CosSim_12)
    np.save(f'data/CosSim_13_{model_name_save}_test_with_difference.npy', CosSim_13)
    np.save(f'data/CosSim_23_{model_name_save}_test_with_difference.npy', CosSim_23)
    np.save(f'data/CosSim_12_primes_{model_name_save}_test_with_difference.npy', CosSim_12_primes)
    np.save(f'data/CosSim_13_primes_{model_name_save}_test_with_difference.npy', CosSim_13_primes)
    np.save(f'data/CosSim_23_primes_{model_name_save}_test_with_difference.npy', CosSim_23_primes)
    np.save(f'data/CosSim_12_diff_{model_name_save}_test_with_difference.npy', CosSim_12_diff)
    np.save(f'data/CosSim_12_primes_diff_{model_name_save}_test_with_difference.npy', CosSim_12_primes_diff)
    # Raw per-sentence arrays (shape: num_layers × num_sentences) needed for paired t-tests
    np.save(f'data/all_sims_12_{model_name_save}_test_with_difference.npy', np.array(all_sims_12))
    np.save(f'data/all_sims_12_primes_{model_name_save}_test_with_difference.npy', np.array(all_sims_12_primes))
    # # np.save(f'data/Euclidean_12_{model_name_save}_test_with_primes.npy', Euclidean_12)
    # np.save(f'data/Euclidean_13_{model_name_save}_test_with_primes.npy', Euclidean_13)
    # np.save(f'data/Euclidean_23_{model_name_save}_test_with_primes.npy', Euclidean_23)
    # np.save(f'data/Euclidean_12_primes_{model_name_save}_test_with_primes.npy', Euclidean_12_primes)
    # np.save(f'data/Euclidean_13_primes_{model_name_save}_test_with_primes.npy', Euclidean_13_primes)
    # np.save(f'data/Euclidean_23_primes_{model_name_save}_test_with_primes.npy', Euclidean_23_primes)
    # subplot three lines, cossims and then primes
    model_name = model_name_map[model_name]

    fig, axes = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=True)

    sig_x = [l for l, lbl in enumerate(sig_labels) if lbl]
    sig_marker_handle = None
    for ax in [axes[0], axes[1]]:
        if sig_x:
            h, = ax.plot(sig_x, [0.0] * len(sig_x), marker='*', linestyle='none',
                         color='red', markersize=5, label='p<0.05 (paired t-test)', zorder=5)
            sig_marker_handle = h

    axes[0].plot(CosSim_12, label='<SB–SA> - <SB–SC>')
    axes[0].set_title('Original')
    axes[0].legend(loc='best')

    axes[1].plot(CosSim_12_primes, label="<SB'–SA'> - <SB'–SC'>")
    axes[1].set_title('Primes')
    axes[1].legend(loc='best')

    fig.supxlabel('Layer')
    fig.supylabel('Average Cosine Similarity Difference')
    fig.suptitle(f'Average Cosine Similarity Difference for {model_name}')
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_twoplot.png')
    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_twoplot.eps')
    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_twoplot.pdf')
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT))
    ax = fig.gca()
    ax.plot(CosSim_12, label='<SB–SA> - <SB–SC>', color='deepskyblue')
    ax.plot(CosSim_12_primes, label="<SB'–SA'> - <SB'–SC'>", color='chartreuse')

    # Asterisks at y=0 for paired t-test significance
    sig_x = [l for l, lbl in enumerate(sig_labels) if lbl]
    if sig_x:
        ax.plot(sig_x, [0.0] * len(sig_x), marker='*', linestyle='none',
                color='red', markersize=5, label='p<0.05 (paired t-test)', zorder=5)

    ax.legend(loc='best')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Cosine Similarity Difference')
    ax.set_title(f'Average Cosine Similarity Difference for {model_name}')
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])
    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_oneplot.png')
    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_oneplot.eps')
    fig.savefig(f'figures/CosineSimilarityDifference_{model_name}_test_oneplot.pdf')
    plt.show()





# plot all six models in one plot


# CosSim_12_finance_Llama = np.load(f'data/CosSim_12_tarun7r_Finance-Llama-8B_test_with_difference.npy')
# CosSim_12_bio_Llama = np.load(f'data/CosSim_12_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
# CosSim_12_llama = np.load(f'data/CosSim_12_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')
# CosSim_12_finance_BERT = np.load(f'data/CosSim_12_marcev_financebert_test_with_difference.npy')
# CosSim_12_bio_BERT = np.load(f'data/CosSim_12_dmis-lab_biobert-base-cased-v1.2_test_with_difference.npy')
# CosSim_12_llama_BERT = np.load(f'data/CosSim_12_google_multiberts-seed_3_test_with_difference.npy')
# CosSim_12_finance_Llama_diff = np.load(f'data/CosSim_12_diff_tarun7r_Finance-Llama-8B_test_with_difference.npy')
# CosSim_12_bio_Llama_diff = np.load(f'data/CosSim_12_diff_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
# CosSim_12_llama_diff = np.load(f'data/CosSim_12_diff_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')
# CosSim_12_finance_BERT_diff = np.load(f'data/CosSim_12_diff_marcev_financebert_test_with_difference.npy')
# CosSim_12_bio_BERT_diff = np.load(f'data/CosSim_12_diff_dmis-lab_biobert-base-cased-v1.2_test_with_difference.npy')
# CosSim_12_llama_BERT_diff = np.load(f'data/CosSim_12_diff_google_multiberts-seed_3_test_with_difference.npy')
# fig, axes = plt.subplots(2, 3, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=True)

# axes[0, 0].plot(CosSim_12_finance_Llama_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[0, 0].set_title('Finance-Llama')
# axes[0, 0].legend(loc='best')
# axes[0, 1].plot(CosSim_12_bio_Llama_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[0, 1].set_title('Bio-Medical-Llama')
# axes[0, 1].legend(loc='best')
# axes[0, 2].plot(CosSim_12_llama_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[0, 2].set_title('Llama')
# axes[0, 2].legend(loc='best')

# axes[1, 0].plot(CosSim_12_finance_BERT_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[1, 0].set_title('FinanceBERT')
# axes[1, 0].legend(loc='best')
# axes[1, 1].plot(CosSim_12_bio_BERT_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[1, 1].set_title('BioBERT')
# axes[1, 1].legend(loc='best')
# axes[1, 2].plot(CosSim_12_llama_BERT_diff, label="<SA–SB> - <SA–SC> - <SB–SC>")
# axes[1, 2].set_title('MultiBERTs')
# axes[1, 2].legend(loc='best')

# fig.supxlabel('Layer')
# fig.supylabel('Average Cosine Similarity')
# fig.suptitle('Average Difference in Cosine Similarity <SA–SB> - <SA–SC> - <SB–SC>')
# fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

# fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_all_models_test_with_difference.png')
# fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_all_models_test_with_difference.eps')
# fig.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_all_models_test_with_difference.pdf')
# plt.show()
# plt.close(fig)






# compare the cosine similarity and euclidean distance for the original and primes for llama

CosSim_12_finance = np.load(f'data/CosSim_12_tarun7r_Finance-Llama-8B_test_with_difference.npy')
CosSim_12_bio = np.load(f'data/CosSim_12_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
CosSim_12_llama = np.load(f'data/CosSim_12_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')

CosSim_12_finance_primes = np.load(f'data/CosSim_12_primes_tarun7r_Finance-Llama-8B_test_with_difference.npy')
CosSim_12_bio_primes = np.load(f'data/CosSim_12_primes_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
CosSim_12_llama_primes = np.load(f'data/CosSim_12_primes_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')


# Paired t-test per layer: Bio-Medical-Llama vs base Llama
# Requires raw per-sentence arrays (shape: num_layers × num_sentences)
all_sims_12_bio   = np.load('data/all_sims_12_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
all_sims_12_llama = np.load('data/all_sims_12_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')
all_sims_12_bio_primes   = np.load('data/all_sims_12_primes_ContactDoctor_Bio-Medical-Llama-3-8B_test_with_difference.npy')
all_sims_12_llama_primes = np.load('data/all_sims_12_primes_meta-llama_Meta-Llama-3-8B_test_with_difference.npy')

# Each row is one layer; each column is one sentence pair.
# One-tailed test: alternative='greater' tests H1: mean(BioLlama) > mean(Llama).
layer_x = list(range(all_sims_12_llama.shape[0]))
ttest_p_values = [ttest_rel(all_sims_12_bio[l], all_sims_12_llama[l], alternative='greater').pvalue for l in layer_x]
sig_labels = [sig_stars(p) for p in ttest_p_values]

ttest_p_values_primes = [ttest_rel(all_sims_12_bio_primes[l], all_sims_12_llama_primes[l], alternative='greater').pvalue for l in layer_x]
sig_labels_primes = [sig_stars(p) for p in ttest_p_values_primes]

# plot

fig, axes = plt.subplots(1, 2, figsize=(ACL_TEXT_WIDTH, ACL_SINGLE_HEIGHT), sharey=True)

# Significance markers at y=0 before data lines so data lines render on top
sig_x_orig   = [l for l, lbl in enumerate(sig_labels) if lbl]
sig_x_primes = [l for l, lbl in enumerate(sig_labels_primes) if lbl]
if sig_x_orig:
    axes[0].plot(sig_x_orig, [0.0] * len(sig_x_orig), marker='*', linestyle='none',
                 color='red', markersize=5, label='BioLlama > Llama', zorder=5)
if sig_x_primes:
    axes[1].plot(sig_x_primes, [0.0] * len(sig_x_primes), marker='*', linestyle='none',
                 color='red', markersize=5, label='BioLlama > Llama', zorder=5)

axes[0].plot(CosSim_12_finance, label="Finance-Llama-8B", color='deepskyblue')
axes[0].plot(CosSim_12_bio, label="Bio-Medical-Llama-3-8B", color='mediumpurple')
axes[0].plot(CosSim_12_llama, label="Llama-3.2-3B", color='chartreuse')
axes[0].set_title('Original')
# axes[0].legend(loc='best')

axes[1].plot(CosSim_12_finance_primes, label="Finance-Llama-8B", color='deepskyblue')
axes[1].plot(CosSim_12_bio_primes, label="Bio-Medical-Llama-3-8B", color='mediumpurple')
axes[1].plot(CosSim_12_llama_primes, label="Llama-3.2-3B", color='chartreuse')
axes[1].set_title('Primes')
axes[1].legend(loc='best')


fig.supxlabel('Layer')
fig.supylabel('Average Cosine Similarity Difference')
fig.suptitle('Average Cosine Similarity Difference for Llama-3.2-3B models')
# fig.suptitle('Cosine Similarity and Euclidean Distance for Original and Primes')
fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

fig.savefig(f'figures/CosineSimilarityDifferenceLlama.eps')
fig.savefig(f'figures/CosineSimilarityDifferenceLlama.png')
fig.savefig(f'figures/CosineSimilarityDifferenceLlama.pdf')
plt.show()
plt.close(fig)

