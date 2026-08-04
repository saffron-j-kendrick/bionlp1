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



access_token = os.environ.get('HF_TOKEN_LLAMA')

if access_token is None:
    raise ValueError("HF_TOKEN is not set")

### FUNCTIONS ###


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
    print(f'Extracting representations from model for layers {layers}')
    # Move data to the torch device (cpu or gpu)
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
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base', 'dmis-lab/biobert-base-cased-v1.2']:
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

### SENTENCES ###


abbr_dataset = pd.read_excel("triple_sentence_test_set.xlsx")
sentence_a_embeddings = []
sentence_b_embeddings = []
sentence_c_embeddings = []
abbrs = []
targets = []

for i in range(len(abbr_dataset)):
    sentence_a_embeddings.append(abbr_dataset.iloc[i]['sentence_a'])
    sentence_b_embeddings.append(abbr_dataset.iloc[i]['sentence_b'])
    sentence_c_embeddings.append(abbr_dataset.iloc[i]['sentence_c'])
    abbrs.append(abbr_dataset.iloc[i]['abbr'])
    targets.append(abbr_dataset.iloc[i]['target'])
 

print(sentence_a_embeddings)


### MODELS ###

dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B'),
                    'microsoft/biogpt' : (AutoConfig.from_pretrained("microsoft/biogpt", token = access_token), AutoModelForCausalLM.from_pretrained("microsoft/biogpt", token = access_token), AutoTokenizer.from_pretrained("microsoft/biogpt", token = access_token), 'microsoft/biogpt'),
                    'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2'),
                    'Qwen/Qwen2.5-7B' : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), 'Qwen/Qwen2.5-7B'),
                    'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"), 'mistralai/Mistral-7B-v0.1'),
                    'tiiuae/Falcon3-7B-Base' : (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), 'tiiuae/Falcon3-7B-Base'),
                    'google/multiberts-seed_3' : (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForMaskedLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), 'google/multiberts-seed_3'),
                    'FacebookAI/roberta-base' : (AutoConfig.from_pretrained("FacebookAI/roberta-base"), AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base"), AutoTokenizer.from_pretrained("FacebookAI/roberta-base"), 'FacebookAI/roberta-base'),
                    'dmis-lab/biobert-base-cased-v1.2' : (AutoConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), 'dmis-lab/biobert-base-cased-v1.2')}
                    
models = dev_model_configs.keys()
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for model_name in tqdm.tqdm(models):
    print('Loading {}'.format(model_name))
    model, tokeniser = load_model(model_name)
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

    for i in range(len(sentence_a_embeddings)):
        sent_a = sentence_a_embeddings[i]
        sent_b = sentence_b_embeddings[i]
        sent_c = sentence_c_embeddings[i]
        target = targets[i]

        inputs_a = tokeniser(sent_a,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_a = inputs_a["input_ids"]
        attention_mask_a = inputs_a["attention_mask"]
        embeddings_a = get_target_token_embeddings(model_name, model, tokeniser, input_ids_a, attention_mask_a, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
        sentence_a_embs.append(embeddings_a)

        inputs_b = tokeniser(sent_b,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_b = inputs_b["input_ids"]
        attention_mask_b = inputs_b["attention_mask"]
        embeddings_b = get_target_token_embeddings(model_name, model, tokeniser, input_ids_b, attention_mask_b, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
        sentence_b_embs.append(embeddings_b)
        inputs_c = tokeniser(sent_c,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_c = inputs_c["input_ids"]
        attention_mask_c = inputs_c["attention_mask"]
        embeddings_c = get_target_token_embeddings(model_name, model, tokeniser, input_ids_c, attention_mask_c, layers, torch_device, batch_size=1, middle_dim=None, target_word=target)
        sentence_c_embs.append(embeddings_c)
    sentence_a_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_a_embs]
    sentence_b_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_b_embs]
    sentence_c_avg_embs = [np.mean(np.vstack(emb), axis=0) for emb in sentence_c_embs]

    layers = range(1, model.config.num_hidden_layers + 1)
    CosSim_12 = []
    CosSim_13 = []
    CosSim_23 = []
    PearsonCorr_12 = []
    PearsonCorr_13 = []
    PearsonCorr_23 = []
   

    for layer_idx in range(len(layers)):
        sims_12, sims_13, sims_23 = [], [], []
        for sent_idx in range(len(sentence_a_embs)):
            emb_a = sentence_a_embs[sent_idx][layer_idx]  # shape (1, hidden_size)
            emb_b = sentence_b_embs[sent_idx][layer_idx]
            emb_c = sentence_c_embs[sent_idx][layer_idx]
            sims_12.append(cosine_similarity(emb_a, emb_b)[0][0])
            sims_13.append(cosine_similarity(emb_a, emb_c)[0][0])
            sims_23.append(cosine_similarity(emb_b, emb_c)[0][0])
        
    
        CosSim_12.append(np.mean(sims_12))
        CosSim_13.append(np.mean(sims_13))
        CosSim_23.append(np.mean(sims_23))
    
    # plot three lines in one plot
    model_name = model_name.replace('/', '_')
    plt.figure(figsize=(10, 5))
    plt.plot(CosSim_12, label='Average Cosine Similarity between Sentence 1 and Sentence 2')
    plt.plot(CosSim_13, label='Average Cosine Similarity between Sentence 1 and Sentence 3')
    plt.plot(CosSim_23, label='Average Cosine Similarity between Sentence 2 and Sentence 3')
    plt.xlabel('Layer')
    plt.ylabel('Average Cosine Similarity')
    plt.title(f'Average Cosine Similarity using the target token embeddings for {model_name}')
    plt.legend()
    plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test.png')
    plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_test.eps')
    plt.show()
    plt.close()

  