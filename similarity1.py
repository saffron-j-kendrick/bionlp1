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



access_token = os.environ.get('HF_TOKEN')

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
                elif model_name in ['meta-llama/Llama-3.2-1B', 'microsoft/phi-1', 'openai-community/gpt2', 'microsoft/biogpt', 'medicalai/ClinicalGPT-base-zh', 'meta-llama/Llama-3.2-3B', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "tiiuae/Falcon3-7B-Base", 'google/multiberts-seed_3', 'FacebookAI/roberta-base']:
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


abbr_1 = "AAA"
# sentence_1 = "OBJECTIVE: abdominal aortic aneurysm represents a chronic degenerative condition associated with atherosclerosis."
sentence_1a = "METHODS: The Canadian Institute for Health Information database (a collection of all acute care hospitalizations) was reviewed to identify patients who received nonemergent repair of an abdominal aortic aneurysm between April 1, 2003 and March 31, 2004."
sentence_1b = "METHODS: The Canadian Institute for Health Information database (a collection of all acute care hospitalizations) was reviewed to identify patients who received nonemergent repair of an AAA  between April 1, 2003 and March 31, 2004."
sentence_1c = "METHODS: The Canadian Institute for Health Information database (a collection of all acute care hospitalizations) was reviewed to identify patients who received nonemergent repair of an BBB  between April 1, 2003 and March 31, 2004."

sentence_1a = remove_punctuation(sentence_1a)
sentence_1b = remove_punctuation(sentence_1b)
sentence_1c = remove_punctuation(sentence_1c)

target_1 = "between"

sentence_2a = "BACKGROUND: A number of the research into the pathogenesis of the abdominal aortic aneurysm have focused on the alteration of gene expression."
sentence_2b = "BACKGROUND: A number of the research into the pathogenesis of the AAA have focused on the alteration of gene expression."
sentence_2c = "BACKGROUND: A number of the research into the pathogenesis of the BBB have focused on the alteration of gene expression."

sentence_2a = remove_punctuation(sentence_2a)
sentence_2b = remove_punctuation(sentence_2b)
sentence_2c = remove_punctuation(sentence_2c)

target_2 = "have"

sentence_3a = "OBJECTIVE: We prospectively studied the clinical implication of plasma level of soluble fibrin monomer (FM)-fibrinogen complex, a recently established molecular marker reflecting thrombin activity, in patients with abdominal aortic aneurysm undergoing elective aortic repair."
sentence_3b = "OBJECTIVE: We prospectively studied the clinical implication of plasma level of soluble fibrin monomer (FM)-fibrinogen complex, a recently established molecular marker reflecting thrombin activity, in patients with AAA undergoing elective aortic repair."
sentence_3c = "OBJECTIVE: We prospectively studied the clinical implication of plasma level of soluble fibrin monomer (FM)-fibrinogen complex, a recently established molecular marker reflecting thrombin activity, in patients with BBB undergoing elective aortic repair."

sentence_3a = remove_punctuation(sentence_3a)
sentence_3b = remove_punctuation(sentence_3b)
sentence_3c = remove_punctuation(sentence_3c)

target_3 = "undergoing"

sentence_4a = "PURPOSE: To use an in vitro flow model to investigate the flow patterns in a bifurcated stent-graft for abdominal aortic aneurysm ( AAA ) repair."
sentence_4b = "PURPOSE: To use an in vitro flow model to investigate the flow patterns in a bifurcated stent-graft for AAA repair."
sentence_4c = "PURPOSE: To use an in vitro flow model to investigate the flow patterns in a bifurcated stent-graft for BBB repair."

sentence_4a = remove_punctuation(sentence_4a)
sentence_4b = remove_punctuation(sentence_4b)
sentence_4c = remove_punctuation(sentence_4c)

target_4 = "repair"

sentence_5a = "OBJECTIVE: To evaluate the results of our experience in the management of patients with symptomatic, unruptured abdominal aortic aneurysm ( AAA ), to identify the predictors of immediate outcome and to define the worldwide postoperative mortality rate through a review of previous studies on this condition."
sentence_5b = "OBJECTIVE: To evaluate the results of our experience in the management of patients with symptomatic, unruptured AAA, to identify the predictors of immediate outcome and to define the worldwide postoperative mortality rate through a review of previous studies on this condition."
sentence_5c = "OBJECTIVE: To evaluate the results of our experience in the management of patients with symptomatic, unruptured BBB, to identify the predictors of immediate outcome and to define the worldwide postoperative mortality rate through a review of previous studies on this condition."

sentence_5a = remove_punctuation(sentence_5a)
sentence_5b = remove_punctuation(sentence_5b)
sentence_5c = remove_punctuation(sentence_5c)

target_5 = "to"

### EMBEDDINGS ###


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch_device = torch.device("cpu")

# model_name = "openai-community/gpt2"
# dev_model_configs = {'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2')}

# model_name = "microsoft/biogpt"
# dev_model_configs = {'microsoft/biogpt' : (AutoConfig.from_pretrained("microsoft/biogpt"), AutoModelForCausalLM.from_pretrained("microsoft/biogpt"), AutoTokenizer.from_pretrained("microsoft/biogpt"), 'microsoft/biogpt')}

# model_name = "medicalai/ClinicalGPT-base-zh"
# dev_model_configs = {'medicalai/ClinicalGPT-base-zh' : (AutoConfig.from_pretrained("medicalai/ClinicalGPT-base-zh"), AutoModelForCausalLM.from_pretrained("medicalai/ClinicalGPT-base-zh"), AutoTokenizer.from_pretrained("medicalai/ClinicalGPT-base-zh"), 'medicalai/ClinicalGPT-base-zh')}

# model_name = "meta-llama/Llama-3.2-3B"
# dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B')}

dev_model_configs = {'meta-llama/Llama-3.2-3B' : (AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token), AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token = access_token) , 'meta-llama/Llama-3.2-3B'),
                    'microsoft/biogpt' : (AutoConfig.from_pretrained("microsoft/biogpt", token = access_token), AutoModelForCausalLM.from_pretrained("microsoft/biogpt", token = access_token), AutoTokenizer.from_pretrained("microsoft/biogpt", token = access_token), 'microsoft/biogpt'),
                    'openai-community/gpt2' : (AutoConfig.from_pretrained("openai-community/gpt2"), AutoModelForCausalLM.from_pretrained("openai-community/gpt2"), AutoTokenizer.from_pretrained("openai-community/gpt2"), 'openai-community/gpt2'),
                    'Qwen/Qwen2.5-7B' : (AutoConfig.from_pretrained("Qwen/Qwen2.5-7B"), AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"), AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"), 'Qwen/Qwen2.5-7B'),
                    'mistralai/Mistral-7B-v0.1' : (AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1"), AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"), 'mistralai/Mistral-7B-v0.1'),
                    'tiiuae/Falcon3-7B-Base' : (AutoConfig.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base"), AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base"), 'tiiuae/Falcon3-7B-Base'),
                    'google/multiberts-seed_3' : (AutoConfig.from_pretrained("google/multiberts-seed_3"), AutoModelForCausalLM.from_pretrained("google/multiberts-seed_3"), AutoTokenizer.from_pretrained("google/multiberts-seed_3"), 'google/multiberts-seed_3'),
                    'FacebookAI/roberta-base' : (AutoConfig.from_pretrained("FacebookAI/roberta-base"), AutoModelForCausalLM.from_pretrained("FacebookAI/roberta-base"), AutoTokenizer.from_pretrained("FacebookAI/roberta-base"), 'FacebookAI/roberta-base')}
                    
models = dev_model_configs.keys()

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
    
    #sentence a
    sentence_a_embeddings = []
    for sentence in [sentence_1a, sentence_2a, sentence_3a, sentence_4a, sentence_5a]:
        if sentence == sentence_1a:
            target_word = target_1
        elif sentence == sentence_2a:
            target_word = target_2
        elif sentence == sentence_3a:
            target_word = target_3
        elif sentence == sentence_4a:
            target_word = target_4
        elif sentence == sentence_5a:
            target_word = target_5
        inputs_1 = tokeniser(sentence,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_1 = inputs_1["input_ids"]
        attention_mask_1 = inputs_1["attention_mask"]
        embeddings = get_target_token_embeddings(model_name, model, tokeniser, input_ids_1, attention_mask_1, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_word)
        sentence_a_embeddings.append(embeddings)
    sentence_a_avg_embeddings = [np.mean(np.vstack(emb), axis=0) for emb in sentence_a_embeddings]

    #sentence b
    sentence_b_embeddings = []
    for sentence in [sentence_1b, sentence_2b, sentence_3b, sentence_4b, sentence_5b]:
        if sentence == sentence_1b:
            target_word = target_1
        elif sentence == sentence_2b:
            target_word = target_2
        elif sentence == sentence_3b:
            target_word = target_3
        elif sentence == sentence_4b:
            target_word = target_4
        elif sentence == sentence_5b:
            target_word = target_5
        inputs_2 = tokeniser(sentence,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_2 = inputs_2["input_ids"]
        attention_mask_2 = inputs_2["attention_mask"]
        embeddings = get_target_token_embeddings(model_name, model, tokeniser, input_ids_2, attention_mask_2, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_word)
        sentence_b_embeddings.append(embeddings)
    sentence_b_avg_embeddings = [np.mean(np.vstack(emb), axis=0) for emb in sentence_b_embeddings]

    #sentence c
    sentence_c_embeddings = []
    for sentence in [sentence_1c, sentence_2c, sentence_3c, sentence_4c, sentence_5c]:
        if sentence == sentence_1c:
            target_word = target_1
        elif sentence == sentence_2c:
            target_word = target_2
        elif sentence == sentence_3c:
            target_word = target_3
        elif sentence == sentence_4c:
            target_word = target_4
        elif sentence == sentence_5c:
            target_word = target_5
        inputs_3 = tokeniser(sentence,  max_length = 512, return_tensors="pt", truncation=True, padding=True)
        input_ids_3 = inputs_3["input_ids"]
        attention_mask_3 = inputs_3["attention_mask"]
        embeddings = get_target_token_embeddings(model_name, model, tokeniser, input_ids_3, attention_mask_3, layers, torch_device, batch_size=1, middle_dim=None, target_word=target_word)
        sentence_c_embeddings.append(embeddings)
    sentence_c_avg_embeddings = [np.mean(np.vstack(emb), axis=0) for emb in sentence_c_embeddings]

    # The embeddings are now a list of numpy arrays, each with shape (num_tokens, embedding_dim)
    # The length of the list is the number of layers in the model

    layers = range(1, model.config.num_hidden_layers + 1)
    CosSim_12 = []
    CosSim_13 = []
    CosSim_23 = []
    PearsonCorr_12 = []
    PearsonCorr_13 = []
    PearsonCorr_23 = []
   

    for layer_idx in range(len(layers)):
        sims_12, sims_13, sims_23 = [], [], []
        for sent_idx in range(len(sentence_a_embeddings)):
            emb_a = sentence_a_embeddings[sent_idx][layer_idx]  # shape (1, hidden_size)
            emb_b = sentence_b_embeddings[sent_idx][layer_idx]
            emb_c = sentence_c_embeddings[sent_idx][layer_idx]
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
    plt.title(f'Average Cosine Similarity using the target token embeddings for {model_name}')
    plt.legend()
    plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_BBB.png')
    plt.savefig(f'figures/CosineSimilarityFinalTokenEmbeddings_{model_name}_BBB.eps')
    plt.show()
    plt.close()
