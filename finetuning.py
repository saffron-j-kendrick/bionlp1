## IMPORTS

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import random
from tqdm.autonotebook import tqdm

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

## ── CUDA CHECK ────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA device found. This script requires a GPU. "
        "Check your environment with: python -c \"import torch; print(torch.cuda.is_available())\""
    )
print(f"Using GPU: {torch.cuda.get_device_name(0)}  "
      f"(VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

## ── TOKENIZER ─────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

## ── DATA ──────────────────────────────────────────────────────────────────

raw_dataset = load_dataset("awinml/medqa")
# Splits: train / validation / test
# Columns: question | options (dict A/B/C/D) | answer_idx | answer

def plot_distribution(token_counts, title):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    plt.hist(token_counts, bins=50, color='#3498db', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel("Number of tokens", fontsize=14)
    plt.ylabel("Number of examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/{title}.png")
    plt.close()

question_token_counts = [len(tokenizer.tokenize(ex["question"])) for ex in raw_dataset['train']]
answer_token_counts   = [len(tokenizer.tokenize(ex["answer"]))   for ex in raw_dataset['train']]
combined_token_counts = [q + a for q, a in zip(question_token_counts, answer_token_counts)]

plot_distribution(question_token_counts, "Distribution of token counts for question only")
plot_distribution(answer_token_counts,   "Distribution of token counts for answer only")
plot_distribution(combined_token_counts, "Distribution of token counts for combined question + answer")

## ── PROMPT FORMATTING ─────────────────────────────────────────────────────

def format_prompt(example):
    opts = example["options"]
    options_str = "\n".join(f"  {k}: {v}" for k, v in opts.items())
    text = (
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Options:\n"
        f"{options_str}\n\n"
        "### Answer:\n"
        f"{example['answer_idx']}: {example['answer']}"
    )
    return {"text": text}

formatted    = raw_dataset.map(format_prompt)
train_dataset = formatted["train"]
val_dataset   = formatted["validation"]
test_dataset  = formatted["test"]   # retains all original columns + "text"

## ── LAYER UTILITIES ───────────────────────────────────────────────────────

def set_all_layers_trainable(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_specific_layers(model, layer_indices):
    freeze_all_layers(model)
    layers = model.model.layers
    num_layers = len(layers)
    valid, skipped = [], []
    for i in layer_indices:
        if 0 <= i < num_layers:
            valid.append(i)
        else:
            skipped.append(i)
    if skipped:
        print(f"Warning: indices {skipped} out of range (0–{num_layers-1}), skipped.")
    print(f"Training layers: {sorted(valid)}")
    for i in valid:
        for param in layers[i].parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

def unfreeze_random_n_layers(model, n=5, seed=42):
    random.seed(seed)
    freeze_all_layers(model)
    layers = model.model.layers
    chosen = random.sample(range(len(layers)), n)
    print(f"Randomly selected layers: {sorted(chosen)}")
    for i in chosen:
        for param in layers[i].parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")

## ── MCQA ACCURACY ─────────────────────────────────────────────────────────

def compute_mcqa_accuracy(model, tokenizer, dataset, max_length=512):
    """
    Score each answer option (A/B/C/D) by the log-likelihood the model assigns
    to the option text conditioned on the question + options prefix.
    The predicted answer is the option with the highest log-likelihood.
    Returns accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="MCQA accuracy"):
            question   = example["question"]
            options    = example["options"]      # dict {A: text, B: text, ...}
            answer_idx = example["answer_idx"]   # e.g. "D"

            opts_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
            prefix = (
                "### Question:\n"
                f"{question}\n\n"
                "### Options:\n"
                f"{opts_str}\n\n"
                "### Answer:\n"
            )
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            prefix_len = len(prefix_ids)

            scores = {}
            for key, text in options.items():
                full_text = prefix + f"{key}: {text}"
                full_ids  = tokenizer.encode(
                    full_text, add_special_tokens=True,
                    truncation=True, max_length=max_length,
                )
                input_tensor = torch.tensor([full_ids], device=model.device)

                outputs = model(input_tensor)
                logits  = outputs.logits  # (1, seq_len, vocab_size)

                # Score only the suffix tokens (the option text after the prefix)
                suffix_start = min(prefix_len - 1, logits.shape[1] - 1)
                suffix_logits = logits[0, suffix_start:-1, :]        # (suffix_len, vocab)
                suffix_ids    = input_tensor[0, suffix_start + 1:]   # (suffix_len,)

                if suffix_ids.numel() == 0:
                    scores[key] = float('-inf')
                    continue

                log_probs = F.log_softmax(suffix_logits, dim=-1)
                score = log_probs[range(len(suffix_ids)), suffix_ids].sum().item()
                scores[key] = score

            predicted = max(scores, key=scores.get)
            if predicted == answer_idx:
                correct += 1

    return correct / len(dataset)

## ── TRAINING CONFIG ───────────────────────────────────────────────────────

def make_args(output_dir, epochs=1, batch_size=4, max_steps=None):
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="tensorboard",
        dataset_text_field="text",
        max_length=512,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return SFTConfig(**kwargs)

def make_trainer(model, args, peft_cfg=None):
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

## ── EXPERIMENT 1 — FULL FINE-TUNING ──────────────────────────────────────

print("\n=== Experiment 1: Full fine-tuning (all layers) ===")
model_full = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model_full.config.pad_token_id = tokenizer.eos_token_id
set_all_layers_trainable(model_full)
count_trainable_params(model_full)

trainer_full = make_trainer(model_full, make_args("checkpoints/full_layers"))
trainer_full.train()
trainer_full.save_model("models/full_layers")

results_full         = trainer_full.evaluate(test_dataset)
accuracy_full        = compute_mcqa_accuracy(model_full, tokenizer, test_dataset)
results_full["mcqa_accuracy"] = accuracy_full
print(f"Full FT  →  loss: {results_full['eval_loss']:.4f}  |  MCQA accuracy: {accuracy_full:.4f}")

del model_full
torch.cuda.empty_cache()

## ── EXPERIMENT 2 — TARGETED LAYERS ───────────────────────────────────────

print("\n=== Experiment 2: Targeted layer fine-tuning ===")
TARGET_LAYERS = [10, 11, 12, 13, 14]   # edit to choose any indices 0–31

model_top5 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model_top5.config.pad_token_id = tokenizer.eos_token_id
unfreeze_specific_layers(model_top5, TARGET_LAYERS)
count_trainable_params(model_top5)

trainer_top5 = make_trainer(model_top5, make_args("checkpoints/top5_layers"))
trainer_top5.train()
trainer_top5.save_model("models/top5_layers")

results_top5         = trainer_top5.evaluate(test_dataset)
accuracy_top5        = compute_mcqa_accuracy(model_top5, tokenizer, test_dataset)
results_top5["mcqa_accuracy"] = accuracy_top5
print(f"Top-5    →  loss: {results_top5['eval_loss']:.4f}  |  MCQA accuracy: {accuracy_top5:.4f}")

del model_top5
torch.cuda.empty_cache()

## ── EXPERIMENT 3 — RANDOM-5 LAYERS ───────────────────────────────────────

print("\n=== Experiment 3: Random-5 layer fine-tuning ===")
model_rand5 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model_rand5.config.pad_token_id = tokenizer.eos_token_id
unfreeze_random_n_layers(model_rand5, n=5, seed=42)
count_trainable_params(model_rand5)

trainer_rand5 = make_trainer(model_rand5, make_args("checkpoints/rand5_layers"))
trainer_rand5.train()
trainer_rand5.save_model("models/rand5_layers")

results_rand5         = trainer_rand5.evaluate(test_dataset)
accuracy_rand5        = compute_mcqa_accuracy(model_rand5, tokenizer, test_dataset)
results_rand5["mcqa_accuracy"] = accuracy_rand5
print(f"Random-5 →  loss: {results_rand5['eval_loss']:.4f}  |  MCQA accuracy: {accuracy_rand5:.4f}")

del model_rand5
torch.cuda.empty_cache()

## ── EXPERIMENT 4 — LoRA (QLoRA 4-bit) ────────────────────────────────────

print("\n=== Experiment 4: LoRA fine-tuning (QLoRA 4-bit) ===")
model_lora = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BNB_CONFIG,
    device_map={"": 0},
)
model_lora.config.pad_token_id = tokenizer.eos_token_id
model_lora = prepare_model_for_kbit_training(model_lora)
count_trainable_params(model_lora)

trainer_lora = make_trainer(model_lora, make_args("checkpoints/lora"), peft_cfg=LORA_CONFIG)
trainer_lora.train()
trainer_lora.model.save_pretrained("models/lora")
tokenizer.save_pretrained("models/lora")

results_lora         = trainer_lora.evaluate(test_dataset)
accuracy_lora        = compute_mcqa_accuracy(trainer_lora.model, tokenizer, test_dataset)
results_lora["mcqa_accuracy"] = accuracy_lora
print(f"LoRA     →  loss: {results_lora['eval_loss']:.4f}  |  MCQA accuracy: {accuracy_lora:.4f}")

del model_lora
torch.cuda.empty_cache()

## ── RESULTS SUMMARY ───────────────────────────────────────────────────────

summary = {
    "full_finetune": results_full,
    "top5_layers":   results_top5,
    "rand5_layers":  results_rand5,
    "lora":          results_lora,
}

with open("results/finetuning1_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Test-set Results ===")
print(f"{'Experiment':<20} {'Loss':>8} {'MCQA Acc':>10}")
print("-" * 42)
for name, res in summary.items():
    print(f"{name:<20} {res.get('eval_loss', 0):>8.4f} {res.get('mcqa_accuracy', 0):>10.4f}")

## ── PLOT: two side-by-side subplots ──────────────────────────────────────

labels     = ["Full FT", "Top-5 Layers", "Random-5 Layers", "LoRA (QLoRA)"]
losses     = [r.get("eval_loss",      0) for r in summary.values()]
accuracies = [r.get("mcqa_accuracy",  0) for r in summary.values()]
colours    = sns.color_palette("muted", 4)

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

bars = ax_loss.bar(labels, losses, color=colours, edgecolor="black")
ax_loss.bar_label(bars, fmt="%.4f", padding=3)
ax_loss.set_ylabel("Test Evaluation Loss (Cross-Entropy)")
ax_loss.set_title("Test-set Loss by Fine-tuning Strategy")
ax_loss.set_ylim(0, max(losses) * 1.2)
sns.despine(ax=ax_loss)

bars = ax_acc.bar(labels, accuracies, color=colours, edgecolor="black")
ax_acc.bar_label(bars, fmt="%.4f", padding=3)
ax_acc.set_ylabel("MCQA Accuracy (proportion correct)")
ax_acc.set_title("MCQA Answer Accuracy by Fine-tuning Strategy")
ax_acc.set_ylim(0, min(max(accuracies) * 1.2, 1.0))
sns.despine(ax=ax_acc)

fig.suptitle("Llama-3-8B  |  MedQA", fontsize=12)
plt.tight_layout()
plt.savefig("figures/finetuning1_results.png", dpi=150)
plt.savefig("figures/finetuning1_results.pdf")
plt.close()
print("Figure saved to figures/finetuning1_results.{png,pdf}")
