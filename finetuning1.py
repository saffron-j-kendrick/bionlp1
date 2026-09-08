## IMPORTS

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
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

## CUDA CHECK

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA device found. This script requires a GPU. "
        "Check your environment with: python -c \"import torch; print(torch.cuda.is_available())\""
    )
print(f"Using GPU: {torch.cuda.get_device_name(0)}  "
      f"(VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

## TOKENIZER

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

## DATA

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

## PROMPT FORMATTING

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

## LAYER UTILITIES

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

## MCQA ACCURACY

def compute_mcqa_accuracy(model, tokenizer, dataset, max_length=512):
    """Compute MCQA accuracy for options (A/B/C/D) by the log-likelihood the model assigns to the option text conditioned on the question + options. The predicted answer is the option with the highest log-likelihood."""
   
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

## ENTROPY LOSS 

def compute_answer_token_loss(model, tokenizer, dataset, max_length=512):
    """ Cross-entropy loss computed only over the answer tokens, averaged across the test set."""
    model.eval()
    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="Answer-token loss"):
            question   = example["question"]
            options    = example["options"]
            answer_idx = example["answer_idx"]
            answer_text = example["answer"]

            opts_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
            prefix = (
                "### Question:\n"
                f"{question}\n\n"
                "### Options:\n"
                f"{opts_str}\n\n"
                "### Answer:\n"
            )
            full_text = prefix + f"{answer_idx}: {answer_text}"

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            full_ids   = tokenizer.encode(
                full_text, add_special_tokens=True,
                truncation=True, max_length=max_length,
            )

            prefix_len  = len(prefix_ids)
            suffix_len  = len(full_ids) - prefix_len
            if suffix_len <= 0:
                continue

            input_tensor = torch.tensor([full_ids], device=model.device)
            outputs      = model(input_tensor)
            logits       = outputs.logits  # (1, seq_len, vocab_size)

            # Logits at position i predict token i+1.
            # Suffix starts at prefix_len in full_ids, so we want
            # logits[prefix_len-1 : prefix_len-1+suffix_len] to predict full_ids[prefix_len:]
            suffix_start  = min(prefix_len - 1, logits.shape[1] - 1)
            suffix_logits = logits[0, suffix_start : suffix_start + suffix_len, :]
            suffix_targets = input_tensor[0, suffix_start + 1 : suffix_start + 1 + suffix_len]

            if suffix_targets.numel() == 0:
                continue

            loss = F.cross_entropy(suffix_logits, suffix_targets, reduction="sum")
            total_loss   += loss.item()
            total_tokens += suffix_targets.numel()

    return total_loss / total_tokens if total_tokens > 0 else float("nan")

## TRAJECTORY HELPERS

def extract_trajectory(log_history):
    """ Returns four lists: train_steps, train_losses, eval_steps, eval_losses."""
    train_steps, train_losses = [], []
    eval_steps,  eval_losses  = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])
    return train_steps, train_losses, eval_steps, eval_losses

def average_trajectories(trajectories):
    
    # Training curve
    all_train_steps  = [t[0] for t in trajectories]
    common_train     = sorted(set(all_train_steps[0]).intersection(*[set(s) for s in all_train_steps]))
    step_to_losses   = {s: [] for s in common_train}
    for t_steps, t_losses, _, _ in trajectories:
        lookup = dict(zip(t_steps, t_losses))
        for s in common_train:
            if s in lookup:
                step_to_losses[s].append(lookup[s])
    avg_train_steps  = common_train
    avg_train_losses = [np.mean(step_to_losses[s]) for s in common_train]
    std_train_losses = [np.std(step_to_losses[s])  for s in common_train]

    # Eval curve
    all_eval_steps  = [t[2] for t in trajectories]
    common_eval     = sorted(set(all_eval_steps[0]).intersection(*[set(s) for s in all_eval_steps]))
    step_to_elosses = {s: [] for s in common_eval}
    for _, _, e_steps, e_losses in trajectories:
        lookup = dict(zip(e_steps, e_losses))
        for s in common_eval:
            if s in lookup:
                step_to_elosses[s].append(lookup[s])
    avg_eval_steps  = common_eval
    avg_eval_losses = [np.mean(step_to_elosses[s]) for s in common_eval]
    std_eval_losses = [np.std(step_to_elosses[s])  for s in common_eval]

    return (avg_train_steps, avg_train_losses, std_train_losses,
            avg_eval_steps,  avg_eval_losses,  std_eval_losses)

## TRAINING CONFIG

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
        optim="adamw_torch",
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

LORA_TARGET_MODULES = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']

SPECIFIC_LAYERS  = [11, 12, 13, 19, 31]   # edit to choose any indices 0–31
RANDOM_SEEDS     = [0, 1, 2, 3, 4]        # five seeds for the random experiment
N_RANDOM_LAYERS  = 5
NUM_LAYERS       = 32                      # Llama-3-8B transformer blocks

def make_lora_config(layers_to_transform=None):
    """
    Build a LoraConfig.  Pass a list of layer indices to restrict adapters
    to those layers only; pass None for full LoRA across all layers.
    """
    kwargs = dict(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = layers_to_transform
    return LoraConfig(**kwargs)

def load_base_model():
    """Load Llama in full bfloat16 precision — no quantisation."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model

## EXPERIMENT 1 — FULL LoRA (all layers)

print("\n=== Experiment 1: Full LoRA (adapters on all layers) ===")
model_lora_full = load_base_model()
lora_config_full = make_lora_config(layers_to_transform=None)
count_trainable_params(model_lora_full)

trainer_lora_full = make_trainer(model_lora_full, make_args("checkpoints/lora_full"), peft_cfg=lora_config_full)
trainer_lora_full.train()
trainer_lora_full.model.save_pretrained("models/lora_full")
tokenizer.save_pretrained("models/lora_full")

# results_lora_full = trainer_lora_full.evaluate(test_dataset)  # full-sequence loss (commented out)
traj_full          = extract_trajectory(trainer_lora_full.state.log_history)
ans_loss_full      = compute_answer_token_loss(trainer_lora_full.model, tokenizer, test_dataset)
accuracy_lora_full = compute_mcqa_accuracy(trainer_lora_full.model, tokenizer, test_dataset)
results_lora_full  = {"answer_token_loss": ans_loss_full, "mcqa_accuracy": accuracy_lora_full}
print(f"Full LoRA   →  answer-token loss: {ans_loss_full:.4f}  |  MCQA accuracy: {accuracy_lora_full:.4f}")

del model_lora_full
torch.cuda.empty_cache()

## EXPERIMENT 2 — SPECIFIC-LAYER LoRA 

print(f"\n=== Experiment 2: Specific-layer LoRA (layers {SPECIFIC_LAYERS}) ===")
model_lora_specific = load_base_model()
lora_config_specific = make_lora_config(layers_to_transform=SPECIFIC_LAYERS)
count_trainable_params(model_lora_specific)

trainer_lora_specific = make_trainer(model_lora_specific, make_args("checkpoints/lora_specific"), peft_cfg=lora_config_specific)
trainer_lora_specific.train()
trainer_lora_specific.model.save_pretrained("models/lora_specific")
tokenizer.save_pretrained("models/lora_specific")

# results_lora_specific = trainer_lora_specific.evaluate(test_dataset)  # full-sequence loss (commented out)
traj_specific          = extract_trajectory(trainer_lora_specific.state.log_history)
ans_loss_specific      = compute_answer_token_loss(trainer_lora_specific.model, tokenizer, test_dataset)
accuracy_lora_specific = compute_mcqa_accuracy(trainer_lora_specific.model, tokenizer, test_dataset)
results_lora_specific  = {"answer_token_loss": ans_loss_specific, "mcqa_accuracy": accuracy_lora_specific}
print(f"Specific LoRA →  answer-token loss: {ans_loss_specific:.4f}  |  MCQA accuracy: {accuracy_lora_specific:.4f}")

del model_lora_specific
torch.cuda.empty_cache()

## EXPERIMENT 3 — RANDOM-LAYER LoRA

print(f"\n=== Experiment 3: Random-layer LoRA ({len(RANDOM_SEEDS)} seeds, {N_RANDOM_LAYERS} layers each) ===")

random_run_results = []   # list of dicts, one per seed
random_trajectories = []  # list of trajectory tuples, one per seed

for seed in RANDOM_SEEDS:
    random.seed(seed)
    chosen_layers = sorted(random.sample(range(NUM_LAYERS), N_RANDOM_LAYERS))
    print(f"\n  Seed {seed}  →  layers {chosen_layers}")

    model_rand = load_base_model()
    lora_cfg   = make_lora_config(layers_to_transform=chosen_layers)
    count_trainable_params(model_rand)

    trainer_rand = make_trainer(
        model_rand,
        make_args(f"checkpoints/lora_random_seed{seed}"),
        peft_cfg=lora_cfg,
    )
    trainer_rand.train()
    trainer_rand.model.save_pretrained(f"models/lora_random_seed{seed}")
    tokenizer.save_pretrained(f"models/lora_random_seed{seed}")

    # res = trainer_rand.evaluate(test_dataset)  # full-sequence loss (commented out)
    ans_loss_rand = compute_answer_token_loss(trainer_rand.model, tokenizer, test_dataset)
    acc           = compute_mcqa_accuracy(trainer_rand.model, tokenizer, test_dataset)
    res = {"answer_token_loss": ans_loss_rand, "mcqa_accuracy": acc,
           "seed": seed, "layers_chosen": chosen_layers}
    random_run_results.append(res)
    random_trajectories.append(extract_trajectory(trainer_rand.state.log_history))
    print(f"  Seed {seed}  →  answer-token loss: {ans_loss_rand:.4f}  |  MCQA accuracy: {acc:.4f}")

    del model_rand
    torch.cuda.empty_cache()

# Aggregate across seeds
rand_losses     = [r["answer_token_loss"] for r in random_run_results]
rand_accuracies = [r["mcqa_accuracy"]     for r in random_run_results]

results_lora_random = {
    "answer_token_loss":     float(np.mean(rand_losses)),
    "answer_token_loss_std": float(np.std(rand_losses)),
    "mcqa_accuracy":         float(np.mean(rand_accuracies)),
    "mcqa_accuracy_std":     float(np.std(rand_accuracies)),
    "per_seed":              random_run_results,
}
traj_random_avg = average_trajectories(random_trajectories)

print(f"\nRandom LoRA (mean ± std across {len(RANDOM_SEEDS)} seeds):")
print(f"  answer-token loss: {results_lora_random['answer_token_loss']:.4f} ± {results_lora_random['answer_token_loss_std']:.4f}")
print(f"  accuracy:          {results_lora_random['mcqa_accuracy']:.4f} ± {results_lora_random['mcqa_accuracy_std']:.4f}")

## RESULTS SUMMARY

summary = {
    "lora_full":     results_lora_full,
    "lora_specific": results_lora_specific,
    "lora_random":   results_lora_random,
}

with open("results/finetuning1_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Test-set Results ===")
print(f"{'Experiment':<20} {'Ans-Token Loss':>16} {'MCQA Acc':>12}")
print("-" * 52)
for name, res in summary.items():
    std_l = f" ±{res['answer_token_loss_std']:.4f}" if "answer_token_loss_std" in res else ""
    std_a = f" ±{res['mcqa_accuracy_std']:.4f}"     if "mcqa_accuracy_std"     in res else ""
    print(f"{name:<20} {res['answer_token_loss']:.4f}{std_l:>10} {res['mcqa_accuracy']:.4f}{std_a:>10}")

## FIGURES

labels     = ["Full LoRA", f"Specific Layers\n{SPECIFIC_LAYERS}", f"Random Layers\n(mean of {len(RANDOM_SEEDS)} seeds)"]
losses     = [r["answer_token_loss"]     for r in summary.values()]
accuracies = [r["mcqa_accuracy"]         for r in summary.values()]
loss_errs  = [r.get("answer_token_loss_std", 0) for r in summary.values()]
acc_errs   = [r.get("mcqa_accuracy_std",    0)  for r in summary.values()]
colours    = sns.color_palette("muted", 3)

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

bars = ax_loss.bar(labels, losses, color=colours, edgecolor="black", yerr=loss_errs,
                   capsize=5, error_kw={"elinewidth": 1.2})
ax_loss.bar_label(bars, fmt="%.4f", padding=6)
ax_loss.set_ylabel("Answer-Token Cross-Entropy Loss")
ax_loss.set_title("Answer-Token Loss by LoRA Strategy")
ax_loss.set_ylim(0, max(l + e for l, e in zip(losses, loss_errs)) * 1.25)
sns.despine(ax=ax_loss)

bars = ax_acc.bar(labels, accuracies, color=colours, edgecolor="black", yerr=acc_errs,
                  capsize=5, error_kw={"elinewidth": 1.2})
ax_acc.bar_label(bars, fmt="%.4f", padding=6)
ax_acc.set_ylabel("MCQA Accuracy (proportion correct)")
ax_acc.set_title("MCQA Answer Accuracy by LoRA Strategy")
ax_acc.set_ylim(0, min(max(a + e for a, e in zip(accuracies, acc_errs)) * 1.25, 1.0))
sns.despine(ax=ax_acc)

fig.suptitle("Llama-3-8B  |  MedQA  |  LoRA (bfloat16)  |  Loss = answer tokens only", fontsize=12)
plt.tight_layout()
plt.savefig("figures/finetuning1_results.png", dpi=150)
plt.savefig("figures/finetuning1_results.pdf")
plt.close()
print("Figure saved to figures/finetuning1_results.{png,pdf}")

traj_colours = {"Full LoRA": "#4C72B0", "Specific Layers": "#DD8452", "Random Layers (avg)": "#55A868"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_train, ax_eval = axes

experiment_trajs = [
    ("Full LoRA",          traj_full,     None),
    ("Specific Layers",    traj_specific, None),
    ("Random Layers (avg)", None,         traj_random_avg),
]

for label, traj, traj_avg in experiment_trajs:
    col = traj_colours[label]

    if traj_avg is not None:
        # averaged random: traj_avg = (tr_steps, tr_mean, tr_std, ev_steps, ev_mean, ev_std)
        tr_steps, tr_mean, tr_std, ev_steps, ev_mean, ev_std = traj_avg
        ax_train.plot(tr_steps, tr_mean, label=label, color=col)
        ax_train.fill_between(tr_steps,
                              [m - s for m, s in zip(tr_mean, tr_std)],
                              [m + s for m, s in zip(tr_mean, tr_std)],
                              alpha=0.2, color=col)
        ax_eval.plot(ev_steps, ev_mean, marker='o', markersize=3, label=label, color=col)
        ax_eval.fill_between(ev_steps,
                             [m - s for m, s in zip(ev_mean, ev_std)],
                             [m + s for m, s in zip(ev_mean, ev_std)],
                             alpha=0.2, color=col)
    else:
        tr_steps, tr_losses, ev_steps, ev_losses = traj
        ax_train.plot(tr_steps, tr_losses, label=label, color=col)
        ax_eval.plot(ev_steps, ev_losses, marker='o', markersize=3, label=label, color=col)

ax_train.set_xlabel("Training Step")
ax_train.set_ylabel("Training Loss")
ax_train.set_title("Training Loss Trajectory")
ax_train.legend(loc="upper right")
sns.despine(ax=ax_train)

ax_eval.set_xlabel("Training Step")
ax_eval.set_ylabel("Eval Loss (full sequence, trainer internal)")
ax_eval.set_title("Validation Loss Trajectory (full-seq)\n(shaded = ±1 std across random seeds)")
ax_eval.legend(loc="upper right")
sns.despine(ax=ax_eval)

fig.suptitle("Llama-3-8B  |  MedQA  |  LoRA Training Trajectories", fontsize=12)
plt.tight_layout()
plt.savefig("figures/finetuning1_trajectories.png", dpi=150)
plt.savefig("figures/finetuning1_trajectories.pdf")
plt.close()
print("Trajectory figure saved to figures/finetuning1_trajectories.{png,pdf}")
