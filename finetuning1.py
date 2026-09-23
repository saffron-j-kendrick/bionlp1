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
import gc
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
    """Compute accuracy from the next-token probabilities of A, B, C, and D."""

    model.eval()
    correct = 0
    option_keys = ("A", "B", "C", "D")
    option_token_ids = {}
    for key in option_keys:
        token_ids = tokenizer.encode(f" {key}", add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Option label {key!r} is not a single token: {token_ids}"
            )
        option_token_ids[key] = token_ids[0]

    with torch.no_grad():
        for example in tqdm(dataset, desc="MCQA accuracy"):
            question   = example["question"]
            options    = example["options"]      # dict {A: text, B: text, ...}
            answer_idx = example["answer_idx"]   # e.g. "D"

            missing_options = set(option_keys).difference(options)
            if missing_options:
                raise ValueError(
                    f"Example is missing options: {sorted(missing_options)}"
                )

            opts_str = "\n".join(f"  {key}: {options[key]}" for key in option_keys)
            prefix = (
                "### Question:\n"
                f"{question}\n\n"
                "### Options:\n"
                f"{opts_str}\n\n"
                "### Answer:"
            )
            encoded = tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {
                key: value.to(model.device)
                for key, value in encoded.items()
            }
            outputs = model(**encoded)
            next_token_probabilities = F.softmax(
                outputs.logits[0, -1, :], dim=-1
            )
            letter_probabilities = {
                key: next_token_probabilities[token_id].item()
                for key, token_id in option_token_ids.items()
            }
            predicted = max(letter_probabilities, key=letter_probabilities.get)
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

def make_args(output_dir, epochs=1, batch_size=4, max_steps=None, seed=42):
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
        seed=seed,
        data_seed=seed,
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

SPECIFIC_LAYERS  = [0, 3, 14, 15, 16]   # edit to choose any indices 0–31
NUM_LAYERS       = 32                  
N_BOUNDARY_LAYERS = 5                      
FIRST_LAYERS     = list(range(N_BOUNDARY_LAYERS))                            # [0,1,2,3,4]
LAST_LAYERS      = list(range(NUM_LAYERS - N_BOUNDARY_LAYERS, NUM_LAYERS))   # [27,28,29,30,31]
RUN_SEEDS        = [0, 1, 2, 3, 4]
N_RANDOM_LAYERS  = 5

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

def free_memory():
    """Aggressively release GPU and CPU memory between experiments."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def load_base_model():
    """
    Load Llama in full bfloat16 precision — no quantisation.

    Caps the GPU allocation to 90 % of available VRAM so that device_map="auto"
    never falls back to meta-device (CPU offload) after earlier experiments have
    fragmented the CUDA allocator.  Meta-device parameters cause gradient errors
    during LoRA backpropagation.
    """
    free_memory()
    free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    max_gpu_gb = int(free_vram * 0.90 / 1024**3)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{max_gpu_gb}GiB", "cpu": "48GiB"},
    )
    # Verify no layers landed on meta device — that would break LoRA backprop
    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        raise RuntimeError(
            f"{len(meta_params)} parameter(s) are on meta device — not enough VRAM to load "
            "the model without CPU offloading.  Reduce batch size, use quantisation, or free "
            f"GPU memory before loading.  First offloaded param: {meta_params[0]}"
        )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model

def set_run_seed(seed):
    """Seed model initialisation, data ordering, and training stochasticity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_lora_repeats(experiment_name, layers_to_transform, random_layers=False):
    """Train one LoRA strategy five times and aggregate metrics/trajectories."""
    run_results = []
    trajectories = []

    for seed in RUN_SEEDS:
        set_run_seed(seed)
        if random_layers:
            layer_rng = random.Random(seed)
            chosen_layers = sorted(
                layer_rng.sample(range(NUM_LAYERS), N_RANDOM_LAYERS)
            )
        else:
            chosen_layers = (
                None
                if layers_to_transform is None
                else list(layers_to_transform)
            )

        layer_description = (
            "all layers" if chosen_layers is None else f"layers {chosen_layers}"
        )
        print(f"\n{experiment_name}, seed {seed}  →  {layer_description}")

        model = load_base_model()
        lora_cfg = make_lora_config(layers_to_transform=chosen_layers)
        output_stem = f"{experiment_name}_seed{seed}"
        trainer = make_trainer(
            model,
            make_args(f"checkpoints/{output_stem}", seed=seed),
            peft_cfg=lora_cfg,
        )
        count_trainable_params(trainer.model)
        trainer.train()

        model_dir = f"models/{output_stem}"
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        answer_loss = compute_answer_token_loss(
            trainer.model, tokenizer, test_dataset
        )
        accuracy = compute_mcqa_accuracy(
            trainer.model, tokenizer, test_dataset
        )
        run_results.append({
            "answer_token_loss": answer_loss,
            "mcqa_accuracy": accuracy,
            "seed": seed,
            "layers_chosen": (
                list(range(NUM_LAYERS))
                if chosen_layers is None
                else chosen_layers
            ),
        })
        trajectories.append(
            extract_trajectory(trainer.state.log_history)
        )
        print(
            f"  answer-token loss: {answer_loss:.4f}  |  "
            f"MCQA accuracy: {accuracy:.4f}"
        )

        del trainer, model
        free_memory()

    losses = [result["answer_token_loss"] for result in run_results]
    accuracies = [result["mcqa_accuracy"] for result in run_results]
    aggregate = {
        "answer_token_loss": float(np.mean(losses)),
        "answer_token_loss_std": float(np.std(losses)),
        "mcqa_accuracy": float(np.mean(accuracies)),
        "mcqa_accuracy_std": float(np.std(accuracies)),
        "per_seed": run_results,
    }
    trajectory_average = average_trajectories(trajectories)

    print(
        f"\n{experiment_name} (mean ± std across {len(RUN_SEEDS)} runs):"
    )
    print(
        f"  answer-token loss: {aggregate['answer_token_loss']:.4f} "
        f"± {aggregate['answer_token_loss_std']:.4f}"
    )
    print(
        f"  accuracy:          {aggregate['mcqa_accuracy']:.4f} "
        f"± {aggregate['mcqa_accuracy_std']:.4f}"
    )
    return aggregate, trajectory_average


## EXPERIMENTS — FIVE RUNS PER LoRA STRATEGY

results_lora_full, traj_full_avg = run_lora_repeats(
    "lora_full", layers_to_transform=None
)
results_lora_specific, traj_specific_avg = run_lora_repeats(
    "lora_specific", layers_to_transform=SPECIFIC_LAYERS
)
results_lora_first, traj_first_avg = run_lora_repeats(
    "lora_first", layers_to_transform=FIRST_LAYERS
)
results_lora_last, traj_last_avg = run_lora_repeats(
    "lora_last", layers_to_transform=LAST_LAYERS
)
results_lora_random, traj_random_avg = run_lora_repeats(
    "lora_random", layers_to_transform=None, random_layers=True
)

## RESULTS SUMMARY

summary = {
    "lora_full":     results_lora_full,
    "lora_specific": results_lora_specific,
    "lora_first":    results_lora_first,
    "lora_last":     results_lora_last,
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

labels = [
    f"Full LoRA\n(mean of {len(RUN_SEEDS)} runs)",
    f"Specific Layers\n{SPECIFIC_LAYERS}\n(mean of {len(RUN_SEEDS)} runs)",
    f"First {N_BOUNDARY_LAYERS} Layers\n{FIRST_LAYERS}\n(mean of {len(RUN_SEEDS)} runs)",
    f"Last {N_BOUNDARY_LAYERS} Layers\n{LAST_LAYERS}\n(mean of {len(RUN_SEEDS)} runs)",
    f"Random Layers\n(mean of {len(RUN_SEEDS)} runs)",
]
losses     = [r["answer_token_loss"]     for r in summary.values()]
accuracies = [r["mcqa_accuracy"]         for r in summary.values()]
loss_errs  = [r.get("answer_token_loss_std", 0) for r in summary.values()]
acc_errs   = [r.get("mcqa_accuracy_std",    0)  for r in summary.values()]
colours    = sns.color_palette("muted", 5)

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 5))

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

traj_colours = {
    "Full LoRA":           "#4C72B0",
    "Specific Layers":     "#DD8452",
    "First Layers":        "#8172B2",
    "Last Layers":         "#C44E52",
    "Random Layers (avg)": "#55A868",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax_train, ax_eval = axes

experiment_trajs = [
    ("Full LoRA",           traj_full_avg),
    ("Specific Layers",     traj_specific_avg),
    ("First Layers",        traj_first_avg),
    ("Last Layers",         traj_last_avg),
    ("Random Layers (avg)", traj_random_avg),
]

for label, traj_avg in experiment_trajs:
    col = traj_colours[label]
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

ax_train.set_xlabel("Training Step")
ax_train.set_ylabel("Training Loss")
ax_train.set_title("Training Loss Trajectory")
ax_train.legend(loc="upper right")
sns.despine(ax=ax_train)

ax_eval.set_xlabel("Training Step")
ax_eval.set_ylabel("Eval Loss (full sequence, trainer internal)")
ax_eval.set_title("Validation Loss Trajectory (full-seq)\n(shaded = ±1 std across runs)")
ax_eval.legend(loc="upper right")
sns.despine(ax=ax_eval)

fig.suptitle("Llama-3-8B  |  MedQA  |  LoRA Training Trajectories", fontsize=12)
plt.tight_layout()
plt.savefig("figures/finetuning1_trajectories.png", dpi=150)
plt.savefig("figures/finetuning1_trajectories.pdf")
plt.close()
print("Trajectory figure saved to figures/finetuning1_trajectories.{png,pdf}")
