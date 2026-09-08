## Paired comparison figure
## Six subplots – each shows the combined similarity difference
## (<SB-SA>-<SB-SC>) + (<SB'-SA'>-<SB'-SC'>) for two models overlaid.
##
## Significance markers: paired t-test (p-value) between the two models'
## per-sentence scores at each layer.  Asterisks are drawn between the curves.

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import ttest_rel

## ── Layout constants ──────────────────────────────────────────────────────
ACL_TEXT_WIDTH = 7.00
ACL_RSA_HEIGHT = 4.50

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

## ── Data loaders ──────────────────────────────────────────────────────────

def stem(model_id):
    return model_id.replace('/', '_')

def load_mean_diff(model_id):
    """Mean combined diff per layer — shape (num_layers,)."""
    path = f'data/CosSim_12_full_diff_{stem(model_id)}_test_with_difference.npy'
    return np.load(path) if os.path.exists(path) else None

def load_per_sentence(model_id):
    """Per-sentence combined diff — shape (num_layers, num_sentences)."""
    path = f'data/all_sim_diff_full_{stem(model_id)}_test_with_difference.npy'
    return np.load(path) if os.path.exists(path) else None

## ── Significance helpers ──────────────────────────────────────────────────

def sig_stars(p):
    """Convert a raw p-value to an asterisk string."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def paired_pvalues(per_sent_1, per_sent_2):
    """
    Paired t-test between two models at each layer.
    Tests H0: mean(model1[layer]) == mean(model2[layer]).
    Both arrays must have the same shape (num_layers, num_sentences).
    Returns an array of raw p-values, one per layer.
    """
    assert per_sent_1.shape == per_sent_2.shape, (
        f"Shape mismatch: {per_sent_1.shape} vs {per_sent_2.shape}"
    )
    num_layers = per_sent_1.shape[0]
    pvals = np.zeros(num_layers)
    for l in range(num_layers):
        _, p = ttest_rel(per_sent_1[l], per_sent_2[l])
        pvals[l] = p
    return pvals

## ── Pairs ─────────────────────────────────────────────────────────────────

PAIRS = [
    (
        'meta-llama/Meta-Llama-3-8B',          'Llama-3-8B',            'steelblue',
        'aaditya/Llama3-OpenBioLLM-8B',        'OpenBioLLM-8B',         'darkorange',
    ),
    (
        'meta-llama/Meta-Llama-3-8B-Instruct', 'Llama-3-8B-Instruct',   'steelblue',
        'ContactDoctor/Bio-Medical-Llama-3-8B','Bio-Medical-Llama-3-8B', 'darkorange',
    ),
    (
        'google-bert/bert-base-uncased',        'BERT-base-uncased',     'steelblue',
        'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12', 'BlueBERT-PubMed', 'darkorange',
    ),
    (
        'microsoft/Phi-3.5-mini-instruct',      'Phi-3.5-mini-instruct', 'steelblue',
        'microsoft/MediPhi-Instruct',           'MediPhi-Instruct',      'darkorange',
    ),
    (
        'mistralai/Mistral-7B-Instruct-v0.1',  'Mistral-7B-Instruct',   'steelblue',
        'BioMistral/BioMistral-7B',            'BioMistral-7B',         'darkorange',
    ),
    (
        'google-bert/bert-base-cased',          'BERT-base-cased',       'steelblue',
        'dmis-lab/biobert-base-cased-v1.2',    'BioBERT',               'darkorange',
    ),
]

## ── Build figure ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    2, 3,
    figsize=(ACL_TEXT_WIDTH * 1.5, ACL_RSA_HEIGHT * 1.2),
    constrained_layout=True,
)

for ax, (id1, lbl1, col1, id2, lbl2, col2) in zip(axes.flatten(), PAIRS):
    mean1 = load_mean_diff(id1)
    mean2 = load_mean_diff(id2)
    ps1   = load_per_sentence(id1)
    ps2   = load_per_sentence(id2)

    if mean1 is None and mean2 is None:
        ax.text(0.5, 0.5, 'Data not yet available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='grey')
        ax.set_title(f'{lbl1}  vs  {lbl2}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Avg Combined Cosine Sim Diff')
        continue

    # Plot each model's mean curve
    for mean, lbl, col in [(mean1, lbl1, col1), (mean2, lbl2, col2)]:
        if mean is None:
            continue
        layers = list(range(1, len(mean) + 1))
        ax.plot(layers, mean, label=lbl, color=col)

    # Paired t-test between the two models per layer
    # Only possible when both per-sentence arrays exist and have matching shapes
    if ps1 is not None and ps2 is not None:
        if ps1.shape == ps2.shape:
            pvals = paired_pvalues(ps1, ps2)
            layers = list(range(1, ps1.shape[0] + 1))

            # Draw asterisks at y=0 for each significant layer
            for l_idx, (layer, p) in enumerate(zip(layers, pvals)):
                stars = sig_stars(p)
                if stars:
                    ax.text(layer, 0.0, stars, ha='center', va='center',
                            fontsize=7, color='black', fontweight='bold', zorder=6)
        else:
            print(f"Warning: shape mismatch for {lbl1} vs {lbl2} "
                  f"({ps1.shape} vs {ps2.shape}) — skipping paired test.")

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_title(f'{lbl1}  vs  {lbl2}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Combined Cosine Sim Diff')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    # Add significance legend entry once per subplot
    ax.plot([], [], marker='$*$', linestyle='none', color='black',
            markersize=6, label='p<0.05 (paired t-test)')
    ax.legend(loc='best', framealpha=0.7)

fig.suptitle(
    "Combined Cosine Similarity Difference by Layer — General vs Biomedical Models",
    fontsize=8, y=1.02,
)

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/PairedComparison_combined_diff.png', dpi=150, bbox_inches='tight')
fig.savefig('figures/PairedComparison_combined_diff.pdf',           bbox_inches='tight')
fig.savefig('figures/PairedComparison_combined_diff.eps',           bbox_inches='tight')
plt.show()
print("Saved to figures/PairedComparison_combined_diff.{png,pdf,eps}")
