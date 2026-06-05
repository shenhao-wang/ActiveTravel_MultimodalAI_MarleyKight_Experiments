"""
Word clouds for BH-significant embedding dimensions.
Word size = |TF-IDF correlation|, color = pole (teal/terracotta).
Separate figures for MiniLM and OpenAI.

VERIFIED SOURCE FILES:
  Dims/thetas: dml_backend_comparison.csv (May 2026)
  TF-IDF terms: zone_embeddings_minilm_tfidf_interpretation.csv (May 13 2026)
                zone_embeddings_openai-lg_tfidf_interpretation.csv (May 13 2026)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

TEAL       = '#2A7F7F'
TERRACOTTA = '#C75B39'
NAVY       = '#1B2A4A'
GREY_LIGHT = '#D0D0D0'

# ── PATHS ────────────────────────────────────────────────────
mini = pd.read_csv('processed_data/zone_embeddings/zone_embeddings_minilm_tfidf_interpretation.csv')
oai  = pd.read_csv('processed_data/zone_embeddings/zone_embeddings_openai-lg_tfidf_interpretation.csv')

# Only plot dimensions with causal evidence (BH-corrected p < 0.05).
# Non-significant dims would add noise without interpretive value.
# Source: dml_backend_comparison.csv
# MiniLM matches the counterfactual pipeline embedding. OpenAI has the most
# coherent TF-IDF correlations (highest max |r| per dim).
mini_sig = ['emb_018', 'emb_023', 'emb_028', 'emb_030']
oai_sig  = ['emb_010', 'emb_012', 'emb_018', 'emb_024', 'emb_026', 'emb_030']

mini_thetas = {
    'emb_018': +0.0631,
    'emb_023': +0.0813,
    'emb_028': +0.0668,
    'emb_030': -0.0684,
}
oai_thetas = {
    'emb_010': -0.1125,
    'emb_012': -0.1572,
    'emb_018': -0.1107,
    'emb_024': +0.1116,
    'emb_026': -0.1011,
    'emb_030': -0.0869,
}


def get_terms(row):
    """Extract positive and negative terms with absolute correlations."""
    pos_terms = {}
    neg_terms = {}
    for i in range(1, 7):
        col_t = f'pos_term_{i}'
        col_c = f'pos_corr_{i}'
        if col_t in row.index and pd.notna(row[col_t]):
            pos_terms[str(row[col_t]).strip()] = abs(float(row[col_c]))
    for i in range(1, 7):
        col_t = f'neg_term_{i}'
        col_c = f'neg_corr_{i}'
        if col_t in row.index and pd.notna(row[col_t]):
            neg_terms[str(row[col_t]).strip()] = abs(float(row[col_c]))
    return pos_terms, neg_terms


def make_bicolor_cloud(pos_terms, neg_terms, ax, title, theta, label):
    """Create a single word cloud with positive terms in teal, negative in terracotta."""
    all_terms = {}
    term_colors = {}
    for term, weight in pos_terms.items():
        all_terms[term] = weight
        term_colors[term] = TEAL
    for term, weight in neg_terms.items():
        all_terms[term] = weight
        term_colors[term] = TERRACOTTA

    if not all_terms:
        ax.text(0.5, 0.5, 'No terms', ha='center', va='center',
                transform=ax.transAxes)
        return

    def color_func(word, **kwargs):
        return term_colors.get(word, '#888888')

    # Word size proportional to |TF-IDF correlation| so the strongest regulatory
    # terms dominate visually. max_words=12 avoids clutter.
    wc = WordCloud(
        width=500, height=300,
        background_color='white',
        max_words=12,
        relative_scaling=0.7,
        min_font_size=10,
        max_font_size=60,
        color_func=color_func,
        prefer_horizontal=0.8,
    ).generate_from_frequencies(all_terms)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    sign = '+' if theta > 0 else ''
    direction = r'$\rightarrow$ AT' if theta > 0 else r'$\rightarrow$ car'
    ax.set_title(
        f'{title}\n{label}  (\u03b8 = {sign}{theta:.3f}, {direction})',
        fontsize=8, fontweight='bold', pad=4
    )


# ── Legend elements ──────────────────────────────────────────
legend_elements = [
    Patch(color=TEAL, label='Positive pole (+corr)'),
    Patch(color=TERRACOTTA, label='Negative pole (\u2212corr)'),
]


# ── FIGURE 1: MiniLM (4 BH-significant dims, 2×2) ───────────
fig_m, axes_m = plt.subplots(2, 2, figsize=(10, 6))
axes_m = axes_m.flatten()

for idx, dim in enumerate(mini_sig):
    row = mini[mini['dimension'] == dim].iloc[0]
    pos_terms, neg_terms = get_terms(row)
    theta = mini_thetas[dim]
    label = row['label']
    make_bicolor_cloud(pos_terms, neg_terms, axes_m[idx], dim, theta, label)

fig_m.legend(handles=legend_elements, loc='lower center', ncol=2,
             framealpha=0.95, fontsize=9, edgecolor=GREY_LIGHT,
             bbox_to_anchor=(0.5, -0.02))
fig_m.suptitle(
    'Panel A: MiniLM-L6-v2 \u2014 BH-significant embedding dimensions\n'
    'Word size \u221d |TF-IDF correlation|',
    fontweight='bold', fontsize=11
)
fig_m.tight_layout(rect=[0, 0.03, 1, 0.93])
fig_m.savefig(f'{BASE}/fig_wordcloud_minilm.png', dpi=300)
fig_m.savefig(f'{BASE}/fig_wordcloud_minilm.pdf')
plt.close(fig_m)
print('Saved: fig_wordcloud_minilm.png / .pdf')


# ── FIGURE 2: OpenAI (6 BH-significant dims, 2×3) ───────────
fig_o, axes_o = plt.subplots(2, 3, figsize=(14, 6))
axes_o = axes_o.flatten()

for idx, dim in enumerate(oai_sig):
    row = oai[oai['dimension'] == dim].iloc[0]
    pos_terms, neg_terms = get_terms(row)
    theta = oai_thetas[dim]
    label = row['label']
    make_bicolor_cloud(pos_terms, neg_terms, axes_o[idx], dim, theta, label)

fig_o.legend(handles=legend_elements, loc='lower center', ncol=2,
             framealpha=0.95, fontsize=9, edgecolor=GREY_LIGHT,
             bbox_to_anchor=(0.5, -0.02))
fig_o.suptitle(
    'Panel B: OpenAI text-embedding-3-large \u2014 BH-significant embedding dimensions\n'
    'Word size \u221d |TF-IDF correlation|',
    fontweight='bold', fontsize=11
)
fig_o.tight_layout(rect=[0, 0.03, 1, 0.93])
fig_o.savefig(f'{BASE}/fig_wordcloud_openai.png', dpi=300)
fig_o.savefig(f'{BASE}/fig_wordcloud_openai.pdf')
plt.close(fig_o)
print('Saved: fig_wordcloud_openai.png / .pdf')
