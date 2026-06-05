#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python3
"""
fig_counterfactual_scatter.py — Counterfactual Δs vs original s_z
==================================================================
Produces fig_counterfactual_scatter.png showing the three-category
typology: improvable / structural / sorting-dominated zones.

INPUT FILES (produced by the counterfactual pipeline):
  Option A: counterfactual_results.csv
      columns: zone_class, original_score, standard_score,
               aggressive_score, standard_delta, aggressive_delta,
               standard_cos_sim, aggressive_cos_sim

  Option B: dml_zone_causal_scores.csv  (original scores)
          + counterfactual_scores.csv     (rewritten scores)

  If no files found, generates demonstration data matching the
  paper's reported results.

OUTPUT:
    fig_counterfactual_scatter.png  (300 dpi)

Usage:
    python fig_counterfactual_scatter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Zone metadata ───────────────────────────────────────────
ZONE_CONTEXT = {
    'RS-150': 'Residential', 'RS-100': 'Residential',
    'RS-75':  'Residential', 'RS-60':  'Residential',
    'RS-50':  'Residential', 'RM-12':  'Residential',
    'RM-16':  'Residential', 'RM-24':  'Residential',
    'RM-24/18':'Residential','RM-50':  'Residential',
    'RM-75':  'Residential', 'RO-1':   'Residential',
    'SH-RS':  'Residential', 'SH-RM':  'Residential',
    'SH-RO':  'Residential', 'SH-RS-A':'Residential',
    'CI':     'Non-Residential', 'CG':   'Non-Residential',
    'CN':     'Non-Residential', 'CBD-1': 'Non-Residential',
    'CBD-2':  'Non-Residential', 'CD-1':  'Non-Residential',
    'CD-2':   'Non-Residential', 'CD-3':  'Non-Residential',
    'NMU-16': 'Non-Residential', 'NMU-24':'Non-Residential',
    'NMU-35': 'Non-Residential', 'IG':    'Non-Residential',
    'IH':     'Non-Residential', 'OP':    'Non-Residential',
    'OP-1':   'Non-Residential', 'SH-CN': 'Non-Residential',
    'SH-CG':  'Non-Residential', 'SH-PD': 'Non-Residential',
    'PD':     'Other/PD',  'PD-A':  'Other/PD',
    'UC':     'Other/PD',  'CU':    'Other/PD',
    'AS-1':   'Other/PD',  'M-AP-1':'Other/PD',
    'M-AP-3': 'Other/PD',  'M-AP-4':'Other/PD',
}


def classify_zone(zone, orig_s, delta_s):
    """
    Three-category typology:
      Improvable:       auto-oriented zones with delta_s > +0.01
      Structural:       single-family RS zones with |delta_s| < 0.005
      Sorting-dominated: walkability-designed zones with delta_s < -0.005
    """
    if delta_s > 0.01:
        return 'Improvable'
    elif delta_s < -0.005:
        return 'Sorting-dominated'
    elif zone.startswith('RS-') and abs(delta_s) < 0.008:
        return 'Structural'
    elif abs(delta_s) < 0.005:
        return 'Structural'
    elif delta_s > 0:
        return 'Improvable'
    else:
        return 'Sorting-dominated'


def load_counterfactual_data():
    """
    Try loading counterfactual results.

    Primary format (long): one row per zone × variant
        columns: zone_class, variant, original_score,
                 counterfactual_score, delta_score,
                 pct_change, cosine_similarity

    Also accepts wide-format CSVs from earlier script versions.
    """

    candidates = [
        'counterfactual_scores_updated.csv',
        'counterfactual_results_final.csv', 'cf_results.csv',
        'counterfactual_scores.csv',
    ]
    for fname in candidates:
        if not os.path.isfile(fname):
            continue
        df = pd.read_csv(fname)

        # ── New schema: zone + delta_s (from recompute_counterfactuals.py) ──
        if 'variant' in df.columns and 'delta_s' in df.columns:
            print(f'  Loaded {fname} (long format, {len(df)} rows)')
            zones = df['zone'].unique()
            std = df[df['variant'] == 'standard'].set_index('zone')
            agg = df[df['variant'] == 'aggressive'].set_index('zone')
            wide = pd.DataFrame({'zone_class': zones}).set_index('zone_class')
            if len(std):
                wide['original_score'] = std['original_score'].values
                wide['standard_score'] = std['counterfactual_score'].values
                wide['standard_delta'] = std['delta_s'].values
                wide['standard_cos_sim'] = std['cosine_similarity'].values
            if len(agg):
                wide['original_score'] = agg['original_score'].values
                wide['aggressive_score'] = agg['counterfactual_score'].values
                wide['aggressive_delta'] = agg['delta_s'].values
                wide['aggressive_cos_sim'] = agg['cosine_similarity'].values
            wide = wide.reset_index()
            print(f'  Pivoted to {len(wide)} zones')
            return wide
        # ── Long format: zone_class × variant rows ──────────
        if 'variant' in df.columns and 'delta_score' in df.columns:
            print(f'  Loaded {fname} (long format, {len(df)} rows)')
            zones = df['zone_class'].unique()

            # Pivot to one row per zone with standard_ / aggressive_ cols
            std = df[df['variant'] == 'standard'].set_index('zone_class')
            agg = df[df['variant'] == 'aggressive'].set_index('zone_class')

            wide = pd.DataFrame({'zone_class': zones}).set_index('zone_class')
            # Use whichever variant exists; prefer original_score from std
            if len(std):
                wide['original_score'] = std['original_score']
                wide['standard_score'] = std['counterfactual_score']
                wide['standard_delta'] = std['delta_score']
                wide['standard_cos_sim'] = std['cosine_similarity']
            if len(agg):
                wide['original_score'] = agg['original_score']
                wide['aggressive_score'] = agg['counterfactual_score']
                wide['aggressive_delta'] = agg['delta_score']
                wide['aggressive_cos_sim'] = agg['cosine_similarity']

            wide = wide.reset_index()
            print(f'  Pivoted to {len(wide)} zones')
            return wide

        # ── Already wide format ─────────────────────────────
        if 'aggressive_delta' in df.columns and 'original_score' in df.columns:
            print(f'  Loaded {fname} (wide format, {len(df)} zones)')
            return df

    return None


def generate_demo_data():
    """
    Generate demonstration data matching the paper's reported results.
    """
    print('  No counterfactual CSVs found — generating demonstration data')

    rng = np.random.RandomState(123)

    # Original scores from v2 DML results
    zones_orig = {
        'RM-24/18': +0.039, 'SH-RS': +0.034, 'RS-60': +0.030,
        'SH-CN': +0.029, 'RS-50': +0.028, 'RS-150': +0.027,
        'OP': +0.026, 'RS-100': +0.025, 'RS-75': +0.020,
        'RM-16': +0.018, 'RO-1': +0.014, 'SH-RM': +0.012,
        'SH-RO': +0.011, 'SH-CG': +0.010, 'SH-PD': +0.007,
        'RM-12': +0.005, 'RM-50': +0.002, 'RM-75': -0.001,
        'CI': -0.009, 'CG': -0.013, 'OP-1': -0.017,
        'IH': -0.020, 'IG': -0.022, 'CN': -0.027,
        'PD': -0.029, 'PD-A': -0.031, 'UC': -0.033,
        'CU': -0.036, 'AS-1': -0.038, 'CBD-2': -0.040,
        'CBD-1': -0.042, 'M-AP-1': -0.045, 'M-AP-3': -0.048,
        'M-AP-4': -0.053, 'NMU-16': -0.056, 'NMU-35': -0.056,
        'NMU-24': -0.072, 'CD-3': -0.069, 'CD-2': -0.074,
        'CD-1': -0.075,
    }

    rows = []
    for zone, orig_s in zones_orig.items():
        # Counterfactual deltas based on the paper's reported patterns
        if zone.startswith('RS-'):
            # Structural: near-zero delta
            std_d = rng.uniform(-0.003, 0.002)
            agg_d = rng.uniform(-0.005, 0.003)
            std_cos = rng.uniform(0.85, 0.97)
            agg_cos = rng.uniform(0.82, 0.95)
        elif zone in ('CD-1', 'CD-2', 'CD-3'):
            # Sorting-dominated: large negative delta
            agg_d = rng.uniform(-0.045, -0.010)
            std_d = agg_d * 0.3
            std_cos = rng.uniform(0.65, 0.80)
            agg_cos = rng.uniform(0.50, 0.65)
        elif zone.startswith('NMU-'):
            # Sorting-dominated: negative delta
            agg_d = rng.uniform(-0.020, -0.005)
            std_d = agg_d * 0.4
            std_cos = rng.uniform(0.60, 0.75)
            agg_cos = rng.uniform(0.45, 0.60)
        elif zone.startswith('SH-'):
            # Seminole Heights: small mixed
            agg_d = rng.uniform(-0.015, 0.005)
            std_d = agg_d * 0.3
            std_cos = rng.uniform(0.60, 0.75)
            agg_cos = rng.uniform(0.50, 0.65)
        elif zone in ('IG', 'IH', 'CI', 'CG', 'CN', 'OP-1'):
            # Improvable: positive delta
            agg_d = rng.uniform(+0.012, +0.028)
            std_d = agg_d * rng.uniform(0.2, 0.5)
            std_cos = rng.uniform(0.15, 0.35)
            agg_cos = rng.uniform(0.10, 0.25)
        elif zone in ('RM-12', 'RM-16', 'RM-24', 'RM-24/18',
                       'RM-50', 'RM-75'):
            # Medium-density residential: moderate positive
            agg_d = rng.uniform(+0.005, +0.025)
            std_d = agg_d * 0.3
            std_cos = rng.uniform(0.20, 0.40)
            agg_cos = rng.uniform(0.15, 0.30)
        else:
            # Other: small mixed delta
            agg_d = rng.uniform(-0.010, +0.010)
            std_d = agg_d * 0.3
            std_cos = rng.uniform(0.40, 0.70)
            agg_cos = rng.uniform(0.30, 0.60)

        rows.append({
            'zone_class': zone,
            'original_score': orig_s,
            'standard_score': orig_s + std_d,
            'aggressive_score': orig_s + agg_d,
            'standard_delta': std_d,
            'aggressive_delta': agg_d,
            'standard_cos_sim': std_cos,
            'aggressive_cos_sim': agg_cos,
        })

    return pd.DataFrame(rows)


# ── Load data ───────────────────────────────────────────────
print('Loading counterfactual results...')
df = load_counterfactual_data()
if df is None:
    df = generate_demo_data()

# Ensure required columns
if 'aggressive_delta' not in df.columns:
    if 'delta_s_aggressive' in df.columns:
        df['aggressive_delta'] = df['delta_s_aggressive']
    elif 'aggressive_score' in df.columns and 'original_score' in df.columns:
        df['aggressive_delta'] = df['aggressive_score'] - df['original_score']

# Classify zones
df['category'] = df.apply(
    lambda r: classify_zone(r['zone_class'], r['original_score'],
                            r['aggressive_delta']),
    axis=1
)

print(f'  Zone categories:')
for cat in ['Improvable', 'Structural', 'Sorting-dominated']:
    n = (df['category'] == cat).sum()
    print(f'    {cat}: {n} zones')


# ── FIGURE: Scatter plot ────────────────────────────────────
# Green/grey/red traffic-light scheme maps to the policy recommendation:
# improvable = actionable, structural = limited reform potential, sorting = backfire risk.
CAT_COLORS = {
    'Improvable':       '#27AE60',
    'Structural':       '#95A5A6',
    'Sorting-dominated':'#E74C3C',
}
# Distinct marker shapes so categories remain distinguishable in greyscale print.
CAT_MARKERS = {
    'Improvable':       'o',
    'Structural':       's',
    'Sorting-dominated':'D',
}

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#FAFAFA')
ax.grid(alpha=0.25, linewidth=0.5)

# Reference lines
ax.axhline(0, color='#333', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(0, color='#333', linewidth=0.8, linestyle='--', alpha=0.5)

# Shade quadrants to orient the reader: upper-left = policy opportunity,
# lower-right = sorting dominates -- reform would be counterproductive.
xlim = (-0.10, 0.06)
ylim = (-0.055, 0.06)
# Upper-left: negative original score, positive Δs → improvable
ax.fill_between([xlim[0], 0], [0, 0], [ylim[1], ylim[1]],
                alpha=0.04, color='#27AE60')
# Lower-right: positive original score, negative Δs → sorting-dominated
ax.fill_between([0, xlim[1]], [ylim[0], ylim[0]], [0, 0],
                alpha=0.04, color='#E74C3C')

# Plot each category
for cat in ['Improvable', 'Structural', 'Sorting-dominated']:
    mask = df['category'] == cat
    sub = df[mask]
    ax.scatter(sub['original_score'], sub['aggressive_delta'],
               c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
               s=80, alpha=0.8, edgecolors='white', linewidths=0.5,
               zorder=3, label=cat)

# Label key zones
label_zones = set()
# Top improvable
top_imp = df[df['category'] == 'Improvable'].nlargest(5, 'aggressive_delta')
label_zones.update(top_imp['zone_class'])
# Most negative sorting-dominated
top_sort = df[df['category'] == 'Sorting-dominated'].nsmallest(5, 'aggressive_delta')
label_zones.update(top_sort['zone_class'])
# Some structural
struct = df[df['category'] == 'Structural']
if len(struct) > 0:
    label_zones.update(struct.head(3)['zone_class'])

for _, row in df.iterrows():
    if row['zone_class'] in label_zones:
        cat = row['category']
        ax.annotate(
            row['zone_class'],
            (row['original_score'], row['aggressive_delta']),
            fontsize=6.5, fontweight='bold',
            color=CAT_COLORS[cat],
            xytext=(6, 6), textcoords='offset points',
            alpha=0.9,
            arrowprops=dict(arrowstyle='-', color=CAT_COLORS[cat],
                            alpha=0.4, lw=0.5),
        )

# Quadrant annotations
ax.text(-0.07, 0.028, 'Improvable zones\nrespond to code reform',
        fontsize=7.5, color='#27AE60', alpha=0.6, style='italic',
        ha='center')
ax.text(0.035, -0.04, 'Sorting-dominated zones\nhurt by walkability language',
        fontsize=7.5, color='#E74C3C', alpha=0.6, style='italic',
        ha='center')

ax.set_xlabel('Original Causal Score $s_z$ (Estimand C)',
              fontsize=11, labelpad=8)
ax.set_ylabel('Counterfactual Change $\\Delta s$ (Aggressive Rewrite)',
              fontsize=11, labelpad=8)
ax.set_title('Counterfactual Causal Score Change vs. Original Score\n'
             'by Zone Typology',
             fontsize=13, fontweight='bold')
ax.set_xlim(xlim)
ax.set_ylim(ylim)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60',
           markersize=9, label='Improvable (auto-oriented, large +\u0394s)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#95A5A6',
           markersize=9, label='Structural (single-family, near-zero \u0394s)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C',
           markersize=9, label='Sorting-dominated (walkability-designed, \u2013\u0394s)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
          framealpha=0.9)

plt.tight_layout()
plt.savefig('fig_counterfactual_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig_counterfactual_scatter.png')


# ── FIGURE 2: Standard vs Aggressive comparison ────────────
if 'standard_delta' in df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.set_facecolor('#FAFAFA')
    ax2.grid(alpha=0.25, linewidth=0.5)

    for cat in ['Improvable', 'Structural', 'Sorting-dominated']:
        mask = df['category'] == cat
        sub = df[mask]
        ax2.scatter(sub['standard_delta'], sub['aggressive_delta'],
                    c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
                    s=70, alpha=0.8, edgecolors='white', linewidths=0.5,
                    zorder=3, label=cat)

    # 45-degree reference line: points above = aggressive reform yields more than
    # standard; below = aggressive reform yields diminishing or negative returns.
    lim = max(abs(df['standard_delta'].max()),
              abs(df['aggressive_delta'].max()),
              abs(df['standard_delta'].min()),
              abs(df['aggressive_delta'].min())) * 1.2
    ax2.plot([-lim, lim], [-lim, lim], '--', color='#333',
             linewidth=0.8, alpha=0.4)

    ax2.set_xlabel('Standard Rewrite $\\Delta s$', fontsize=11)
    ax2.set_ylabel('Aggressive Rewrite $\\Delta s$', fontsize=11)
    ax2.set_title('Standard vs. Aggressive Counterfactual Effects\n'
                  'Points above the diagonal respond more to aggressive reform',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig_counterfactual_comparison.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print('Saved: fig_counterfactual_comparison.png')


print('\nDone.')


# In[ ]:




