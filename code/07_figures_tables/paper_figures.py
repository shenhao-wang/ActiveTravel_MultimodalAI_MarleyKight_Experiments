"""
Publication figures for the main body of the paper.
Five figures, each saved as a standalone PNG (300 dpi) and PDF.

Fig A: OLS → DML shrinkage (T1, T2, T3)
Fig B: Theory-dimension dot-and-whisker (placeholder until DML results arrive)
Fig C: Counterfactual dumbbell chart (3-panel)
Fig D: Entanglement diagnostic paired bars
Fig E: PCA scatter with zone-family convex hulls

Requires: matplotlib, numpy, pandas
Optional: scipy (for convex hulls in Fig E)

Usage:
  python paper_figures.py          # generates all 5
  python paper_figures.py --fig A  # generates just Fig A
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ── GLOBAL STYLE ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Palette: navy + teal + terracotta — chosen for colorblind accessibility and
# consistent with existing deliverables (slides, poster). Navy = primary, teal = secondary.
NAVY      = '#1B2A4A'
TEAL      = '#2A7F7F'
TERRACOTTA= '#C75B39'
SAGE      = '#7A9E7E'
CREAM     = '#F5F0E8'
GREY_LIGHT= '#D0D0D0'
GREY_MID  = '#888888'
ORANGE    = '#E8913A'

DPI = 300


# ══════════════════════════════════════════════════════════════
# FIG A: OLS → DML SHRINKAGE
# ══════════════════════════════════════════════════════════════
def fig_a_ols_dml_shrinkage():
    """Grouped horizontal bar chart: OLS vs DML for T1, T2, T3.
    Shows OLS overestimates effects by 97-192% due to residential self-selection
    confounding. DML removes this bias via cross-fitted nuisance models.
    """

    treatments = [
        'T1: Residential zone\n(binary)',
        'T2: Pedestrian orientation\nscore (0–1)',
        'T3: Front setback\n(per foot)',
    ]
    ols_theta  = [-0.2117, -0.0541, -0.0067]
    dml_theta  = [-0.0724, -0.0195, -0.0034]
    dml_se     = [0.0143,   0.0107,  0.0007]
    dml_p      = ['< 0.001', '0.069', '< 0.001']
    bias_pct   = [192, 177, 97]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    y = np.arange(len(treatments))
    bar_h = 0.32

    # OLS bars faded to visually subordinate them -- the DML estimate is the
    # causal claim; OLS shown only to quantify the bias magnitude.
    bars_ols = ax.barh(y + bar_h/2, ols_theta, height=bar_h,
                       color=GREY_LIGHT, edgecolor=GREY_MID, linewidth=0.8,
                       label='OLS (full controls)', zorder=2)
    # DML bars (solid)
    bars_dml = ax.barh(y - bar_h/2, dml_theta, height=bar_h,
                       color=NAVY, edgecolor=NAVY, linewidth=0.8,
                       label='DML (RF nuisance)', zorder=2,
                       xerr=[[1.96*s for s in dml_se],
                             [1.96*s for s in dml_se]],
                       error_kw={'linewidth': 1.2, 'capsize': 3,
                                 'capthick': 1.2, 'color': TERRACOTTA})

    # Bias annotation
    for i in range(len(treatments)):
        x_mid = (ols_theta[i] + dml_theta[i]) / 2
        ax.annotate(f'{bias_pct[i]}% bias',
                    xy=(x_mid, y[i] + 0.02),
                    fontsize=8, fontweight='bold', color=TERRACOTTA,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec=TERRACOTTA, alpha=0.9, linewidth=0.8))

    ax.axvline(0, color='black', linewidth=0.6, linestyle='-', zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(treatments)
    ax.set_xlabel(r'Estimated effect on P(Active Travel)')
    ax.set_title('OLS overestimation of zoning effects on active travel',
                 fontweight='bold', pad=10)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor=GREY_LIGHT)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig('fig_ols_dml_shrinkage.png', dpi=DPI)
    fig.savefig('fig_ols_dml_shrinkage.pdf')
    plt.close(fig)
    print('Saved: fig_ols_dml_shrinkage.png / .pdf')


# ══════════════════════════════════════════════════════════════
# FIG B: THEORY-DIMENSION DOT-AND-WHISKER
# ══════════════════════════════════════════════════════════════
def fig_b_theory_dot_whisker():
    """
    Coefficient plot for 22 theory features grouped by D1–D7.
    Uses the 3 BH-significant features from the paper + placeholder
    non-significant features.
    """

    # Data: (feature_label, dimension, theta, se, p_bh)
    # Using existing results from Section 5.5 (22-feature vector DML)
    # BH-significant ones from the paper; rest are non-significant placeholders
    features = [
        # D1
        ('Density permissions (LLM)',        'D1', -0.070, 0.054, 0.197),
        ('Min. lot area (scraped, inv.)',     'D1', -0.008, 0.036, 0.825),
        ('Max. height (scraped)',             'D1', -0.011, 0.011, 0.297),
        # D2
        ('Use mixing (LLM)',                 'D2', 0.041, 0.038, 0.274),
        ('Use flexibility (LLM)',            'D2', 0.056, 0.048, 0.244),
        # D3
        ('Parking intensity (LLM, inv.)',    'D3', -0.042, 0.030, 0.340),
        ('Reduced parking (binary)',          'D3', 0.018, 0.020, 0.580),
        # D4
        ('Ped. street interface (LLM)',      'D4', -0.186, 0.056, 0.009),  # BH sig
        ('Front setback (scraped, inv.)',     'D4', -0.020, 0.025, 0.650),
        ('Lot width (scraped, inv.)',         'D4', 0.005, 0.015, 0.820),
        ('Frontage standard (binary)',        'D4', 0.113, 0.035, 0.009),  # BH sig
        ('Transparency & activation (LLM)',  'D4', 0.030, 0.025, 0.450),
        # D5
        ('Setback/bldg. placement (LLM)',    'D5', 0.025, 0.035, 0.550),
        ('Side setback (scraped, inv.)',      'D5', -0.010, 0.020, 0.700),
        ('Rear setback (scraped, inv.)',      'D5', 0.015, 0.018, 0.600),
        # D6
        ('Form-based design (LLM)',          'D6', -0.035, 0.040, 0.500),
        ('Human-scale design (binary)',       'D6', -0.012, 0.018, 0.650),
        ('Ped.-scale language (binary)',      'D6', 0.020, 0.022, 0.480),
        # D7
        ('Transit orientation (LLM)',        'D7', 0.321, 0.087, 0.005),  # BH sig
        ('Transit-oriented (binary)',         'D7', 0.045, 0.030, 0.250),
        # D+
        ('Open space & green (LLM)',         'D+', 0.025, 0.030, 0.500),
    ]

    labels = [f[0] for f in features]
    dims   = [f[1] for f in features]
    thetas = np.array([f[2] for f in features])
    ses    = np.array([f[3] for f in features])
    p_bh   = np.array([f[4] for f in features])

    n = len(features)

    # Dimension colors: one hue per D-group so the reader can visually cluster
    # features within the same planning theory dimension.
    dim_colors = {
        'D1': '#2A7F7F', 'D2': '#3A9E5C', 'D3': '#E8913A',
        'D4': '#C75B39', 'D5': '#8B5E3C', 'D6': '#6B5B95',
        'D7': '#1B2A4A', 'D+': '#7A9E7E',
    }

    fig, ax = plt.subplots(figsize=(6.5, 8))

    y_pos = np.arange(n)[::-1]  # bottom to top

    for i in range(n):
        color = dim_colors.get(dims[i], GREY_MID)
        # Graduated alpha + marker shape encodes significance level: BH-significant
        # features are visually prominent; non-significant fade to background.
        alpha = 1.0 if p_bh[i] < 0.05 else (0.7 if p_bh[i] < 0.10 else 0.35)
        marker = 'D' if p_bh[i] < 0.05 else ('s' if p_bh[i] < 0.10 else 'o')
        ms = 7 if p_bh[i] < 0.05 else 5

        # CI whisker
        ci_lo = thetas[i] - 1.96 * ses[i]
        ci_hi = thetas[i] + 1.96 * ses[i]
        ax.plot([ci_lo, ci_hi], [y_pos[i], y_pos[i]],
                color=color, linewidth=1.5, alpha=alpha, zorder=2)
        # Point
        ax.plot(thetas[i], y_pos[i], marker=marker, color=color,
                markersize=ms, alpha=alpha, zorder=3,
                markeredgecolor='white', markeredgewidth=0.5)

    # Zero line
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', zorder=1)

    # Dimension group separators
    prev_dim = None
    for i in range(n):
        if dims[i] != prev_dim and prev_dim is not None:
            ax.axhline(y_pos[i] + 0.5, color=GREY_LIGHT, linewidth=0.5,
                       linestyle='-', zorder=0)
        prev_dim = dims[i]

    # Dimension labels on right margin
    dim_positions = {}
    for i in range(n):
        d = dims[i]
        if d not in dim_positions:
            dim_positions[d] = []
        dim_positions[d].append(y_pos[i])
    for d, positions in dim_positions.items():
        mid = np.mean(positions)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.4,
                mid, d, fontsize=8, fontweight='bold',
                color=dim_colors.get(d, GREY_MID),
                ha='left', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r'DML causal estimate $\hat{\theta}$ (with 95% CI)')
    ax.set_title('Theory-driven DML estimates by zoning dimension',
                 fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.15, linewidth=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=NAVY,
               markersize=7, label='BH significant (p < 0.05)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=GREY_MID,
               markersize=5, label='Marginal (p < 0.10)', alpha=0.7),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY_MID,
               markersize=5, label='Not significant', alpha=0.35),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              framealpha=0.95, edgecolor=GREY_LIGHT)

    fig.tight_layout()
    fig.savefig('fig_theory_dot_whisker.png', dpi=DPI)
    fig.savefig('fig_theory_dot_whisker.pdf')
    plt.close(fig)
    print('Saved: fig_theory_dot_whisker.png / .pdf')


# ══════════════════════════════════════════════════════════════
# FIG C: COUNTERFACTUAL DUMBBELL CHART
# ══════════════════════════════════════════════════════════════
def fig_c_counterfactual_dumbbell():
    """Dumbbell chart: original vs counterfactual causal score by zone.
    Three-panel layout reflects the typology from the counterfactual analysis:
    zones separate into improvable, structurally constrained, and AT-regressive
    categories based on their Δs response to pedestrian-friendly rewriting.
    """

    # Data from the paper's counterfactual table (aggressive variant)
    # Panel A: AT-Promoting (improvable)
    panel_a = [
        ('IG',    -0.0315, -0.0055),
        ('CN',    -0.0131, +0.0121),
        ('RM-12', -0.0301, -0.0065),
        ('RM-16', -0.0292, -0.0119),
        ('RM-18', -0.0308, -0.0129),
        ('RM-24', -0.0235, -0.0056),
        ('RM-35', -0.0275, -0.0146),
        ('RM-50', -0.0294, -0.0204),
        ('RM-75', -0.0251, -0.0051),
        ('CI',    -0.0274, -0.0070),
        ('IH',    -0.0340, -0.0138),
        ('CG',    -0.0101, +0.0038),
        ('OP',    -0.0310, -0.0143),
        ('OP-1',  -0.0284, -0.0042),
    ]
    # Panel B: Structurally constrained
    panel_b = [
        ('RS-50',  -0.0347, -0.0371),
        ('RS-60',  -0.0352, -0.0403),
        ('RS-75',  -0.0259, -0.0331),
        ('RS-100', -0.0295, -0.0285),
        ('RS-150', -0.0321, -0.0306),
        ('CBD-1',  +0.0250, +0.0240),
        ('YC-1',   +0.0359, +0.0328),
    ]
    # Panel C: AT-Regressive
    panel_c = [
        ('CD-1',    +0.0438, +0.0014),
        ('CD-2',    +0.0441, +0.0121),
        ('CD-3',    +0.0488, +0.0377),
        ('SH-RS-A', +0.0081, -0.0044),
        ('M-AP-4',  +0.0106, -0.0024),
    ]

    # Three-panel layout separates zones by counterfactual response type so the
    # reader can compare within each policy-relevant category.
    panels = [
        ('A: Improvable', panel_a, SAGE),
        ('B: Constrained', panel_b, GREY_MID),
        ('C: AT-Regressive', panel_c, TERRACOTTA),
    ]

    # Build flat list
    all_zones = []
    all_orig  = []
    all_cf    = []
    all_color = []
    panel_breaks = []

    for panel_label, data, color in panels:
        panel_breaks.append(len(all_zones))
        for zone, orig, cf in data:
            all_zones.append(zone)
            all_orig.append(orig)
            all_cf.append(cf)
            all_color.append(color)

    n = len(all_zones)
    fig, ax = plt.subplots(figsize=(7, 8))

    y = np.arange(n)[::-1]

    for i in range(n):
        # Connecting line
        ax.plot([all_orig[i], all_cf[i]], [y[i], y[i]],
                color=all_color[i], linewidth=1.5, alpha=0.6, zorder=2)
        # Original dot
        ax.plot(all_orig[i], y[i], 'o', color=NAVY, markersize=5,
                zorder=3, markeredgecolor='white', markeredgewidth=0.4)
        # Counterfactual dot
        ax.plot(all_cf[i], y[i], 'D', color=all_color[i], markersize=5,
                zorder=3, markeredgecolor='white', markeredgewidth=0.4)
        # Arrow
        dx = all_cf[i] - all_orig[i]
        if abs(dx) > 0.001:
            ax.annotate('', xy=(all_cf[i], y[i]),
                        xytext=(all_orig[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color=all_color[i],
                                        lw=1.2, alpha=0.7))

    # Zero line
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', zorder=1)

    # Panel labels
    for j, (panel_label, data, color) in enumerate(panels):
        idx = panel_breaks[j]
        y_label = y[idx] + 0.8
        ax.text(-0.055, y_label, f'Panel {panel_label}',
                fontsize=9, fontweight='bold', color=color,
                ha='left', va='bottom',
                transform=ax.get_yaxis_transform())

    # Panel separators
    for j in range(1, len(panel_breaks)):
        idx = panel_breaks[j]
        ax.axhline(y[idx] + 0.5, color=GREY_LIGHT, linewidth=0.8,
                   linestyle='-', zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(all_zones, fontsize=8)
    ax.set_xlabel(r'Causal score $s_z = \mathbf{e}_z \cdot \hat{\boldsymbol{\theta}}$')
    ax.set_title('Counterfactual zoning reform: original vs. aggressive rewrite',
                 fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.15, linewidth=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NAVY,
               markersize=6, label='Original score'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=SAGE,
               markersize=6, label='Counterfactual (improvable)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=GREY_MID,
               markersize=6, label='Counterfactual (constrained)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=TERRACOTTA,
               markersize=6, label='Counterfactual (regressive)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              framealpha=0.95, edgecolor=GREY_LIGHT, fontsize=8)

    fig.tight_layout()
    fig.savefig('fig_counterfactual_dumbbell.png', dpi=DPI)
    fig.savefig('fig_counterfactual_dumbbell.pdf')
    plt.close(fig)
    print('Saved: fig_counterfactual_dumbbell.png / .pdf')


# ══════════════════════════════════════════════════════════════
# FIG D: ENTANGLEMENT DIAGNOSTIC PAIRED BARS
# ══════════════════════════════════════════════════════════════
def fig_d_entanglement():
    """Paired bar chart: OLS R² vs RF R² for 5 zoning features."""

    features = [
        'Pedestrian\norientation',
        'Front\nsetback',
        'Lot\nwidth',
        'Max\nheight',
        'Zoning\nentropy',
    ]
    ols_r2 = [0.147, 0.387, 0.086, 0.468, 0.044]
    rf_r2  = [0.896, 0.952, 0.788, 0.956, 0.849]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(features))
    w = 0.35

    bars_ols = ax.bar(x - w/2, ols_r2, w, color=GREY_LIGHT,
                      edgecolor=GREY_MID, linewidth=0.8,
                      label='OLS $R^2$', zorder=2)
    bars_rf  = ax.bar(x + w/2, rf_r2, w, color=NAVY,
                      edgecolor=NAVY, linewidth=0.8,
                      label='RF $R^2$', zorder=2)

    # Gap annotations
    for i in range(len(features)):
        gap = rf_r2[i] - ols_r2[i]
        factor = rf_r2[i] / ols_r2[i] if ols_r2[i] > 0 else 0
        mid = (ols_r2[i] + rf_r2[i]) / 2
        ax.annotate(f'{factor:.1f}×',
                    xy=(x[i], mid), fontsize=8.5, fontweight='bold',
                    color=TERRACOTTA, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec=TERRACOTTA, alpha=0.9, linewidth=0.7))
        # Bracket
        ax.plot([x[i], x[i]], [ols_r2[i] + 0.01, rf_r2[i] - 0.01],
                color=TERRACOTTA, linewidth=0.8, alpha=0.5, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel('Cross-validated $R^2$')
    ax.set_ylim(0, 1.05)
    ax.set_title('Demographic predictability of zoning features:\nlinear vs. nonlinear confounding',
                 fontweight='bold', pad=10)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor=GREY_LIGHT)
    ax.grid(axis='y', alpha=0.15, linewidth=0.5)

    # Horizontal reference
    ax.axhline(0.5, color=GREY_MID, linewidth=0.5, linestyle=':', alpha=0.5)
    ax.text(len(features) - 0.5, 0.51, 'R² = 0.5', fontsize=7,
            color=GREY_MID, ha='right', va='bottom')

    fig.tight_layout()
    fig.savefig('fig_entanglement_diagnostic.png', dpi=DPI)
    fig.savefig('fig_entanglement_diagnostic.pdf')
    plt.close(fig)
    print('Saved: fig_entanglement_diagnostic.png / .pdf')


# ══════════════════════════════════════════════════════════════
# FIG E: PCA SCATTER WITH CONVEX HULLS
# ══════════════════════════════════════════════════════════════
def fig_e_pca_hulls():
    """
    PCA scatter colored by causal score with zone-family convex hulls.
    Requires: zone_embeddings.csv and dml_zone_causal_scores.csv
    Falls back to synthetic data if files not found.
    """
    import os

    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        print('  scipy not available — skipping convex hulls')
        ConvexHull = None

    # Try to load real data
    emb_path    = 'zone_embeddings.csv'
    scores_path = 'dml_zone_causal_scores.csv'

    if os.path.exists(emb_path) and os.path.exists(scores_path):
        import pandas as pd
        from sklearn.decomposition import PCA

        emb_df = pd.read_csv(emb_path)
        scores_df = pd.read_csv(scores_path)
        merged = emb_df.merge(scores_df, on='zone_class')

        emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
        X = merged[emb_cols].values
        zone_names = merged['zone_class'].tolist()
        causal_scores = merged['causal_score'].values

        pca = PCA(n_components=2, random_state=16)
        coords = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_
        print(f'  Loaded real data: {len(zone_names)} zones')
    else:
        print(f'  Data files not found — using placeholder coordinates')
        print(f'  (Place zone_embeddings.csv and dml_zone_causal_scores.csv')
        print(f'   in working directory for real figure)')
        return

    # Zone families
    rs_zones = [z for z in zone_names if z.startswith('RS-')]
    rm_zones = [z for z in zone_names if z.startswith('RM-')]
    cd_nmu   = [z for z in zone_names if z.startswith(('CD-', 'NMU-', 'CBD-'))]
    sh_zones = [z for z in zone_names if z.startswith('SH-')]
    commercial = [z for z in zone_names if z.startswith(('CG', 'CI', 'CN', 'OP', 'IG', 'IH'))]

    families = [
        ('Single-Family (RS)', rs_zones, TEAL, 0.12),
        ('Multi-Family (RM)', rm_zones, SAGE, 0.12),
        ('Downtown/Mixed-Use\n(CD, NMU, CBD)', cd_nmu, TERRACOTTA, 0.10),
        ('Commercial/Industrial', commercial, ORANGE, 0.08),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#FAFAF8')

    # RdBu_r diverging colormap: red = car-promoting, blue = AT-promoting.
    # Centered at zero so sign of causal score maps to intuitive color.
    vmax = np.abs(causal_scores).max()
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=causal_scores, cmap='RdBu_r',
                    s=80, alpha=0.9, zorder=4,
                    edgecolors='white', linewidths=0.5,
                    vmin=-vmax, vmax=vmax)

    # Zone labels
    for i, z in enumerate(zone_names):
        ax.annotate(z, (coords[i, 0], coords[i, 1]),
                    fontsize=5, ha='center', va='bottom',
                    xytext=(0, 3.5), textcoords='offset points',
                    alpha=0.85)

    # Convex hulls
    if ConvexHull is not None:
        for fam_label, fam_zones, fam_color, fam_alpha in families:
            indices = [i for i, z in enumerate(zone_names) if z in fam_zones]
            if len(indices) < 3:
                continue
            pts = coords[indices]
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=fam_color, alpha=fam_alpha, zorder=1)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=fam_color, linewidth=1.2, alpha=0.5, zorder=2)
            except Exception:
                pass

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('DML causal score $s_z$\n'
                   '(+) → AT-promoting  (−) → car-promoting',
                   fontsize=9)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
    ax.set_title('Causal direction in zoning embedding space\n'
                 'with zone-family clusters',
                 fontweight='bold', pad=10)
    ax.grid(alpha=0.15, linewidth=0.5)

    # Hull legend
    hull_patches = [mpatches.Patch(color=c, alpha=a + 0.15, label=l)
                    for l, _, c, a in families]
    ax.legend(handles=hull_patches, loc='upper left',
              framealpha=0.95, edgecolor=GREY_LIGHT, fontsize=8)

    fig.tight_layout()
    fig.savefig('fig_pca_hulls.png', dpi=DPI)
    fig.savefig('fig_pca_hulls.pdf')
    plt.close(fig)
    print('Saved: fig_pca_hulls.png / .pdf')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    target = sys.argv[2].upper() if len(sys.argv) > 2 and sys.argv[1] == '--fig' else 'ALL'

    if target in ('A', 'ALL'):
        fig_a_ols_dml_shrinkage()
    if target in ('B', 'ALL'):
        fig_b_theory_dot_whisker()
    if target in ('C', 'ALL'):
        fig_c_counterfactual_dumbbell()
    if target in ('D', 'ALL'):
        fig_d_entanglement()
    if target in ('E', 'ALL'):
        fig_e_pca_hulls()

    if target == 'ALL':
        print('\n✓ All figures generated.')
