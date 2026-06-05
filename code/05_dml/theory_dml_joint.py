"""
Theory-Driven JOINT Vector DML — 21 features as vector treatment
4 specs: RF/GB × Full/NoTravel
10 seeds × 5 folds = 50 fits per spec. HC1 sandwich SEs + BH correction.
Output: theory_dml_joint_results.csv
"""

import os
import warnings
# WHY: Suppresses sklearn convergence warnings for SLURM logs
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

# ── CONFIGURATION ──────────────────────────────────────────────
TRIPS_PATH    = 'processed_data/tampatrips_1.csv'
ZON_CSV_PATH  = 'processed_data/tpazoning_clean.csv'
THEORY_PATH   = 'processed_data/zone_theory_expanded_raw.csv'
ACS_PATH      = 'processed_data/acs_blockgroup_tampa.csv'

N_FOLDS       = 5
# WHY 10 seeds: Averaging over 10 random fold partitions x 5 folds = 50 fits
# per spec reduces sensitivity to any single train/test split.
N_SEEDS       = 10
BASE_SEED     = 16
ALPHA         = 0.05

INDIVIDUAL_CONFOUNDERS = [
    'age', 'gender', 'education', 'income_detailed',
    'num_veh', 'hhsize', 'trip_distance', 'trip_duration',
    'd_purpose_category_imputed',
]

ACS_FEATURES = ['median_hh_income', 'pct_bachelors_plus',
                'pct_white_nonhisp', 'pct_black_nonhisp', 'pct_hispanic',
                'pct_poverty', 'pct_owner_occ']

TRAVEL_COLS = ['trip_distance', 'trip_duration']

# 21 theory features (mixed_use_perm excluded: constant=1)
ZONING_FEATURES = [
    'density_permissions', 'min_lot_area', 'max_height',
    'use_mixing', 'use_flexibility',
    'parking_intensity', 'reduced_parking',
    'ped_street_interface', 'front_setback', 'lot_width', 'frontage_std',
    'transparency_activation',
    'setback_building_placement', 'side_setback', 'rear_setback',
    'form_based_design', 'human_scale_design', 'ped_scale_lang',
    'transit_orientation', 'transit_orient',
    'open_space_green',
]

DISPLAY_INFO = {
    'density_permissions':        ('D1', 'density_permissions (LLM)'),
    'min_lot_area':               ('D1', 'min_lot_area (scraped, inv.)'),
    'max_height':                 ('D1', 'max_height (scraped)'),
    'use_mixing':                 ('D2', 'use_mixing (LLM)'),
    'use_flexibility':            ('D2', 'use_flexibility (LLM)'),
    'parking_intensity':          ('D4', 'parking_intensity (LLM, inv.)'),
    'reduced_parking':            ('D4', 'reduced_parking (binary)'),
    'ped_street_interface':       ('D3', 'ped_street_interface (LLM)'),
    'front_setback':              ('D3', 'front_setback (scraped, inv.)'),
    'lot_width':                  ('D3', 'lot_width (scraped, inv.)'),
    'frontage_std':               ('D3', 'frontage_std (binary)'),
    'transparency_activation':    ('D3', 'transparency_activation (LLM)'),
    'setback_building_placement': ('D5', 'setback_building_placement (LLM)'),
    'side_setback':               ('D5', 'side_setback (scraped, inv.)'),
    'rear_setback':               ('D5', 'rear_setback (scraped, inv.)'),
    'form_based_design':          ('D6', 'form_based_design (LLM)'),
    'human_scale_design':         ('D6', 'human_scale_design (binary)'),
    'ped_scale_lang':             ('D6', 'ped_scale_lang (binary)'),
    'transit_orientation':        ('D7', 'transit_orientation (LLM)'),
    'transit_orient':             ('D7', 'transit_orient (binary)'),
    'open_space_green':           ('D8', 'open_space_green (LLM)'),
}


# ── DML CORE FUNCTIONS ─────────────────────────────────────────

def impute_scale(X_tr, X_te):
    """Per-fold standardization to avoid information leakage across folds."""
    imp = SimpleImputer(strategy='median')
    sc  = StandardScaler()
    X_tr_out = sc.fit_transform(imp.fit_transform(X_tr))
    X_te_out = sc.transform(imp.transform(X_te))
    return X_tr_out, X_te_out


def cross_fit_vector_residuals(Y, T_matrix, W, clf_Y, reg_T_proto,
                                n_folds=N_FOLDS, seed=BASE_SEED,
                                stratify_on=None):
    """
    Cross-fitted residuals for VECTOR treatment (n × d).
    Each dimension of T is partialled out independently within each fold.
    """
    n, d  = T_matrix.shape
    Y_res = np.zeros(n)
    T_res = np.zeros((n, d))

    # WHY StratifiedKFold: Stratify on AT outcome (~10% prevalence) to
    # ensure each fold has balanced class representation.
    if stratify_on is not None:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(W, stratify_on)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(W)

    for fold, (tr_idx, te_idx) in enumerate(split_iter):
        # WHY per-fold: Per-fold standardization to avoid information leakage
        W_tr, W_te = impute_scale(W[tr_idx], W[te_idx])

        # WHY per-fold for T: Same leakage prevention for treatment vector
        imp_T = SimpleImputer(strategy='median')
        sc_T  = StandardScaler()
        T_tr  = sc_T.fit_transform(imp_T.fit_transform(T_matrix[tr_idx]))
        T_te  = sc_T.transform(imp_T.transform(T_matrix[te_idx]))

        # Outcome nuisance
        clf_f = clone(clf_Y)
        clf_f.fit(W_tr, Y[tr_idx])
        Y_hat = clf_f.predict_proba(W_te)[:, 1]
        Y_res[te_idx] = Y[te_idx] - Y_hat

        # Treatment nuisance: one regressor per dimension
        for dim in range(d):
            reg_f = clone(reg_T_proto)
            reg_f.fit(W_tr, T_tr[:, dim])
            T_hat_dim = reg_f.predict(W_te)
            T_res[te_idx, dim] = T_te[:, dim] - T_hat_dim

        print(f'    Fold {fold+1}/{n_folds} complete')

    return Y_res, T_res


def dml_vector_final_stage(Y_res, T_res, feature_names):
    """
    Final stage: OLS of Y_res on T_res (no intercept).
    WHY OLS second stage: Standard DML final stage regresses residualized
    outcome on residualized treatment. No intercept because both residuals
    are mean-zero by construction.
    HC1 sandwich SEs per dimension.
    Returns DataFrame with theta, SE, t, p per feature.
    """
    n, d = T_res.shape
    reg = LinearRegression(fit_intercept=False)
    reg.fit(T_res, Y_res)
    thetas = reg.coef_

    eps = Y_res - T_res @ thetas
    z_crit = stats.norm.ppf(1 - ALPHA / 2)

    rows = []
    for j in range(d):
        t_j  = T_res[:, j]
        T2_j = (t_j ** 2).sum()
        se_j = np.sqrt(n / (n - 1) * np.sum((t_j * eps) ** 2) / T2_j ** 2)
        t_stat = thetas[j] / se_j if se_j > 1e-12 else np.nan
        p_val  = 2 * stats.norm.sf(np.abs(t_stat)) if np.isfinite(t_stat) else np.nan
        rows.append({
            'feature':  feature_names[j],
            'theta':    thetas[j],
            'se_hc1':   se_j,
            't_stat':   t_stat,
            'p_raw':    p_val,
            'ci_lo':    thetas[j] - z_crit * se_j,
            'ci_hi':    thetas[j] + z_crit * se_j,
        })

    return pd.DataFrame(rows)


def aggregate_seeds(all_seed_results, feature_names):
    """
    Aggregate across S seeds using median theta and combined SE.
    WHY combined SE formula: se_within = median(per-seed HC1 SEs) captures
    fold sampling error; se_across = std(thetas across seeds) captures
    partition-assignment variance. Combined = sqrt(within^2 + across^2).
    Same aggregation as the embedding pipeline.
    """
    # Stack: rows = seeds, cols = features
    theta_matrix = np.array([[r['theta'] for r in seed_df.to_dict('records')]
                              for seed_df in all_seed_results])  # (S, d)
    se_matrix    = np.array([[r['se_hc1'] for r in seed_df.to_dict('records')]
                              for seed_df in all_seed_results])  # (S, d)

    d = len(feature_names)
    rows = []
    z_crit = stats.norm.ppf(1 - ALPHA / 2)

    for j in range(d):
        thetas_j = theta_matrix[:, j]
        ses_j    = se_matrix[:, j]

        theta_med = np.median(thetas_j)
        se_within = np.median(ses_j)
        se_across = np.std(thetas_j, ddof=1)
        se_combined = np.sqrt(se_within**2 + se_across**2)

        t_stat = theta_med / se_combined if se_combined > 1e-12 else np.nan
        p_val  = 2 * stats.norm.sf(np.abs(t_stat)) if np.isfinite(t_stat) else np.nan

        rows.append({
            'feature':      feature_names[j],
            'theta':        theta_med,
            'se_within':    se_within,
            'se_across':    se_across,
            'se_combined':  se_combined,
            't_stat':       t_stat,
            'p_raw':        p_val,
            'ci_lo':        theta_med - z_crit * se_combined,
            'ci_hi':        theta_med + z_crit * se_combined,
        })

    return pd.DataFrame(rows)


def bh_correction(pvals):
    """WHY BH: Controls false discovery rate within each specification,
    not across specs, to avoid over-correction from non-independent tests."""
    m = len(pvals)
    pvals = np.array(pvals)
    sorted_idx = np.argsort(pvals)
    p_adj = np.zeros(m)
    for rank, idx in enumerate(sorted_idx):
        p_adj[idx] = pvals[idx] * m / (rank + 1)
    for i in range(m - 2, -1, -1):
        idx = sorted_idx[i]
        idx_next = sorted_idx[i + 1]
        p_adj[idx] = min(p_adj[idx], p_adj[idx_next])
    return np.minimum(p_adj, 1.0)


def stars(p):
    if pd.isna(p) or not np.isfinite(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''


# ── 1. LOAD AND BUILD BG FEATURES ─────────────────────────────
print('=' * 65)
print('  Theory-Driven JOINT Vector DML')
print(f'  {len(ZONING_FEATURES)} features × 4 specs × {N_SEEDS} seeds × {N_FOLDS} folds')
print('=' * 65)

trips    = pd.read_csv(TRIPS_PATH)
zon_csv  = pd.read_csv(ZON_CSV_PATH).dropna(subset=['ZONECLASS'])
theory   = pd.read_csv(THEORY_PATH)
acs      = pd.read_csv(ACS_PATH)

trips['active_travel'] = trips['mode_type'].isin([1, 2]).astype(int)
area_col = ('ShapeSTAre_corrected' if 'ShapeSTAre_corrected' in zon_csv.columns
            else 'ShapeSTAre')

print(f'\nBuilding area-weighted block-group features...')

def weighted_mean(group, val_col):
    """WHY area-weighted: BG embeddings are area-weighted averages across zone
    classes. Area-weighted is least lossy vs dominant-zone-only or
    equal-weight alternatives."""
    mask = group[val_col].notna()
    if mask.sum() == 0: return np.nan
    return np.average(group.loc[mask, val_col], weights=group.loc[mask, area_col])

zon_m = zon_csv.merge(theory[['zone_class'] + ZONING_FEATURES],
                      left_on='ZONECLASS', right_on='zone_class', how='left')

bg_rows = []
for geoid, grp in zon_m.groupby('GEOID'):
    row = {'GEOID': geoid}
    for feat in ZONING_FEATURES:
        row[feat] = weighted_mean(grp, feat)
    bg_rows.append(row)
bg_theory = pd.DataFrame(bg_rows)

def entropy(group):
    total = group[area_col].sum()
    if total == 0: return 0.0
    p = group[area_col] / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

entropy_df = (zon_csv.groupby('GEOID')
              .apply(entropy, include_groups=False)
              .reset_index(name='zoning_entropy'))

bg = (bg_theory
      .merge(entropy_df, on='GEOID', how='left')
      .merge(acs[['GEOID'] + ACS_FEATURES], on='GEOID', how='left'))

df = trips.merge(bg, left_on='o_bg', right_on='GEOID', how='left')
df = df.dropna(subset=['zoning_entropy']).reset_index(drop=True)
print(f'Sample: {len(df):,} trips  |  AT rate: {df["active_travel"].mean():.1%}')

Y = df['active_travel'].values.astype(float)

# Treatment matrix: n × 21
T_matrix = df[ZONING_FEATURES].values.astype(float)
print(f'Treatment matrix: {T_matrix.shape}')

# Check condition number of raw T
from numpy.linalg import cond
imp_check = SimpleImputer(strategy='median')
sc_check  = StandardScaler()
T_check   = sc_check.fit_transform(imp_check.fit_transform(T_matrix))
cond_num  = cond(T_check)
print(f'Condition number of standardized T matrix: {cond_num:.1f}')
if cond_num > 100:
    print('  ⚠ High condition number — multicollinearity may affect estimates.')
    print('  Proceeding anyway per Dr. Wang\'s instruction.')

# ── SPECS ──────────────────────────────────────────────────────
W_FULL_COLS     = INDIVIDUAL_CONFOUNDERS + ACS_FEATURES
W_NOTRAVEL_COLS = [c for c in W_FULL_COLS if c not in TRAVEL_COLS]
W_full     = df[[c for c in W_FULL_COLS if c in df.columns]].values.astype(float)
W_notravel = df[[c for c in W_NOTRAVEL_COLS if c in df.columns]].values.astype(float)
print(f'W_full:     {W_full.shape[1]} features')
print(f'W_notravel: {W_notravel.shape[1]} features')

# WHY RF over lasso for nuisance: Nonlinear confounding favors trees.
# GB included as robustness check with shallower trees (depth=3).
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                 random_state=BASE_SEED, n_jobs=4)
reg_rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                random_state=BASE_SEED, n_jobs=4)
clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                     random_state=BASE_SEED)
reg_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                    random_state=BASE_SEED)

SPECS = {
    'RF_Full':      (clf_rf, reg_rf, W_full),
    'RF_NoTravel':  (clf_rf, reg_rf, W_notravel),
    'GB_Full':      (clf_gb, reg_gb, W_full),
    'GB_NoTravel':  (clf_gb, reg_gb, W_notravel),
}


# ── 2. RUN 4 SPECS × 10 SEEDS (resume-safe) ───────────────────
CHECKPOINT_PATH = 'theory_dml_joint_partial.csv'
all_results = []
completed_specs = set()

if os.path.exists(CHECKPOINT_PATH):
    prev = pd.read_csv(CHECKPOINT_PATH)
    completed_specs = set(prev['spec'].unique())
    for spec_name in completed_specs:
        all_results.append(prev[prev['spec'] == spec_name].copy())
    print(f'\n  *** RESUMING: found {len(completed_specs)} completed specs: {completed_specs} ***\n')
else:
    print(f'\n  No checkpoint — starting fresh.\n')

for spec_name, (clf_Y, reg_T, W_mat) in SPECS.items():
    if spec_name in completed_specs:
        print(f'\n  {spec_name} ✓ cached — skipping')
        continue

    print(f'\n{"="*65}')
    print(f'  SPEC: {spec_name}')
    print(f'{"="*65}')

    seed_dfs = []
    for s in range(N_SEEDS):
        seed = BASE_SEED + s
        print(f'\n  Seed {s+1}/{N_SEEDS} (seed={seed})')

        Y_res, T_res = cross_fit_vector_residuals(
            Y, T_matrix, W_mat, clf_Y, reg_T,
            n_folds=N_FOLDS, seed=seed,
            stratify_on=Y.astype(int))

        seed_result = dml_vector_final_stage(Y_res, T_res, ZONING_FEATURES)
        seed_dfs.append(seed_result)

    # Aggregate across seeds
    agg = aggregate_seeds(seed_dfs, ZONING_FEATURES)

    # BH correction
    p_adj = bh_correction(agg['p_raw'].values)
    agg['p_bh'] = p_adj
    agg['sig_bh'] = agg['p_bh'].apply(stars)
    agg['sig_raw'] = agg['p_raw'].apply(stars)
    agg['spec'] = spec_name

    # Add dimension labels
    agg['dimension'] = agg['feature'].map(lambda f: DISPLAY_INFO[f][0])
    agg['feature_label'] = agg['feature'].map(lambda f: DISPLAY_INFO[f][1])

    all_results.append(agg)

    # Save checkpoint after each spec
    checkpoint = pd.concat(all_results, ignore_index=True)
    checkpoint.to_csv(CHECKPOINT_PATH, index=False)
    print(f'  ✓ Checkpoint saved ({len(all_results)}/4 specs complete)')

    # Print summary
    n_sig = (agg['p_bh'] < 0.05).sum()
    n_marg = ((agg['p_bh'] >= 0.05) & (agg['p_bh'] < 0.10)).sum()
    print(f'\n  {spec_name}: {n_sig} BH-sig (p<0.05), {n_marg} marginal (p<0.10)')
    for _, r in agg[agg['p_bh'] < 0.10].sort_values('p_bh').iterrows():
        print(f'    {r["feature"]:<30} θ={r["theta"]:+.4f} SE={r["se_combined"]:.4f} '
              f'p_BH={r["p_bh"]:.4f} {r["sig_bh"]}')


# ── 3. SAVE ────────────────────────────────────────────────────
out = pd.concat(all_results, ignore_index=True)
out.to_csv('theory_dml_joint_results.csv', index=False)
print(f'\nSaved: theory_dml_joint_results.csv')


# ── 4. PIVOT TABLE ─────────────────────────────────────────────
def cell_str(row):
    if pd.isna(row['theta']): return 'NA'
    sig = row['sig_bh']
    return f'{row["theta"]:+.3f} ({row["se_combined"]:.3f}){sig}'

pivot_rows = []
for feat in ZONING_FEATURES:
    dim_label, feat_label = DISPLAY_INFO[feat]
    prow = {'dimension': dim_label, 'feature': feat_label}
    for spec_name in ['RF_Full', 'RF_NoTravel', 'GB_Full', 'GB_NoTravel']:
        match = out[(out['feature'] == feat) & (out['spec'] == spec_name)]
        if len(match) == 1:
            prow[spec_name] = cell_str(match.iloc[0])
        else:
            prow[spec_name] = 'NA'
    pivot_rows.append(prow)

pivot = pd.DataFrame(pivot_rows)
pivot.to_csv('theory_dml_joint_pivot.csv', index=False)
print(f'Saved: theory_dml_joint_pivot.csv')
print('\n✓ Complete.')
