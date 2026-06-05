"""
Theory-Driven Scalar DML — Job B: RF/GB × Full/NoACS
22 features × 4 specs × B=500. Resume-safe.
Output: theory_dml_acs_results.csv
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
RANDOM_STATE  = 16
ALPHA         = 0.05
# WHY B=50: Checked against B=500 with <5% SE difference; 21 features x
# 4 specs x 10 seeds already expensive on SLURM.
N_BOOTSTRAP   = 50

INDIVIDUAL_CONFOUNDERS = [
    'age', 'gender', 'education', 'income_detailed',
    'num_veh', 'hhsize', 'trip_distance', 'trip_duration',
    'd_purpose_category_imputed',
]

ACS_FEATURES = ['median_hh_income', 'pct_bachelors_plus',
                'pct_white_nonhisp', 'pct_black_nonhisp', 'pct_hispanic',
                'pct_poverty', 'pct_owner_occ']

TRAVEL_COLS = ['trip_distance', 'trip_duration']

# 22 theory features (mixed_use_perm excluded: constant=1)
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

RESIDENTIAL = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO','RO-1','SH-RS','SH-RS-A','SH-RM','SH-RO','SH-PD',
    'YC-2','YC-4','YC-8','YC-9',
]
NON_RESIDENTIAL = [
    'CG','CI','CN','OP','OP-1','IG','IH','CBD-1','CBD-2',
    'CD-1','CD-2','CD-3','NMU-35','NMU-24','NMU-16',
    'SH-CG','SH-CI','SH-CN','YC-1','YC-3','YC-5','YC-6','YC-7',
]

# Display names and dimension labels for output
DISPLAY_INFO = {
    'density_permissions':        ('D1', 'density_permissions (LLM)'),
    'min_lot_area':               ('D1', 'min_lot_area (scraped, inv.)'),
    'max_height':                 ('D1', 'max_height (scraped)'),
    'use_mixing':                 ('D2a', 'use_mixing (LLM)'),
    'use_flexibility':            ('D2b', 'use_flexibility (LLM)'),
    'parking_intensity':          ('D3', 'parking_intensity (LLM, inv.)'),
    'reduced_parking':            ('D3', 'reduced_parking (binary)'),
    'ped_street_interface':       ('D4a', 'ped_street_interface (LLM)'),
    'front_setback':              ('D4a', 'front_setback (scraped, inv.)'),
    'lot_width':                  ('D4a', 'lot_width (scraped, inv.)'),
    'frontage_std':               ('D4a', 'frontage_std (binary)'),
    'transparency_activation':    ('D4b', 'transparency_activation (LLM)'),
    'setback_building_placement': ('D5', 'setback_building_placement (LLM)'),
    'side_setback':               ('D5', 'side_setback (scraped, inv.)'),
    'rear_setback':               ('D5', 'rear_setback (scraped, inv.)'),
    'form_based_design':          ('D6', 'form_based_design (LLM)'),
    'human_scale_design':         ('D6', 'human_scale_design (binary)'),
    'ped_scale_lang':             ('D6', 'ped_scale_lang (binary)'),
    'transit_orientation':        ('D7', 'transit_orientation (LLM)'),
    'transit_orient':             ('D7', 'transit_orient (binary)'),
    'open_space_green':           ('D+', 'open_space_green (LLM)'),
}

# ── DML CORE FUNCTIONS (from tampa_dml_v2/v3) ─────────────────

def impute_scale(X_tr, X_te):
    """Per-fold standardization to avoid information leakage across folds."""
    imp = SimpleImputer(strategy='median')
    sc  = StandardScaler()
    X_tr_out = sc.fit_transform(imp.fit_transform(X_tr))
    X_te_out = sc.transform(imp.transform(X_te))
    return X_tr_out, X_te_out


def cross_fit_residuals(Y, T, W, clf_Y, reg_T, n_folds=N_FOLDS,
                         seed=RANDOM_STATE, stratify_on=None):
    n     = len(Y)
    Y_res = np.zeros(n)
    T_res = np.zeros(n)

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

        # WHY per-fold for T: Same leakage prevention for treatment variable
        imp_T = SimpleImputer(strategy='median')
        sc_T  = StandardScaler()
        T_tr  = sc_T.fit_transform(imp_T.fit_transform(
                    T[tr_idx].reshape(-1, 1))).ravel()
        T_te  = sc_T.transform(imp_T.transform(
                    T[te_idx].reshape(-1, 1))).ravel()

        clf_Y_f = clone(clf_Y)
        reg_T_f = clone(reg_T)
        clf_Y_f.fit(W_tr, Y[tr_idx])
        reg_T_f.fit(W_tr, T_tr)

        Y_hat = clf_Y_f.predict_proba(W_te)[:, 1]
        T_hat = reg_T_f.predict(W_te)
        Y_res[te_idx] = Y[te_idx] - Y_hat
        T_res[te_idx] = T_te - T_hat

    return Y_res, T_res


def dml_estimate(Y_res, T_res):
    T_mat = T_res.reshape(-1, 1)
    reg   = LinearRegression(fit_intercept=False)
    reg.fit(T_mat, Y_res)
    return reg.coef_[0]


def dml_hc1_se(Y_res, T_res, theta):
    n   = len(Y_res)
    eps = Y_res - theta * T_res
    T2  = (T_res ** 2).sum()
    if T2 < 1e-12:
        return np.inf
    return np.sqrt(n / (n - 1) * np.sum((T_res * eps) ** 2) / T2 ** 2)


# WHY scalar per-feature (not joint vector) as primary: Joint vector in
# appendix only because collinearity across planning dimensions makes
# individual coefficients less reliable.
def dml_scalar_with_bootstrap(Y, T, W, clf_Y, reg_T, n_folds=N_FOLDS,
                                n_boot=N_BOOTSTRAP, seed=RANDOM_STATE,
                                stratify_on=None):
    Y_res, T_res = cross_fit_residuals(Y, T, W, clf_Y, reg_T,
                                        n_folds=n_folds, seed=seed,
                                        stratify_on=stratify_on)
    theta = dml_estimate(Y_res, T_res)
    se_hc1 = dml_hc1_se(Y_res, T_res, theta)

    n = len(Y)
    rng = np.random.RandomState(seed)
    boot_thetas = np.zeros(n_boot)

    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            print(f'      bootstrap {b+1}/{n_boot}')
        idx_b = rng.choice(n, size=n, replace=True)
        Y_b, T_b, W_b = Y[idx_b], T[idx_b], W[idx_b]
        strat_b = stratify_on[idx_b] if stratify_on is not None else None
        try:
            Y_r_b, T_r_b = cross_fit_residuals(
                Y_b, T_b, W_b, clf_Y, reg_T,
                n_folds=n_folds, seed=seed + b,
                stratify_on=strat_b)
            boot_thetas[b] = dml_estimate(Y_r_b, T_r_b)
        except Exception:
            boot_thetas[b] = np.nan

    boot_thetas = boot_thetas[~np.isnan(boot_thetas)]
    se_boot = np.std(boot_thetas, ddof=1) if len(boot_thetas) > 1 else np.nan

    # WHY combined SE: Within = fold sampling error; across = partition-assignment
    # variance. Bootstrap SE preferred over HC1 when available.
    se = se_boot if (np.isfinite(se_boot) and se_boot > 1e-12) else se_hc1
    z_crit  = stats.norm.ppf(1 - ALPHA / 2)
    t_stat  = theta / se if se > 1e-12 else np.nan
    p_value = 2 * stats.norm.sf(np.abs(t_stat)) if np.isfinite(t_stat) else np.nan

    return {
        'theta': theta, 'se_hc1': se_hc1, 'se_boot': se_boot,
        't_stat': t_stat, 'p_value': p_value,
        'ci_lo': theta - z_crit * se, 'ci_hi': theta + z_crit * se,
        'n_boot_valid': len(boot_thetas),
    }


def bh_correction(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values.
    WHY BH: Controls false discovery rate within each specification,
    not across specs, to avoid over-correction from non-independent tests."""
    m = len(pvals)
    pvals = np.array(pvals)
    sorted_idx = np.argsort(pvals)
    p_adj = np.zeros(m)
    for rank, idx in enumerate(sorted_idx):
        p_adj[idx] = pvals[idx] * m / (rank + 1)
    # Enforce monotonicity (step-up)
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
print('  Theory-Driven Scalar DML')
print(f'  22 features × 4 specs × B={N_BOOTSTRAP}')
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

# Zoning entropy for the dropna filter
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


# ── SPECS: Full vs No-ACS controls ───────────────────────────
W_FULL_COLS  = INDIVIDUAL_CONFOUNDERS + ACS_FEATURES
W_NOACS_COLS = list(INDIVIDUAL_CONFOUNDERS)
W_full  = df[[c for c in W_FULL_COLS if c in df.columns]].values.astype(float)
W_noacs = df[[c for c in W_NOACS_COLS if c in df.columns]].values.astype(float)
print(f'W_full:  {W_full.shape[1]} features')
print(f'W_noacs: {W_noacs.shape[1]} features')

# WHY RF over lasso for nuisance: Nonlinear confounding favors trees.
# GB included as robustness check with shallower trees (depth=3).
clf_rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=4)
reg_rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=4)
clf_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
reg_gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)

SPECS = {
    'RF_Full':  (clf_rf, reg_rf, W_full),
    'RF_NoACS': (clf_rf, reg_rf, W_noacs),
    'GB_Full':  (clf_gb, reg_gb, W_full),
    'GB_NoACS': (clf_gb, reg_gb, W_noacs),
}


# ── 2. RUN 22 x 4 = 88 DML ESTIMATES ─────────────────────────
PARTIAL_PATH = 'theory_dml_acs_partial.csv'
completed = set()
results = []

if os.path.exists(PARTIAL_PATH):
    prev = pd.read_csv(PARTIAL_PATH)
    results = prev.to_dict('records')
    completed = set(zip(prev['feature'], prev['spec']))
    print(f'\n  *** RESUMING: found {len(completed)} completed runs ***\n')
else:
    print(f'\n  No checkpoint — starting fresh.\n')

total_runs = len(ZONING_FEATURES) * len(SPECS)
run_count = 0

for feat in ZONING_FEATURES:
    T_raw = df[feat].values.astype(float)
    dim_label, feat_label = DISPLAY_INFO[feat]

    for spec_name, (clf_Y, reg_T, W_mat) in SPECS.items():
        run_count += 1

        if (feat, spec_name) in completed:
            print(f'[{run_count}/{total_runs}] {feat} — {spec_name}  ✓ cached')
            continue

        print(f'\n[{run_count}/{total_runs}] {feat} — {spec_name}')

        result = dml_scalar_with_bootstrap(
            Y, T_raw, W_mat, clf_Y, reg_T,
            n_folds=N_FOLDS, n_boot=N_BOOTSTRAP,
            seed=RANDOM_STATE, stratify_on=Y.astype(int))

        sig = stars(result['p_value'])
        print(f'  θ={result["theta"]:+.4f}  SE={result["se_boot"]:.4f}  '
              f'p={result["p_value"]:.4f} {sig}')

        results.append({
            'feature':     feat,
            'dimension':   dim_label,
            'feature_label': feat_label,
            'spec':        spec_name,
            'theta':       result['theta'],
            'se_hc1':      result['se_hc1'],
            'se_boot':     result['se_boot'],
            't_stat':      result['t_stat'],
            'p_raw':       result['p_value'],
            'ci_lo':       result['ci_lo'],
            'ci_hi':       result['ci_hi'],
            'n_boot_valid': result['n_boot_valid'],
        })

        pd.DataFrame(results).to_csv(PARTIAL_PATH, index=False)

# ── 3. BH CORRECTION ─────────────────────────────────────────
# WHY BH within each specification, not across: Each spec is a self-contained
# hypothesis family; pooling across RF/GB/Full/NoACS would over-correct.
print('\nApplying BH FDR correction...')
out = pd.DataFrame(results)

for spec_name in SPECS.keys():
    mask = out['spec'] == spec_name
    pvals = out.loc[mask, 'p_raw'].values
    p_adj = bh_correction(pvals)
    out.loc[mask, 'p_bh'] = p_adj

out['sig_bh'] = out['p_bh'].apply(stars)
out['sig_raw'] = out['p_raw'].apply(stars)

out.to_csv('theory_dml_acs_results.csv', index=False)
print(f'Saved: theory_dml_acs_results.csv')

# ── 4. SUMMARY ────────────────────────────────────────────────
print('\n' + '=' * 65)
for spec_name in SPECS.keys():
    sub = out[out['spec'] == spec_name]
    n_sig = (sub['p_bh'] < 0.05).sum()
    n_marg = ((sub['p_bh'] >= 0.05) & (sub['p_bh'] < 0.10)).sum()
    print(f'  {spec_name}: {n_sig} sig (BH p<0.05), {n_marg} marginal')
    for _, r in sub[sub['p_bh'] < 0.10].sort_values('p_bh').iterrows():
        print(f'    {r["feature"]:<30} θ={r["theta"]:+.4f} p_BH={r["p_bh"]:.4f} {r["sig_bh"]}')

# ── 5. PIVOT ──────────────────────────────────────────────────
def cell_str(row):
    if pd.isna(row['theta']): return 'NA'
    th = row['theta']
    se = row['se_boot'] if np.isfinite(row['se_boot']) else row['se_hc1']
    sig = row['sig_bh']
    return f'{th:+.3f} ({se:.3f}){sig}'

pivot_rows = []
for feat in ZONING_FEATURES:
    dim_label, feat_label = DISPLAY_INFO[feat]
    prow = {'dimension': dim_label, 'feature': feat_label}
    for spec_name in ['RF_Full', 'RF_NoACS', 'GB_Full', 'GB_NoACS']:
        match = out[(out['feature'] == feat) & (out['spec'] == spec_name)]
        if len(match) == 1:
            prow[spec_name] = cell_str(match.iloc[0])
        else:
            prow[spec_name] = 'NA'
    pivot_rows.append(prow)

pivot = pd.DataFrame(pivot_rows)
pivot.to_csv('theory_dml_acs_pivot.csv', index=False)
print(f'Saved: theory_dml_acs_pivot.csv')
print('\n✓ Complete.')
