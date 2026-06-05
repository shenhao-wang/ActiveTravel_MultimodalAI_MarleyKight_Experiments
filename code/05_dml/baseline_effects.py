"""
Baseline effects of zoning features on active travel:
logistic regression coefficients vs. random forest marginal effects.

Uses zone_theory_expanded_raw.csv (22 granular D1–D7 features per zone class),
area-weighted to block-group level. Produces CSV for the longtable in the paper.

Panel A: Logistic regression log-odds with HC1 robust SEs (statsmodels)
Panel B: Random forest marginal effects via finite-difference, bootstrap SEs

Output:
  baseline_zoning_effects_expanded.csv   — all variables (controls + zoning)
"""

import warnings
# WHY: Suppresses sklearn convergence warnings for SLURM logs
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# ── CONFIGURATION ──────────────────────────────────────────────
TRIPS_PATH    = 'processed_data/tampatrips_1.csv'
ZON_CSV_PATH  = 'processed_data/tpazoning_clean.csv'
THEORY_PATH   = 'processed_data/zone_theory_expanded_raw.csv'
ACS_PATH      = 'processed_data/acs_blockgroup_tampa.csv'

RANDOM_STATE  = 16
N_BOOT        = 200   # bootstrap reps for RF marginal effects

BASE_FEATURES = ['age', 'gender', 'education', 'income_detailed', 'num_veh',
                 'hhsize', 'trip_distance', 'trip_duration',
                 'd_purpose_category_imputed']

ACS_FEATURES  = ['median_hh_income', 'pct_bachelors_plus',
                 'pct_white_nonhisp', 'pct_black_nonhisp', 'pct_hispanic',
                 'pct_poverty', 'pct_owner_occ']

# Zoning features from zone_theory_expanded_raw.csv (skip zone_class, context)
# mixed_use_perm is constant=1 across all 56 zones → drop (zero variance)
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

# For the LaTeX table: display names matching the longtable rows
DISPLAY_NAMES = {
    # Sociodemographics
    'age':                        'Age',
    'gender':                     'Gender',
    'education':                  'Education (1–5)',
    'income_detailed':            'HH income (1–8)',
    'num_veh':                    'Vehicle ownership',
    'hhsize':                     'Household size',
    # Travel
    'trip_distance':              'Trip distance (mi)',
    'trip_duration':              'Trip duration (min)',
    'd_purpose_category_imputed': 'Destination purpose (categorical)',
    # ACS
    'median_hh_income':           'Median HH income (BG)',
    'pct_bachelors_plus':         'Pct. bachelor\'s+ (BG)',
    'pct_white_nonhisp':          'Pct. white non-Hispanic (BG)',
    'pct_black_nonhisp':          'Pct. Black non-Hispanic (BG)',
    'pct_hispanic':               'Pct. Hispanic (BG)',
    'pct_poverty':                'Pct. poverty (BG)',
    'pct_owner_occ':              'Pct. owner-occupied (BG)',
    # D1: Density
    'density_permissions':        'Density permissions (0–10)',
    'min_lot_area':               'Min. lot area, sq ft (inv.)',
    'max_height':                 'Max. height, ft',
    # D2: Use Mix & Flexibility
    'use_mixing':                 'Use mixing (0–10)',
    'use_flexibility':            'Use flexibility (0–10)',
    # D3: Parking
    'parking_intensity':          'Parking intensity (0–10, inv.)',
    'reduced_parking':            'Reduced parking (binary)',
    # D4: Pedestrian Design
    'ped_street_interface':       'Ped. street interface (0–10)',
    'transparency_activation':    'Transparency & activation (0–10)',
    'front_setback':              'Front setback, ft (inv.)',
    'lot_width':                  'Lot width, ft (inv.)',
    'frontage_std':               'Frontage standard (binary)',
    # D5: Building Placement
    'setback_building_placement': 'Setback/bldg. placement (0–10)',
    'side_setback':               'Side setback, ft (inv.)',
    'rear_setback':               'Rear setback, ft (inv.)',
    # D6: Regulatory Mode
    'form_based_design':          'Form-based design (0–10)',
    'human_scale_design':         'Human-scale design (binary)',
    'ped_scale_lang':             'Ped.-scale language (binary)',
    # D7: Transit Orientation
    'transit_orientation':        'Transit orientation (0–10)',
    'transit_orient':             'Transit-oriented (binary)',
    # D+: Open Space
    'open_space_green':           'Open space & green (0–10)',
}

# Category labels for grouping in the output
CATEGORY = {
    'age': 'Sociodemographics', 'gender': 'Sociodemographics',
    'education': 'Sociodemographics', 'income_detailed': 'Sociodemographics',
    'num_veh': 'Sociodemographics', 'hhsize': 'Sociodemographics',
    'trip_distance': 'Travel Attributes', 'trip_duration': 'Travel Attributes',
    'd_purpose_category_imputed': 'Travel Attributes',
    'median_hh_income': 'ACS Block Group', 'pct_bachelors_plus': 'ACS Block Group',
    'pct_white_nonhisp': 'ACS Block Group', 'pct_black_nonhisp': 'ACS Block Group',
    'pct_hispanic': 'ACS Block Group', 'pct_poverty': 'ACS Block Group',
    'pct_owner_occ': 'ACS Block Group',
    'density_permissions': 'D1: Density', 'min_lot_area': 'D1: Density',
    'max_height': 'D1: Density',
    'use_mixing': 'D2: Use Mix & Flexibility', 'use_flexibility': 'D2: Use Mix & Flexibility',
    'parking_intensity': 'D3: Parking', 'reduced_parking': 'D3: Parking',
    'ped_street_interface': 'D4: Pedestrian Design',
    'transparency_activation': 'D4: Pedestrian Design',
    'front_setback': 'D4: Pedestrian Design', 'lot_width': 'D4: Pedestrian Design',
    'frontage_std': 'D4: Pedestrian Design',
    'setback_building_placement': 'D5: Building Placement',
    'side_setback': 'D5: Building Placement', 'rear_setback': 'D5: Building Placement',
    'form_based_design': 'D6: Regulatory Mode', 'human_scale_design': 'D6: Regulatory Mode',
    'ped_scale_lang': 'D6: Regulatory Mode',
    'transit_orientation': 'D7: Transit Orientation',
    'transit_orient': 'D7: Transit Orientation',
    'open_space_green': 'D+: Open Space',
}

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

# ── 1. LOAD DATA ───────────────────────────────────────────────
print('Loading data...')
trips     = pd.read_csv(TRIPS_PATH)
zon_csv   = pd.read_csv(ZON_CSV_PATH).dropna(subset=['ZONECLASS'])
theory    = pd.read_csv(THEORY_PATH)
acs       = pd.read_csv(ACS_PATH)

trips['active_travel'] = trips['mode_type'].isin([1, 2]).astype(int)
area_col = ('ShapeSTAre_corrected' if 'ShapeSTAre_corrected' in zon_csv.columns
            else 'ShapeSTAre')

# ── 2. AREA-WEIGHTED BG FEATURES (theory-expanded) ────────────
print('Building area-weighted block-group features...')

# Merge zone-level theory features onto parcel-level zoning
zon_m = zon_csv.merge(theory[['zone_class'] + ZONING_FEATURES],
                      left_on='ZONECLASS', right_on='zone_class', how='left')

def weighted_mean(group, val_col):
    mask = group[val_col].notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(group.loc[mask, val_col], weights=group.loc[mask, area_col])

bg_rows = []
for geoid, grp in zon_m.groupby('GEOID'):
    row = {'GEOID': geoid}
    for feat in ZONING_FEATURES:
        row[feat] = weighted_mean(grp, feat)
    bg_rows.append(row)

bg_theory = pd.DataFrame(bg_rows)

# Dominant zone for context
dominant = (zon_csv.sort_values(area_col, ascending=False)
            .drop_duplicates(subset='GEOID', keep='first')[['GEOID','ZONECLASS']]
            .rename(columns={'ZONECLASS': 'dominant_zone'}))

bg = (dominant
      .merge(bg_theory, on='GEOID', how='left')
      .merge(acs[['GEOID'] + ACS_FEATURES], on='GEOID', how='left'))

df = trips.merge(bg, left_on='o_bg', right_on='GEOID', how='left')

# Drop rows missing all zoning (no BG match)
zoning_avail = [f for f in ZONING_FEATURES if f in df.columns]
df = df.dropna(subset=zoning_avail[:1]).reset_index(drop=True)
print(f'Sample: {len(df):,} trips  |  AT rate: {df["active_travel"].mean():.1%}')

# ── 3. CHECK FOR ZERO/NEAR-ZERO VARIANCE ──────────────────────
drop_feats = []
for f in ZONING_FEATURES:
    if f in df.columns:
        nuniq = df[f].dropna().nunique()
        if nuniq <= 1:
            print(f'  Dropping {f}: constant or single-value after area-weighting')
            drop_feats.append(f)

ZONING_FEATURES_FINAL = [f for f in ZONING_FEATURES if f not in drop_feats]
print(f'Zoning features retained: {len(ZONING_FEATURES_FINAL)}')

# ── 4. PREPARE MATRICES ───────────────────────────────────────
all_features = BASE_FEATURES + ACS_FEATURES + ZONING_FEATURES_FINAL
Y = df['active_travel'].values

imp = SimpleImputer(strategy='median')
X_raw = imp.fit_transform(df[all_features].values.astype(float))

scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

n_controls = len(BASE_FEATURES) + len(ACS_FEATURES)
n_zoning   = len(ZONING_FEATURES_FINAL)
n_total    = len(all_features)

print(f'Features: {n_controls} controls + {n_zoning} zoning = {n_total} total')

# ── 5. PANEL A: LOGISTIC REGRESSION ───────────────────────────
# WHY OLS/logistic as naive benchmark: Naive baseline for comparison with
# DML; not causal. Logistic coefficients are log-odds, not marginal effects.
print('\n── Panel A: Logistic Regression (HC1 robust SEs) ──')
X_sm = sm.add_constant(X_std)
logit_result = sm.Logit(Y, X_sm).fit(cov_type='HC1', disp=0, maxiter=1000)

lr_data = {}
for i, feat in enumerate(all_features):
    idx = i + 1  # +1 for constant
    lr_data[feat] = {
        'LR_Estimate': logit_result.params[idx],
        'LR_SE':       logit_result.bse[idx],
        'LR_p':        logit_result.pvalues[idx],
    }

# ── 6. PANEL B: RANDOM FOREST MARGINAL EFFECTS (ALL VARS) ────
print(f'\n── Panel B: Random Forest Marginal Effects (B={N_BOOT}) ──')

def rf_marginal_effects_all(X, y, n_boot, seed):
    """
    WHY RF marginal effects via finite-difference: Nonparametric alternative
    to logistic coefficients. Captures nonlinear and interaction effects that
    logistic regression misses.
    ME_j = mean[ P(Y=1|X_j+d) - P(Y=1|X_j-d) ] / (2d), d=0.5 SD.
    Bootstrap SEs from resampling.
    """
    delta = 0.5
    n, p = X.shape
    rng = np.random.RandomState(seed)

    def compute_me(X_tr, y_tr, X_eval):
        rf = RandomForestClassifier(n_estimators=500, max_depth=12,
                                     min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        mes = np.zeros(p)
        for j in range(p):
            Xp = X_eval.copy(); Xp[:, j] += delta
            Xm = X_eval.copy(); Xm[:, j] -= delta
            mes[j] = np.mean((rf.predict_proba(Xp)[:, 1] -
                              rf.predict_proba(Xm)[:, 1]) / (2 * delta))
        return mes

    me_point = compute_me(X, y, X)

    boot_mes = np.zeros((n_boot, p))
    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f'    bootstrap {b+1}/{n_boot}')
        idx = rng.choice(n, size=n, replace=True)
        boot_mes[b] = compute_me(X[idx], y[idx], X[idx])

    boot_se = np.std(boot_mes, axis=0, ddof=1)
    return me_point, boot_se

me_point, me_se = rf_marginal_effects_all(X_std, Y, N_BOOT, RANDOM_STATE)

rf_data = {}
for i, feat in enumerate(all_features):
    est = me_point[i]
    se  = me_se[i]
    if se > 0:
        z = est / se
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        pval = np.nan
    rf_data[feat] = {'RF_Estimate': est, 'RF_SE': se, 'RF_p': pval}

# ── 7. ASSEMBLE OUTPUT ────────────────────────────────────────
def stars(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return ''

rows = []
for feat in all_features:
    row = {
        'Variable':     feat,
        'Display':      DISPLAY_NAMES.get(feat, feat),
        'Category':     CATEGORY.get(feat, ''),
        'LR_Estimate':  round(lr_data[feat]['LR_Estimate'], 6),
        'LR_SE':        round(lr_data[feat]['LR_SE'], 6),
        'LR_p':         round(lr_data[feat]['LR_p'], 6),
        'LR_sig':       stars(lr_data[feat]['LR_p']),
        'RF_Estimate':  round(rf_data[feat]['RF_Estimate'], 6),
        'RF_SE':        round(rf_data[feat]['RF_SE'], 6),
        'RF_p':         round(rf_data[feat]['RF_p'], 6),
        'RF_sig':       stars(rf_data[feat]['RF_p']),
    }
    rows.append(row)

out = pd.DataFrame(rows)

# ── 8. SAVE ───────────────────────────────────────────────────
out_path = 'baseline_zoning_effects_expanded.csv'
out.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')

# Print summary
print('\n── Zoning features summary ──')
zoning_out = out[out['Variable'].isin(ZONING_FEATURES_FINAL)]
for _, r in zoning_out.iterrows():
    print(f"  {r['Display']:<38} LR: {r['LR_Estimate']:+.4f}{r['LR_sig']:<4}  "
          f"RF: {r['RF_Estimate']:+.6f}{r['RF_sig']:<4}")

# ── 9. MODEL FIT ──────────────────────────────────────────────
print(f'\n── Model Fit ──')
print(f'  LR pseudo-R²:  {logit_result.prsquared:.4f}')
print(f'  LR AIC:        {logit_result.aic:.1f}')
print(f'  N:             {logit_result.nobs:.0f}')

rf_full = RandomForestClassifier(n_estimators=500, max_depth=12,
                                  min_samples_leaf=5,
                                  random_state=RANDOM_STATE, n_jobs=-1)
cv_acc = cross_val_score(rf_full, X_std, Y, cv=5, scoring='accuracy')
cv_auc = cross_val_score(rf_full, X_std, Y, cv=5, scoring='roc_auc')
print(f'  RF 5-fold accuracy:  {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
print(f'  RF 5-fold AUC:       {cv_auc.mean():.4f} ± {cv_auc.std():.4f}')

# Append model-fit row to CSV
fit_row = pd.DataFrame([{
    'Variable': '_model_fit',
    'Display': 'Model fit',
    'Category': 'Model fit',
    'LR_Estimate': logit_result.prsquared,
    'LR_SE': logit_result.nobs,
    'LR_p': logit_result.aic,
    'LR_sig': '',
    'RF_Estimate': cv_auc.mean(),
    'RF_SE': cv_acc.mean(),
    'RF_p': np.nan,
    'RF_sig': '',
}])
out_with_fit = pd.concat([out, fit_row], ignore_index=True)
out_with_fit.to_csv(out_path, index=False)
print(f'Updated: {out_path} (with model fit row)')