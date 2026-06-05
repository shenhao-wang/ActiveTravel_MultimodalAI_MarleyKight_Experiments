"""
Baseline effects of zoning features on active travel:
logistic regression coefficients vs. random forest marginal effects.

Uses zone_theory_expanded_raw.csv (22 granular D1–D7 features per zone class),
area-weighted to block-group level. Produces CSV for the longtable in the paper.

Panel A: Logistic regression log-odds with HC1 robust SEs (statsmodels)
Panel B: Random forest marginal effects via finite-difference, bootstrap SEs

Now also computes: log loss, accuracy, and AUC for both panels (5-fold CV).

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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
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

# ── 8. MODEL FIT: LOG LOSS, ACCURACY, AUC (5-fold CV) ─────────
print('\n── Computing model fit metrics (5-fold stratified CV) ──')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --- Panel A: Logistic Regression (sklearn for CV-compatible predictions) ---
lr_sklearn = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                solver='lbfgs')

# Get out-of-fold predicted probabilities
lr_proba = cross_val_predict(lr_sklearn, X_std, Y, cv=cv, method='predict_proba')
lr_preds = (lr_proba[:, 1] >= 0.5).astype(int)

lr_logloss = log_loss(Y, lr_proba)
lr_acc     = accuracy_score(Y, lr_preds)
lr_auc     = roc_auc_score(Y, lr_proba[:, 1])

print(f'  LR  — Log Loss: {lr_logloss:.4f}  Accuracy: {lr_acc:.4f}  AUC: {lr_auc:.4f}')

# --- Panel B: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=500, max_depth=12,
                                   min_samples_leaf=5,
                                   random_state=RANDOM_STATE, n_jobs=-1)

rf_proba = cross_val_predict(rf_model, X_std, Y, cv=cv, method='predict_proba')
rf_preds = (rf_proba[:, 1] >= 0.5).astype(int)

rf_logloss = log_loss(Y, rf_proba)
rf_acc     = accuracy_score(Y, rf_preds)
rf_auc     = roc_auc_score(Y, rf_proba[:, 1])

print(f'  RF  — Log Loss: {rf_logloss:.4f}  Accuracy: {rf_acc:.4f}  AUC: {rf_auc:.4f}')

# --- Also get per-fold values for reporting SD ---
lr_fold_ll, lr_fold_acc, lr_fold_auc = [], [], []
rf_fold_ll, rf_fold_acc, rf_fold_auc = [], [], []

for train_idx, test_idx in cv.split(X_std, Y):
    X_tr, X_te = X_std[train_idx], X_std[test_idx]
    y_tr, y_te = Y[train_idx], Y[test_idx]

    # LR
    lr_f = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                               solver='lbfgs')
    lr_f.fit(X_tr, y_tr)
    lr_p = lr_f.predict_proba(X_te)
    lr_fold_ll.append(log_loss(y_te, lr_p))
    lr_fold_acc.append(accuracy_score(y_te, (lr_p[:, 1] >= 0.5).astype(int)))
    lr_fold_auc.append(roc_auc_score(y_te, lr_p[:, 1]))

    # RF
    rf_f = RandomForestClassifier(n_estimators=500, max_depth=12,
                                   min_samples_leaf=5,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    rf_f.fit(X_tr, y_tr)
    rf_p = rf_f.predict_proba(X_te)
    rf_fold_ll.append(log_loss(y_te, rf_p))
    rf_fold_acc.append(accuracy_score(y_te, (rf_p[:, 1] >= 0.5).astype(int)))
    rf_fold_auc.append(roc_auc_score(y_te, rf_p[:, 1]))

print(f'\n  Per-fold breakdown:')
print(f'  LR  — LL: {np.mean(lr_fold_ll):.4f}±{np.std(lr_fold_ll):.4f}  '
      f'Acc: {np.mean(lr_fold_acc):.4f}±{np.std(lr_fold_acc):.4f}  '
      f'AUC: {np.mean(lr_fold_auc):.4f}±{np.std(lr_fold_auc):.4f}')
print(f'  RF  — LL: {np.mean(rf_fold_ll):.4f}±{np.std(rf_fold_ll):.4f}  '
      f'Acc: {np.mean(rf_fold_acc):.4f}±{np.std(rf_fold_acc):.4f}  '
      f'AUC: {np.mean(rf_fold_auc):.4f}±{np.std(rf_fold_auc):.4f}')

# ── 9. STATSMODELS FIT STATS ──────────────────────────────────
print(f'\n── Statsmodels LR fit ──')
print(f'  Pseudo-R²:  {logit_result.prsquared:.4f}')
print(f'  AIC:        {logit_result.aic:.1f}')
print(f'  N:          {logit_result.nobs:.0f}')

# ── 10. SAVE ──────────────────────────────────────────────────
out_path = 'baseline_zoning_effects_expanded.csv'
out.to_csv(out_path, index=False)

# Also save model fit as separate CSV for easy reference
fit_df = pd.DataFrame([
    {'Panel': 'A: Logistic Regression',
     'Pseudo_R2': round(logit_result.prsquared, 4),
     'CV_LogLoss': round(lr_logloss, 4),
     'CV_LogLoss_SD': round(np.std(lr_fold_ll), 4),
     'CV_Accuracy': round(lr_acc, 4),
     'CV_Accuracy_SD': round(np.std(lr_fold_acc), 4),
     'CV_AUC': round(lr_auc, 4),
     'CV_AUC_SD': round(np.std(lr_fold_auc), 4),
     'N': int(logit_result.nobs),
     'AIC': round(logit_result.aic, 1)},
    {'Panel': 'B: Random Forest',
     'Pseudo_R2': np.nan,
     'CV_LogLoss': round(rf_logloss, 4),
     'CV_LogLoss_SD': round(np.std(rf_fold_ll), 4),
     'CV_Accuracy': round(rf_acc, 4),
     'CV_Accuracy_SD': round(np.std(rf_fold_acc), 4),
     'CV_AUC': round(rf_auc, 4),
     'CV_AUC_SD': round(np.std(rf_fold_auc), 4),
     'N': len(Y),
     'AIC': np.nan},
])
fit_path = 'baseline_model_fit.csv'
fit_df.to_csv(fit_path, index=False)

print(f'\nSaved: {out_path}')
print(f'Saved: {fit_path}')

# Print summary
print('\n── Zoning features summary ──')
zoning_out = out[out['Variable'].isin(ZONING_FEATURES_FINAL)]
for _, r in zoning_out.iterrows():
    print(f"  {r['Display']:<38} LR: {r['LR_Estimate']:+.4f}{r['LR_sig']:<4}  "
          f"RF: {r['RF_Estimate']:+.6f}{r['RF_sig']:<4}")

print(f'\n── Model Fit Summary ──')
print(f'  {"Panel":<30} {"Log Loss":>10} {"Accuracy":>10} {"AUC":>10}')
print(f'  {"─"*62}')
print(f'  {"A: Logistic Regression":<30} {lr_logloss:>10.4f} {lr_acc:>10.4f} {lr_auc:>10.4f}')
print(f'  {"B: Random Forest":<30} {rf_logloss:>10.4f} {rf_acc:>10.4f} {rf_auc:>10.4f}')
print(f'\n  LR Pseudo-R²: {logit_result.prsquared:.4f}  |  RF OOB AUC: {rf_auc:.4f}')

print(f'\n✓ Done.')
