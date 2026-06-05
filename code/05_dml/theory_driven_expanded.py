"""
Tampa Zoning — Expanded Theory-Driven Feature Assembly
========================================================
Combines LLM-scored rubric features (from llm_zoning_features.py)
with scraped numeric standards (from zone_features_combined.csv)
into a single expanded theory-driven treatment vector organized
under 7 thematic dimensions.


Current feature inventory (22 features across 7+1 dimensions):

  D1: Density & Intensity (3 features)
    - LLM density_permissions score (0-10)
    - min_lot_area (sq ft, from Table 4-2)
    - max_height (feet, from Table 4-2)

  D2a: Use Mixing & Diversity (2 features)
    - LLM use_mixing score (0-10)
    - mixed_use_perm (binary, from ped rubric)

  D2b: Use Flexibility (1 feature)
    - LLM use_flexibility score (0-10)

  D3: Parking Requirements (2 features)
    - LLM parking_intensity score (0-10)
    - reduced_parking (binary, from ped rubric)

  D4a: Ped-Street Interface (4 features)
    - LLM ped_street_interface score (0-10)
    - front_setback (feet, from Table 4-2)
    - lot_width (feet, from Table 4-2)
    - frontage_std (binary, from ped rubric)

  D4b: Ground-Floor Transparency & Activation (1 feature)
    - LLM transparency_activation score (0-10)

  D5: Setback & Building Placement (4 features)
    - LLM setback_building_placement score (0-10)
    - front_setback (feet) [shared with D4a]
    - side_setback (feet, from Table 4-2)
    - rear_setback (feet, from Table 4-2)

  D6: Regulatory Mode / Form-Based Design (3 features)
    - LLM form_based_design score (0-10)
    - human_scale_design (binary, from ped rubric)
    - ped_scale_lang (binary, from ped rubric)

  D7: Transit & Pedestrian Overlay (2 features)
    - LLM transit_orientation score (0-10)
    - transit_orient (binary, from ped rubric)

  D+: Open Space & Green Infrastructure (1 feature)
    - LLM open_space_green score (0-10)

ADDING NEW FEATURES:
  1. Add it to zone_features_combined.csv (or a new CSV)
  2. Add a row to the FEATURE_REGISTRY below
  3. Re-run this script

Output:
  zone_embeddings_theory_expanded.csv  — DML-ready (standardized)
  zone_theory_expanded_map.csv         — maps each feature to its dimension
  zone_theory_expanded_raw.csv         — unstandardized values for inspection

Usage:
  python theory_driven_expanded.py
  python theory_driven_expanded.py --llm-raw zone_llm_features_raw.csv
  python theory_driven_expanded.py --extra-features my_new_features.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# FEATURE REGISTRY
# ─────────────────────────────────────────────────────────────
# Each entry: (feature_name, dimension_id, dimension_label, source, description, invert)
#
# WHY 22 features across 7 dimensions: Each dimension has both LLM holistic
# scores AND scraped binary/numeric standards. Within-dimension sign reversals
# (e.g. D4a parking) are informative, not errors to reconcile.
#
# source: 'llm' = from zone_llm_features_raw.csv
#         'scraped' = from zone_features_combined.csv
#         'extra' = from --extra-features CSV
#
# WHY LLM scores on a coarse scale: Coarse enough to be interpretable,
# fine enough to discriminate. Rationales required from LLM to force
# grounded scoring.
#
# WHY scraped dimensional standards: Complement LLM scores with objective
# numeric values from LDC Table 4-2.
#
# invert: True if higher values mean LESS pedestrian-friendly
#         (e.g., min_lot_area: bigger lots = more auto-oriented)
#         Inverted features are multiplied by -1 before standardization
#         so that positive values always mean more AT-promoting.

FEATURE_REGISTRY = [
    # ── D1: Density & Intensity ──
    ('density_permissions', 'D1', 'Density & Intensity',
     'llm', 'LLM holistic density score (0-10)', False),
    ('min_lot_area', 'D1', 'Density & Intensity',
     'scraped', 'Minimum lot area (sq ft) from Table 4-2', True),
    ('max_height', 'D1', 'Density & Intensity',
     'scraped', 'Maximum building height (feet) from Table 4-2', False),

    # ── D2a: Use Mixing & Diversity ──
    ('use_mixing', 'D2a', 'Use Mixing & Diversity',
     'llm', 'LLM holistic use mixing score (0-10)', False),
    ('mixed_use_perm', 'D2a', 'Use Mixing & Diversity',
     'scraped', 'Mixed-use permitted (binary)', False),

    # ── D2b: Use Flexibility ──
    ('use_flexibility', 'D2b', 'Use Flexibility',
     'llm', 'LLM holistic use flexibility score (0-10)', False),

    # ── D3: Parking Requirements ──
    ('parking_intensity', 'D3', 'Parking Requirements',
     'llm', 'LLM parking intensity score (0-10, higher=more auto)', True),
    ('reduced_parking', 'D3', 'Parking Requirements',
     'scraped', 'Reduced parking provisions present (binary)', False),

    # ── D4a: Pedestrian-Street Interface ──
    ('ped_street_interface', 'D4a', 'Pedestrian-Street Interface',
     'llm', 'LLM pedestrian-street interface score (0-10)', False),
    ('front_setback', 'D4a', 'Pedestrian-Street Interface',
     'scraped', 'Front setback (feet) from Table 4-2', True),
    ('lot_width', 'D4a', 'Pedestrian-Street Interface',
     'scraped', 'Lot width (feet) from Table 4-2', True),
    ('frontage_std', 'D4a', 'Pedestrian-Street Interface',
     'scraped', 'Frontage standards present (binary)', False),

    # ── D4b: Ground-Floor Transparency & Activation ──
    ('transparency_activation', 'D4b', 'Transparency & Activation',
     'llm', 'LLM transparency/activation score (0-10)', False),

    # ── D5: Setback & Building Placement ──
    ('setback_building_placement', 'D5', 'Setback & Building Placement',
     'llm', 'LLM overall setback regime score (0-10)', False),
    ('side_setback', 'D5', 'Setback & Building Placement',
     'scraped', 'Side setback (feet) from Table 4-2', True),
    ('rear_setback', 'D5', 'Setback & Building Placement',
     'scraped', 'Rear setback (feet) from Table 4-2', True),

    # ── D6: Regulatory Mode (Form-Based vs. Use-Based) ──
    ('form_based_design', 'D6', 'Form-Based Design',
     'llm', 'LLM form-based design score (0-10)', False),
    ('human_scale_design', 'D6', 'Form-Based Design',
     'scraped', 'Human-scale design standards present (binary)', False),
    ('ped_scale_lang', 'D6', 'Form-Based Design',
     'scraped', 'Pedestrian-scale language present (binary)', False),

    # ── D7: Transit & Pedestrian Overlay ──
    ('transit_orientation', 'D7', 'Transit Orientation',
     'llm', 'LLM transit orientation score (0-10)', False),
    ('transit_orient', 'D7', 'Transit Orientation',
     'scraped', 'Transit-oriented provisions present (binary)', False),

    # ── D+: Open Space & Green Infrastructure ──
    ('open_space_green', 'D+', 'Open Space & Green Infra.',
     'llm', 'LLM open space/green infrastructure score (0-10)', False),
]


# ─────────────────────────────────────────────────────────────
# ZONE CONTEXT
# ─────────────────────────────────────────────────────────────
RESIDENTIAL = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO','RO-1','SH-RS','SH-RS-A','SH-RM','SH-RO','SH-PD',
    'YC-2','YC-4','YC-8','YC-9','RM-24/18',
]
NON_RESIDENTIAL = [
    'CG','CI','CN','OP','OP-1','IG','IH','CBD-1','CBD-2',
    'CD-1','CD-2','CD-3','NMU-35','NMU-24','NMU-16',
    'SH-CG','SH-CI','SH-CN','YC-1','YC-3','YC-5','YC-6','YC-7',
]


def get_context(zone):
    if zone in RESIDENTIAL:
        return 'Residential'
    elif zone in NON_RESIDENTIAL:
        return 'Non-Residential'
    return 'Other/PD'


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build expanded theory-driven embedding')
    parser.add_argument('--llm-raw', type=str, default='zone_llm_features_raw.csv',
                        help='Path to LLM raw scores CSV')
    parser.add_argument('--scraped', type=str, default='zone_features_combined.csv',
                        help='Path to scraped features CSV')
    parser.add_argument('--extra-features', type=str, default=None,
                        help='Path to additional features CSV (must have zone_class column)')
    parser.add_argument('--output-prefix', type=str, default='zone',
                        help='Prefix for output filenames (default: zone)')
    args = parser.parse_args()

    print('=' * 65)
    print('  Tampa Zoning — Expanded Theory-Driven Feature Assembly')
    print('=' * 65)

    # ── Load LLM scores
    llm_df = pd.read_csv(args.llm_raw)
    llm_cols = [c for c in llm_df.columns
                if not c.endswith('_rationale') and c not in ['zone_class', 'context']]
    print(f'Loaded LLM scores: {len(llm_df)} zones × {len(llm_cols)} dimensions')
    print(f'  LLM columns: {llm_cols}')

    # ── Load scraped features
    scraped_df = pd.read_csv(args.scraped)
    print(f'Loaded scraped features: {len(scraped_df)} zones × {len(scraped_df.columns)-1} features')

    # ── Load extra features if provided
    extra_df = None
    if args.extra_features and os.path.exists(args.extra_features):
        extra_df = pd.read_csv(args.extra_features)
        print(f'Loaded extra features: {len(extra_df)} zones × {len(extra_df.columns)-1} features')

    # ── Merge all sources on zone_class
    merged = llm_df[['zone_class'] + llm_cols].merge(
        scraped_df, on='zone_class', how='outer')
    if extra_df is not None:
        merged = merged.merge(extra_df, on='zone_class', how='left')

    print(f'\nMerged: {len(merged)} zones')

    # ── Assemble features from registry
    print(f'\nAssembling {len(FEATURE_REGISTRY)} features across dimensions...')
    feature_cols = []
    feature_map = []
    missing_features = []

    for feat_name, dim_id, dim_label, source, description, invert in FEATURE_REGISTRY:
        if feat_name not in merged.columns:
            missing_features.append((feat_name, source))
            continue

        # Create standardized column name: emb_NNN
        col_idx = len(feature_cols)
        col_name = f'emb_{col_idx:03d}'
        feature_cols.append(col_name)

        vals = merged[feat_name].astype(float).values.copy()

        # Invert if higher = less pedestrian-friendly
        if invert:
            vals = -vals

        merged[col_name] = vals

        feature_map.append({
            'col': col_name,
            'feature': feat_name,
            'dimension_id': dim_id,
            'dimension_label': dim_label,
            'source': source,
            'description': description,
            'inverted': invert,
            'n_unique': merged[feat_name].nunique(),
            'pct_missing': merged[feat_name].isna().mean() * 100,
        })

        status = '(inverted)' if invert else ''
        print(f'  {col_name} ← {feat_name:<30} [{dim_id}] {status}')

    if missing_features:
        print(f'\n  WARNING: {len(missing_features)} features not found:')
        for fname, src in missing_features:
            print(f'    {fname} (expected from {src})')

    print(f'\nTotal features assembled: {len(feature_cols)}')

    # ── Report dimension summary
    map_df = pd.DataFrame(feature_map)
    print(f'\n── Dimension Summary ──')
    for dim_id in map_df['dimension_id'].unique():
        sub = map_df[map_df.dimension_id == dim_id]
        label = sub.dimension_label.iloc[0]
        n_feat = len(sub)
        sources = sub.source.value_counts().to_dict()
        src_str = ', '.join(f'{v} {k}' for k, v in sources.items())
        print(f'  {dim_id:<5} {label:<30} {n_feat} features ({src_str})')

    # ── Save raw (unstandardized) for inspection
    raw_out = merged[['zone_class'] + feature_cols].copy()
    raw_out.insert(1, 'context', merged['zone_class'].apply(get_context))
    # Replace emb_ names with feature names for readability
    rename_raw = {row['col']: row['feature'] for _, row in map_df.iterrows()}
    raw_out_readable = raw_out.rename(columns=rename_raw)
    raw_out_readable.to_csv(f'{args.output_prefix}_theory_expanded_raw.csv', index=False)
    print(f'\nSaved: {args.output_prefix}_theory_expanded_raw.csv (unstandardized)')

    # ── Standardize to zero-mean unit-variance
    # WHY standardize: DML treatment vector needs comparable scales across
    # heterogeneous features (LLM 0-10, setbacks in feet, binaries 0/1).
    emb_out = merged[['zone_class'] + feature_cols].copy()
    emb_out.insert(1, 'context', merged['zone_class'].apply(get_context))

    for col in feature_cols:
        vals = emb_out[col].values.astype(float)
        # Fill NaN with median before standardizing
        median_val = np.nanmedian(vals)
        vals = np.where(np.isnan(vals), median_val, vals)
        mu = np.mean(vals)
        sd = np.std(vals)
        if sd > 0:
            emb_out[col] = (vals - mu) / sd
        else:
            emb_out[col] = 0.0

    emb_out.to_csv(f'{args.output_prefix}_embeddings_theory_expanded.csv', index=False)
    print(f'Saved: {args.output_prefix}_embeddings_theory_expanded.csv (standardized, DML-ready)')

    # ── Save feature map
    map_df.to_csv(f'{args.output_prefix}_theory_expanded_map.csv', index=False)
    print(f'Saved: {args.output_prefix}_theory_expanded_map.csv')

    # ── Final summary
    n_dims = map_df['dimension_id'].nunique()
    n_feats = len(feature_cols)
    n_llm = (map_df.source == 'llm').sum()
    n_scraped = (map_df.source == 'scraped').sum()
    n_extra = (map_df.source == 'extra').sum()
    n_inverted = map_df['inverted'].sum()

    print(f'\n── Final Summary ──')
    print(f'  Dimensions: {n_dims}')
    print(f'  Total features: {n_feats} ({n_llm} LLM + {n_scraped} scraped + {n_extra} extra)')
    print(f'  Inverted features: {n_inverted} (higher raw value = less AT-friendly)')
    print(f'  Zones: {len(emb_out)}')
    print(f'\n  Output files:')
    print(f'    {args.output_prefix}_embeddings_theory_expanded.csv  → feed to dml_multi_backend.py')
    print(f'    {args.output_prefix}_theory_expanded_raw.csv         → human-readable scores')
    print(f'    {args.output_prefix}_theory_expanded_map.csv         → dimension mapping')

    print(f'\n  To add new features:')
    print(f'    1. Add column to zone_features_combined.csv (or a new CSV)')
    print(f'    2. Add entry to FEATURE_REGISTRY in this script')
    print(f'    3. Re-run: python theory_driven_expanded.py')

    print(f'\n  To run DML on expanded features:')
    print(f'    Add to ALL_BACKENDS in dml_multi_backend.py:')
    print(f"    ('theory_exp', '{args.output_prefix}_embeddings_theory_expanded.csv', 'Theory-Driven Expanded')")

    print(f'\n✓ Expanded theory-driven feature assembly complete.')


if __name__ == '__main__':
    main()