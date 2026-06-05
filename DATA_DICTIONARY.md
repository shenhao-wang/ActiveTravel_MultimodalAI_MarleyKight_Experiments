# Data Dictionary

Column definitions for all CSV files in `processed_data/` and `tables/`.
Files in `raw_data/` are documented in their original source codebooks.

---

## processed_data/

### tpazoning_clean.csv (6,238 rows)

Parcel-level zoning assignments joined to Census geography.

| Column | Type | Description |
|--------|------|-------------|
| STATEFP | str | State FIPS code (12 = Florida) |
| COUNTYFP | str | County FIPS code (057 = Hillsborough) |
| TRACTCE | str | Census tract code |
| BLKGRPCE | str | Block group code |
| GEOID | str | Full block group GEOID (state+county+tract+blkgrp) |
| GEOIDFQ | str | Fully qualified GEOID |
| NAMELSAD | str | Block group name |
| OBJECTID | int | Parcel object ID from city GIS |
| ZONECLASS | str | Zoning district code (e.g., RS-60, CBD-1, RM-24) |
| ZONEDESC | str | Zoning district description |
| LASTUPDATE | str | Date of last zoning update |
| HEIGHT | float | Maximum permitted building height |
| BASEELEV | float | Base elevation |
| ShapeSTAre | float | Original shape area (sq ft) |
| ShapeSTAre_corrected | float | Corrected shape area after split-parcel adjustment |
| ShapeSTLen | float | Shape perimeter length |
| geometry | str | WKT geometry |

### acs_blockgroup_tampa.csv (881 rows)

ACS 5-Year (2015--2019) sociodemographic variables at the block-group level.

| Column | Type | Description |
|--------|------|-------------|
| GEOID | str | Block group GEOID |
| median_hh_income | float | Median household income ($) |
| total_pop | int | Total population |
| total_hh | int | Total households |
| hh_no_vehicle | int | Households with no vehicle available |
| educ_total | int | Population 25+ (education universe) |
| educ_bachelors_plus | int | Population 25+ with bachelor's degree or higher |
| total_race | int | Total population (race universe) |
| white_nonhisp | int | White non-Hispanic population |
| black_nonhisp | int | Black non-Hispanic population |
| hispanic | int | Hispanic/Latino population |
| poverty_total | int | Population for poverty status determination |
| poverty_below | int | Population below poverty level |
| housing_total | int | Total housing units |
| owner_occupied | int | Owner-occupied housing units |
| pct_no_vehicle | float | % households with no vehicle |
| pct_bachelors_plus | float | % population 25+ with bachelor's+ |
| pct_white_nonhisp | float | % white non-Hispanic |
| pct_black_nonhisp | float | % Black non-Hispanic |
| pct_hispanic | float | % Hispanic/Latino |
| pct_poverty | float | % below poverty level |
| pct_owner_occ | float | % owner-occupied housing |

### tampatrips_1.csv (5,749 rows)

Individual trip records from the 2019 TBRTS Household Travel Survey, filtered
to trips with Tampa block-group origins or destinations.

| Column | Type | Description |
|--------|------|-------------|
| personid | str | Unique person identifier |
| age | int | Respondent age |
| gender | str | Respondent gender |
| hh_id | str | Household identifier |
| education | str | Education level |
| income_detailed | str | Household income bracket |
| num_veh | int | Number of vehicles in household |
| hhsize | int | Household size |
| d_bg | str | Destination block group GEOID |
| o_bg | str | Origin block group GEOID |
| trip_duration | float | Trip duration (minutes) |
| trip_distance | float | Trip distance (miles) |
| mode_type | str | Travel mode (walk, bike, auto, transit, etc.) |
| d_purpose_category_imputed | str | Imputed trip destination purpose |
| origin_geom | str | Origin point geometry |
| dest_geom | str | Destination point geometry |

### zone_text_corpora_deduped.csv (56 rows)

Deduplicated zoning text corpus, one row per zoning district.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| context | str | Zone context category (Residential, Non-Residential, Other/PD) |
| corpus_chars | int | Character count of original corpus |
| corpus | str | Full original zoning text |
| corpus_deduped | str | Deduplicated zoning text |
| orig_len | int | Original text length |
| dedup_len | int | Deduplicated text length |
| reduction_pct | float | % reduction from deduplication |

### zone_text_corpora_new.csv (56 rows)

Intermediate corpus before deduplication. Same schema as above minus
deduplication columns.

### zone_embeddings.csv (56 rows) -- `processed_data/zone_embeddings/`

Default (MiniLM) zone-level text embeddings, 32 PCA dimensions.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| context | str | Zone context category |
| emb_000 -- emb_031 | float | PCA-reduced embedding dimensions |

### zone_embeddings_{minilm,mpnet,multi-qa,openai-lg,google,hybrid}.csv (56 rows each) -- `processed_data/zone_embeddings/`

Same schema as `zone_embeddings.csv`, one file per embedding backend.
Hybrid has 42 dimensions (10 LLM scores + 32 MiniLM PCA).

### zone_embeddings_tfidf.csv (56 rows) -- `processed_data/zone_embeddings/`

TF-IDF baseline embeddings (32 PCA dimensions). Same schema.

### zone_embeddings_theory_expanded.csv (56 rows) -- `processed_data/zone_embeddings/`

Combined theory-driven features (LLM scores + dimensional standards +
embedding-derived features). Same zone_class/context prefix, variable
number of feature columns.

### zone_embeddings_{backend}_tfidf_interpretation.csv -- `processed_data/zone_embeddings/`

TF-IDF term correlations for each embedding dimension, per backend.

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Embedding dimension (emb_000, etc.) |
| label | str | Human-readable dimension label |
| top_pos_terms | str | Top positively correlated TF-IDF terms |
| top_pos_corrs | str | Correlation values for top positive terms |
| top_neg_terms | str | Top negatively correlated TF-IDF terms |

### zone_embeddings_llm_scored.csv (56 rows) -- `processed_data/zone_embeddings/`

Embeddings with appended LLM feature scores. Zone_class + context + 10
LLM feature dimensions (feat_000--feat_009).

### zone_ped_scores.csv (56 rows)

Pedestrian-friendliness scores derived from zoning code provisions.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| sidewalk_req | int | Sidewalk requirement (0/1) |
| frontage_std | int | Frontage standard (0/1) |
| active_ground_flr | int | Active ground floor requirement (0/1) |
| mixed_use_perm | int | Mixed-use permitted (0/1) |
| ped_scale_lang | int | Pedestrian-scale language present (0/1) |
| bicycle_prov | int | Bicycle provisions (0/1) |
| transit_orient | int | Transit-oriented provisions (0/1) |
| reduced_parking | int | Reduced parking provisions (0/1) |
| landscaping_req | int | Landscaping requirement (0/1) |
| human_scale_design | int | Human-scale design standards (0/1) |
| ped_score_total | int | Sum of all binary indicators (0--11) |
| ped_score_norm | float | Normalized pedestrian score (0--1) |

### zone_at_rates.csv (26 rows)

Active travel rates by zoning district, aggregated from trip data.

| Column | Type | Description |
|--------|------|-------------|
| ZONECLASS | str | Zoning district code |
| trips | int | Total trips in zone |
| at_rate | float | Active travel rate (% walk/bike trips) |
| bgs | int | Number of block groups in zone |

### zone_dimensional_standards.csv (56 rows)

Quantitative zoning dimensional standards extracted from code.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| min_lot_area | float | Minimum lot area (sq ft) |
| lot_width | float | Minimum lot width (ft) |
| front_setback | float | Front setback (ft) |
| side_setback | float | Side setback (ft) |
| rear_setback | float | Rear setback (ft) |
| max_height | float | Maximum building height (ft) |

### zone_features_combined.csv (56 rows)

Combined feature table (ped scores + dimensional standards).
Columns are the union of `zone_ped_scores.csv` and
`zone_dimensional_standards.csv`.

### zone_llm_features_raw.csv (56 rows)

GPT-4o scored planning dimensions with rationales.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| context | str | Zone context category |
| use_mixing | int | Use mixing score (1--5) |
| use_mixing_rationale | str | LLM rationale for score |
| density_permissions | int | Density permissions score (1--5) |
| ... | ... | (10 dimensions, each with score + rationale) |

Dimensions: use_mixing, density_permissions, ped_street_interface,
parking_intensity, form_based_design, use_flexibility, transit_orientation,
open_space_green, setback_building_placement, transparency_activation.

### zone_llm_dimension_map.csv

Maps column names to dimension metadata.

| Column | Type | Description |
|--------|------|-------------|
| col | str | Column name in feature files |
| dimension | str | Dimension identifier |
| display_name | str | Human-readable dimension name |
| rubric | str | Scoring rubric description |

### zone_theory_expanded_raw.csv (56 rows)

Expanded theory-driven features combining LLM scores with dimensional
standards and pedestrian scores into a unified feature set.

### zone_theory_expanded_map.csv

Metadata mapping for the expanded theory features.

| Column | Type | Description |
|--------|------|-------------|
| col | str | Feature column name |
| feature | str | Feature identifier |
| dimension_id | str | Dimension identifier |
| dimension_label | str | Human-readable dimension label |
| source | str | Data source (llm, code, composite) |
| description | str | Feature description |
| inverted | bool | Whether higher values mean less pedestrian-friendly |
| n_unique | int | Number of unique values |
| pct_missing | float | % missing values |

### zone_llm_features_prompt.txt

The system prompt and rubric sent to GPT-4o for dimension scoring.

### parcel_embedding_acs_table.csv (6,238 rows)

Parcel-level analysis table joining zoning embeddings and ACS data.
Contains OBJECTID, GEOID, ZONECLASS, zone_context, area_share,
ped_score_norm, dimensional standards, and all zone embedding dimensions
(zone_emb_000--zone_emb_031).

### tpa_zon_geoid.csv

Zoning-to-block-group crosswalk (same schema as tpazoning_clean.csv,
without the corrected area column).

### split_parcel_report.csv

Diagnostic report for parcels that span multiple block groups.

| Column | Type | Description |
|--------|------|-------------|
| OBJECTID | int | Parcel object ID |
| ZONECLASS | str | Zoning district |
| n_geoids | int | Number of block groups the parcel intersects |
| geoids | str | List of intersected GEOIDs |
| ShapeSTAre | float | Total parcel area |
| total_intersect_area | float | Sum of intersection areas |

### counterfactual_results.csv (112 rows)

Full counterfactual generation results (56 zones x 2 variants).

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| variant | str | Counterfactual type (standard, aggressive) |
| original_score | float | DML causal score from original text |
| counterfactual_score | float | DML causal score from rewritten text |
| delta_score | float | Change in causal score |
| pct_change | float | % change in causal score |
| cosine_similarity | float | Cosine similarity between original and counterfactual embeddings |
| original_chars | int | Original text length |
| counterfactual_chars | int | Counterfactual text length |
| generation_time | float | LLM generation time (seconds) |
| text_file | str | Path to generated text file |

### counterfactual_scores.csv (112 rows)

Compact counterfactual score summary.

| Column | Type | Description |
|--------|------|-------------|
| zone | str | Zoning district code |
| variant | str | standard or aggressive |
| original_score | float | Original DML causal score |
| counterfactual_score | float | Counterfactual DML causal score |
| delta_s | float | Score delta |
| cosine_similarity | float | Embedding cosine similarity |

### counterfactual_texts/ (112 files)

LLM-generated counterfactual zoning text. One `.txt` file per
zone-variant combination (e.g., `RS-60_standard.txt`,
`CBD-1_aggressive.txt`).

### embedding_dimensions_comprehensive.csv

Ranked embedding dimensions with importance scores, value ranges, and
top correlated TF-IDF terms across all backends.

### embedding_interpretations_full32.csv

Full 32-dimension interpretation table with TF-IDF term correlations.

### embedding_classifier_results.csv

Classifier performance comparison (accuracy and log-loss) for baseline
vs. embedding-augmented models.

| Column | Type | Description |
|--------|------|-------------|
| Classifier | str | Classifier name |
| Acc_Base | float | Baseline accuracy |
| LL_Base | float | Baseline log-loss |
| Acc_Emb | float | Embedding-augmented accuracy |
| LL_Emb | float | Embedding-augmented log-loss |

---

## tables/

### dml_backend_comparison.csv (244 rows)

DML theta estimates for each embedding dimension, across all backends.

| Column | Type | Description |
|--------|------|-------------|
| backend | str | Embedding backend (minilm, mpnet, google, etc.) |
| dimension | str | Embedding dimension (emb_000, etc.) |
| theta | float | DML point estimate |
| se | float | Combined standard error |
| se_within | float | Within-fold standard error |
| se_across | float | Across-fold standard error |
| pval | float | Raw p-value |
| p_bh | float | Benjamini-Hochberg adjusted p-value |
| bh_sig | bool | Significant after BH correction |
| n_fits | int | Number of cross-fitting folds |

### dml_backend_summary.csv (8 rows)

Summary statistics per backend: number of significant dimensions, top/bottom
zones, and effect magnitudes.

| Column | Type | Description |
|--------|------|-------------|
| backend | str | Embedding backend |
| label | str | Human-readable backend name |
| n_dims | int | Total embedding dimensions |
| n_significant | int | BH-significant dimensions |
| pct_significant | float | % significant |
| top_3_at | str | Top 3 zones by AT causal score |
| bottom_3_car | str | Bottom 3 zones (most car-oriented) |
| max_theta | float | Largest positive theta |
| min_theta | float | Most negative theta |
| mean_abs_theta | float | Mean absolute theta |

### dml_estimand_c_comparison.csv (448 rows)

Zone-level causal scores (Estimand C) across all backends.

| Column | Type | Description |
|--------|------|-------------|
| backend | str | Embedding backend |
| zone_class | str | Zoning district code |
| context | str | Zone context category |
| causal_score | float | Composite causal score (dot product of zone embedding and theta) |

### dml_results.csv

Per-dimension DML results from the primary specification.

| Column | Type | Description |
|--------|------|-------------|
| dim | str | Embedding dimension |
| coef | float | DML coefficient |
| se | float | Standard error |
| t_stat | float | t-statistic |
| p_raw | float | Raw p-value |
| ci_lo, ci_hi | float | 95% confidence interval |
| p_bh | float | BH-adjusted p-value |
| sig_bh, sig_raw | bool | Significance flags |

### dml_scalar_results.csv

Scalar (aggregate) DML results with multiple treatment/nuisance specifications.

| Column | Type | Description |
|--------|------|-------------|
| treatment | str | Treatment variable |
| col | str | Column/dimension |
| nuisance | str | Nuisance model (RF, Lasso, etc.) |
| sample | str | Sample specification |
| n | int | Sample size |
| theta | float | DML estimate |
| se_hc1 | float | HC1 robust standard error |
| se_boot | float | Bootstrap standard error |
| t_stat | float | t-statistic |
| p_value | float | p-value |
| ci_lo, ci_hi | float | 95% confidence interval |

### dml_zone_causal_scores.csv (56 rows)

Zone-level causal scores from the primary DML specification.

| Column | Type | Description |
|--------|------|-------------|
| zone_class | str | Zoning district code |
| context | str | Zone context |
| causal_score | float | Composite causal score |

### theory_dml_{acs,joint,travel,veh}_results.csv

Theory-driven DML results for each control specification. All share the
same schema:

| Column | Type | Description |
|--------|------|-------------|
| feature | str | Feature column name |
| dimension | str | Planning dimension |
| feature_label | str | Human-readable feature label |
| spec | str | Model specification |
| theta | float | DML estimate |
| se_hc1 | float | HC1 standard error |
| se_boot | float | Bootstrap standard error |
| t_stat | float | t-statistic |
| p_raw | float | Raw p-value |
| ci_lo, ci_hi | float | 95% confidence interval |
| n_boot_valid | int | Valid bootstrap iterations |
| p_bh | float | BH-adjusted p-value |
| sig_bh, sig_raw | bool | Significance flags |

### theory_dml_{acs,joint,travel,veh}_pivot.csv

Pivoted comparison across RF and GB nuisance models, with and without ACS.

| Column | Type | Description |
|--------|------|-------------|
| dimension | str | Planning dimension |
| feature | str | Feature name |
| RF_Full | float | RF estimate, full controls |
| RF_NoACS | float | RF estimate, no ACS |
| GB_Full | float | Gradient boosting, full controls |
| GB_NoACS | float | Gradient boosting, no ACS |

### theory_dml_{acs,joint,travel,veh}_partial.csv

Partial effect results (subset of significant features).
Same schema as the `_results.csv` files.

### baseline_zoning_effects.csv (37 rows)

OLS/RF baseline estimates of zoning effects on active travel.

| Column | Type | Description |
|--------|------|-------------|
| Variable | str | Zoning variable or control |
| Display | str | Display label |
| Category | str | Variable category |
| LR_Estimate | float | Linear regression estimate |
| LR_SE | float | LR standard error |
| LR_p | float | LR p-value |
| LR_sig | str | LR significance flag |
| RF_Estimate | float | Random forest estimate |
| RF_SE | float | RF standard error |
| RF_p | float | RF p-value |
| RF_sig | str | RF significance flag |

### tampatrips_summary_stats.csv

Descriptive statistics for the trip dataset.

| Column | Type | Description |
|--------|------|-------------|
| (index) | str | Variable name |
| n | int | Count |
| mean | float | Mean |
| std | float | Standard deviation |
| min | float | Minimum |
| median | float | Median |
| max | float | Maximum |
