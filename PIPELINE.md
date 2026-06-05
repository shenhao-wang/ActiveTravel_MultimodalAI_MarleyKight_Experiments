# Pipeline Execution Order

All scripts run from the project root. Data is read from `raw_data/` and
`processed_data/`; analysis outputs go to `tables/` and `figures/`.

---

## 01 -- Zoning Scraping & Data Sourcing

Scrapes Tampa's municipal zoning code, parses parcel-level zoning assignments,
and pulls Census sociodemographic data.

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `zoning_scraperv3.ipynb` | Scrapes Tampa zoning code and builds text corpora | Tampa Municipal Code (web) | `zone_text_corpora.csv`, `zone_text_corpora_deduped.csv` |
| `zoning_residualization.ipynb` | Cleans raw zoning parcel data | `raw_data/` shapefiles, `tpazoning.csv` | `tpazoning_clean.csv` |
| `tampa_zon_splitparcel.ipynb` | Maps parcels that span multiple block groups | `tpazoning_clean.csv`, shapefiles | `split_parcel_report.csv`, `tpa_zon_geoid.csv` |
| `censusdata.ipynb` | Pulls ACS 5-year block-group data from Census API | Census API (Florida FIPS 12, Hillsborough 057) | `acs_blockgroup_tampa.csv` |

**Run order:** `zoning_scraperv3` -> `zoning_residualization` -> `tampa_zon_splitparcel` -> `censusdata`

---

## 02 -- Text Corpora

Corpora preparation is handled within Stage 01 (scraping + deduplication).
The final corpus used downstream is `processed_data/zone_text_corpora_deduped.csv`
(56 zones, one row per zone with full deduplicated text).

---

## 03 -- Embeddings

Generates zone-level text embeddings using multiple models, then interprets
embedding dimensions via TF-IDF correlation.

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `zone_embeddingsv3.ipynb` | Multi-backend embeddings (6 models + TF-IDF) | `processed_data/zone_text_corpora_deduped.csv` | `processed_data/zone_embeddings/zone_embeddings_{minilm,mpnet,...}.csv`, `processed_data/zone_embeddings/zone_embeddings.csv` (best backend), `processed_data/zone_embeddings/zone_embeddings_tfidf.csv` |
| `zone_embeddings_interpretation.ipynb` | TF-IDF term correlation for each embedding dimension | `processed_data/zone_embeddings/zone_embeddings_*.csv`, `processed_data/zone_text_corpora_deduped.csv` | `processed_data/zone_embeddings/*_tfidf_interpretation.csv`, `processed_data/embedding_dimensions_comprehensive.csv`, `processed_data/embedding_interpretations_full32.csv` |

**Embedding models:**

| Backend key | Model | Dimensions |
|-------------|-------|------------|
| `minilm` | all-MiniLM-L6-v2 | 32 (PCA) |
| `mpnet` | all-mpnet-base-v2 | 32 (PCA) |
| `multi-qa` | multi-qa-MiniLM-L6-cos-v1 | 32 (PCA) |
| `openai-lg` | text-embedding-3-large | 32 (PCA) |
| `google` | text-embedding-004 | 32 (PCA) |
| `hybrid` | LLM scores + MiniLM concat | 42 |
| `tfidf` | TF-IDF baseline | 32 (PCA) |

**Run order:** `zone_embeddingsv3` -> `zone_embeddings_interpretation`

---

## 04 -- LLM Feature Scoring

Uses GPT-4o to score each zoning district on 10 theory-driven planning
dimensions (1--5 scale with rationales).

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `llm_zoning_features.py` | Scores zones via OpenAI API | `zone_text_corpora_deduped.csv` | `zone_llm_features_raw.csv`, `zone_llm_dimension_map.csv` |
| `run_llm_features.sh` | SLURM launcher | -- | -- |

**Dimensions scored:** use_mixing, density_permissions, ped_street_interface,
parking_intensity, form_based_design, use_flexibility, transit_orientation,
open_space_green, setback_building_placement, transparency_activation

**Run order:** `run_llm_features.sh` (calls `llm_zoning_features.py`)

---

## 05 -- Double Machine Learning

Causal inference pipeline: estimates the effect of zoning text on active travel
rates, controlling for ACS sociodemographics.

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `dml_multi_backend.py` | Multi-backend DML (Estimands B & C) | Embeddings, `zone_at_rates.csv`, `acs_blockgroup_tampa.csv` | `tables/dml_backend_comparison.csv`, `tables/dml_backend_summary.csv`, `tables/dml_estimand_c_comparison.csv` |
| `doubleml_v3.ipynb` | Interactive DML notebook | Same | `tables/dml_results.csv`, `tables/dml_scalar_results.csv`, `tables/dml_zone_causal_scores.csv` |
| `theory_driven_expanded.py` | Theory-driven DML using LLM + dimensional features | LLM scores, embeddings, ACS | `zone_theory_expanded_raw.csv`, `zone_theory_expanded_map.csv`, `processed_data/zone_embeddings/zone_embeddings_theory_expanded.csv` |
| `theory_dml_acs.py` | Theory-driven DML, ACS controls | Theory features, ACS | `tables/theory_dml_acs_*.csv` |
| `theory_dml_joint.py` | Joint specification (all controls) | Theory features, ACS, travel, vehicle | `tables/theory_dml_joint_*.csv` |
| `theory_dml_travel.py` | Travel behavior controls | Theory features, travel data | `tables/theory_dml_travel_*.csv` |
| `theory_dml_veh.py` | Vehicle ownership controls | Theory features, vehicle data | `tables/theory_dml_veh_*.csv` |
| `baseline_effects.py` | OLS baseline for comparison | ACS, zoning dummies | `tables/baseline_zoning_effects.csv` |
| `baseline_zoning_effects_expanded.py` | Expanded baseline (LR + RF) | ACS, zoning features | `tables/baseline_zoning_effects.csv` |
| `run_dml_*.sh` | SLURM launchers for each specification | -- | -- |

**Run order:** `dml_multi_backend.py` -> `theory_driven_expanded.py` ->
`theory_dml_{acs,joint,travel,veh}.py` (can run in parallel) ->
`baseline_*.py`

---

## 06 -- Counterfactuals

Generates counterfactual zoning text using a local LLM, re-embeds it, and
projects through the DML model to estimate predicted changes in active travel.

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `counterfactual_zoning.py` | Generate counterfactual text, embed, and score | `processed_data/zone_text_corpora_deduped.csv`, `tables/dml_backend_comparison.csv` | `processed_data/counterfactual_results.csv`, `processed_data/counterfactual_scores.csv`, `processed_data/counterfactual_texts/` |
| `run_counterfactual.sh` | SLURM launcher (GPU, Mistral-7B default) | -- | -- |

Each zone gets two counterfactual variants (`standard` and `aggressive`
pedestrian-friendly rewrites). Output text files are in
`processed_data/counterfactual_texts/`.

**Run order:** `run_counterfactual.sh` (calls `counterfactual_zoning.py`).

---

## 07 -- Figures & Tables

Generates all publication figures and summary statistics.

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `paper_figures.py` | Main publication figure set | `figures/map*.png`, spatial context figures |
| `make_figs.py` | Additional figure generation | Various `figures/*.png` |
| `fig_wordclouds.py` | Embedding dimension word clouds | `figures/fig_wordcloud_{minilm,openai}.{png,pdf}` |
| `visuals.ipynb` / `visualsv2.py` | Maps and exploratory visualizations | `figures/map*.png`, `figures/viz*.png` |
| `summary_stats.ipynb` | Trip and zoning summary statistics | `tables/tampatrips_summary_stats.csv` |
| `table.ipynb` | Formatted tables for paper | Various `tables/*.csv` |
| `tpacityshp.ipynb` | City boundary and spatial base maps | `figures/map*.png` |

**Run order:** Can run in any order after Stages 01--06 are complete.
