# Project TODO: AI Agent Survey Analysis

> **Rule:** All code should be implemented as Python scripts (.py).

## 0. Project Synopsis
- [x] Define working title, manuscript type, target venue, principal questions, and data description.

## 1. Ethics & Reproducibility
- [x] Implement privacy measures: Drop `id`, jitter `created_at` (`scripts/01_prepare.py`).
- [x] Add IRB exemption note in script (`scripts/01_prepare.py`).
- [x] Set code (MIT) and data (CC-BY-NC 4.0) licenses.
- [x] Ensure `random_state=42` or `np.random.seed(42)` is used for all stochastic processes (`scripts/01_prepare.py`, `scripts/03_infer.py`).
- [x] Ensure reproducibility via Quarto (`manuscript/paper.qmd`) and Docker (`Dockerfile`, `environment.yml`).

## 2. Environment
- [x] Finalize `environment.yml` with all necessary dependencies.
- [x] Test `Dockerfile` for building the environment.
- [x] Update `README.md` with setup and run instructions for Conda and Docker.

## 3. Data Pipeline
- [x] Implement `scripts/01_prepare.py` for:
  - [x] Loading raw data.
  - [x] Dropping `id`, jittering `created_at`.
  - [x] Expanding JSON `answers` into columns.
  - [x] Extracting `region` from metadata.
  - [x] Data cleaning for Q3 and Q5.
  - [x] Removing duplicates and stripping whitespace.
  - [x] Converting Q1-Q10 to categorical.
  - [x] Saving `data/clean_survey.parquet`.
  - [x] Saving `data/q11.txt`.

## 4. Exploratory Analysis & Visualization
- [x] Implement `scripts/02_explore_v2.py` for:
  - [x] Generating and saving 100% horizontal bar charts for Q1-Q10 (individual, grid, heatmap of proportions) to `manuscript/figs/`.
  - [x] Saving proportion data to `results/all_questions_results.json`.
  - [x] Generating and saving Mosaic plot for Q1 × Q3 to `manuscript/figs/` (as `Q1xQ3_mosaic_supplementary_s1.png`).
  - [x] Generating and saving Cramér's V heatmap for Q1-Q10 to `manuscript/figs/` (as `cramers_v_heatmap_figure2.png`).

## 5. Inferential Statistics & Modeling
- [x] Implement `scripts/03_infer_v2.py` for all inferential analyses:
  - [x] Pairwise chi-squared/Fisher's tests, Cramér's V, FDR correction.
  - [x] MCA and K-Modes clustering (with elbow plot and Table 1 output).
  - [x] Logistic regression with VIF checks, forest plot, and summary outputs.

## 6. Qualitative Analysis (Q11)
- [x] Manual thematic summary only (n=11). No further scripting or automation planned. Key themes and illustrative quotes summarized in manuscript Discussion.
- [x] Add 1-2 anonymized illustrative quotes for key themes in Section 3.5

## 7. Address Reviewer Feedback (2024-06)

- [ ] **Survey Instrument and Distribution Details**
  - [x] Add detailed description of survey instrument development (pre-testing, validation, etc.)
  - [x] Add detailed description of distribution methodology (recruitment, platforms, sampling bias discussion)
- [x] **Clustering Results Consistency**
  - [x] Reconcile cluster descriptions in main text with Supplementary Table S1
  - [x] Check and correct cluster labeling, modal answers, and textual descriptions for consistency
  - [x] Justify k=3 choice in light of interpretability and any changes
- [x] **MCA Inertia**
  - [x] Explicitly discuss low explained inertia in MCA and its implications for clustering validity
- [x] **Logistic Regression Predictors**
  - [x] Clarify in Methods why only certain Q3 barriers were included as predictors
  - [x] Consider testing other Q3 barriers as predictors for deployment status
- [x] **Discussion - Limitations**
  - [x] Add explicit discussion of survey instrument limitations (wording, fixed choice, nuance loss)
