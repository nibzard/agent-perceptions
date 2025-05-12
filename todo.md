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
- [x] Implement `scripts/02_explore.py` for:
  - [x] Generating and saving 100% horizontal bar charts for Q1-Q10 (individual, grid, heatmap of proportions) to `figs/`.
  - [x] Saving proportion data to `results/all_questions_results.json`.
  - [x] Generating and saving Mosaic plot for Q1 × Q3 to `figs/`.
  - [x] Generating and saving Cramér's V heatmap for Q1-Q10 to `figs/`.

## 5. Inferential Statistics & Modeling
- [x] Implement `scripts/03_infer.py` for:
  - [x] **Pairwise Associations**:
    - [x] Pearson Chi-squared tests and Cramér's V for all 45 (Q1-Q10) pairs.
    - [x] Save detailed results (`pairwise_tests.csv`), matrices (`pairwise_matrices.json`), and summary (`pairwise_summary.json`) to `results/`.
  - [x] **Pairwise Associations - Enhancement**:
    - [x] Implement Benjamini-Hochberg FDR correction for p-values in `scripts/03_infer.py`.
    - [x] Add check for low expected cell counts in Chi-squared tests; note limitations or use Fisher's Exact Test if appropriate.
  - [x] **MCA**:
    - [x] Perform MCA on Q1-Q10.
    - [x] Determine number of components to retain (e.g., explaining ≥20% inertia).
    - [x] Save inertia stats (`mca_inertia.json`) and row coordinates (`mca_row_coordinates.csv`) to `results/`.
  - [x] **MCA - Enhancement for Paper**:
    - [x] Generate MCA biplot (variables and/or individuals colored by cluster) for Fig 3.
  - [x] **K-Modes Clustering**:
    - [x] Perform K-Modes clustering for k=2-5.
    - [x] Generate and save elbow plot (`kmodes_elbow.png`) to `results/`.
    - [x] Save detailed K-Modes results (`kmodes_results.json`) to `results/`.
    - [x] Generate and save Table 1 (cluster profiles) for best `k` (`kmodes_table1.csv`, `kmodes_table1.json`) to `results/`.
  - [x] **K-Modes Clustering - Refinement for Paper**:
    - [x] Manually inspect `kmodes_elbow.png` to confirm/select `best_k` for the paper. Document justification.
  - [x] **Logistic Regression**:
    - [x] Implement fixed-effects logistic regression for `deploy_now` (Q3) with Q1,Q2,Q4-Q10 as predictors.
    - [x] Save model summary (`logit_deployment_summary.txt`) and odds ratios (`logit_deployment_oddsratios.csv`) to `results/`.
    - [x] Generate and save forest plot of odds ratios (`logit_deployment_forest.png`) to `results/`.
  - [x] **Logistic Regression - Enhancements**:
    - [x] Check for multicollinearity (e.g., VIFs) among predictors.
    - [x] Consider implementing mixed-effects logistic regression (with `region` as random effect) if deemed important and feasible.

## 6. Manuscript (`manuscript/paper.qmd`)
- [x] Create initial `paper.qmd` with section structure.
- [ ] Write Abstract.
- [ ] Write Introduction section.
- [ ] Write Methods section, detailing survey, participants, data processing, statistical analyses, and qualitative approach.
- [ ] Write Results section, presenting findings from exploratory, pairwise, MCA, clustering, logistic regression, and qualitative analyses, referencing figures and tables.
- [ ] Write Discussion section: interpret results, compare to literature, discuss limitations.
- [ ] Write Conclusion section.
- [ ] Add Acknowledgements, References, Appendix (Survey Items).
- [ ] Create Supplementary Figures & Tables section if needed.
- [ ] Ensure all figures (Fig 1-4) and Table 1 are correctly generated, formatted, and referenced.

## 7. Project Management & Review
- [x] Review and update `specs.md` and `todo.md` (this task).
