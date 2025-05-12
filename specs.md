# Project Specifications: AI Agent Survey Analysis

Below is an **end‑to‑end, production‑ready project plan**—from raw CSV to an arXiv‑ready short‑communication PDF—fully updated for nominal survey items and free-text analysis.

> **Rule:** All code should be implemented as Python scripts (.py).

---

# 0 Project Synopsis

| Item                    | Decision                                                                                                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Working title**       | *"Will Agents Replace Us? Perceptions of Autonomous Multi‑Agent AI"*                                                                                                                                  |
| **Manuscript type**     | Short communication (≤ 4 000 words, 4 main figures, 1 table, ≤ 10 pages)                                                                                                                           |
| **Target venue**        | arXiv → cs.HC / cs.AI (cross‑list stat.AP)                                                                                                                                                         |
| **Principal questions** | • Expected timeline for agent dominance in coding (Q1)<br>• First systems perceived as disruptable (Q2)<br>• Barriers to deployment (Q3)<br>• Responsibility & governance beliefs (Q4, Q9)<br>• Key future capabilities & limitations (Q5, Q8)<br>• Optimal agent team size (Q6)<br>• Sacrifices for productivity (Q7)<br>• Future human roles (Q10)<br>• Predictors of *actual* agent deployment (Q3) |
| **Data**                | 126 survey responses: 10 nominal questions (Q1-Q10) + 1 free-text (Q11) + metadata (timestamp, region). Raw data in `data/raw/survey_responses_rows_20250512.csv`. Questions in `data/raw/questions.json`. |

---

# 1 Ethics & Reproducibility

| Check         | Action                                                                                                                               | Status (as per codebase) |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| **Privacy**   | Drop `id` column, jitter `created_at` by ±5 min.                                                                                     | Implemented (`01_prepare.py`) |
| **Consent**   | Survey voluntary, no PII collected (as stated) ⇒ IRB exemption note in script.                                                        | Implemented (`01_prepare.py`) |
| **Licensing** | Code: MIT. Data: CC‑BY‑NC 4.0.                                                                                                       | Implemented (LICENSE, data/LICENSE.txt, README.md) |
| **Seeds**     | `random_state=42` or `np.random.seed(42)` used in `01_prepare.py`, `03_infer.py` (MCA, KModes).                                      | Implemented |
| **Repro**     | Quarto project (`manuscript/paper.qmd`) + Dockerfile with `conda env create -f environment.yml`. Builds reproducible environment.      | Implemented |

---

# 2 Environment

Uses `environment.yml` for Conda and `Dockerfile` for containerization.
Key dependencies: `python=3.11`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `kmodes`, `prince`, `statsmodels`, `quarto`.

---

# 3 Data Pipeline

## 3.1 Data Wrangling (`scripts/01_prepare.py`)

*   **Input**: `data/raw/survey_responses_rows_20250512.csv`, `data/raw/questions.json` (for context, not directly used by script).
*   **Processing**:
    1.  Loads raw survey data.
    2.  Drops `id` column.
    3.  Jitters `created_at` timestamps by ±5 minutes for de-identification.
    4.  Expands JSON `answers` blob into individual columns (Q1-Q11).
    5.  Extracts coarse `region` (e.g., "Europe", "America") from `metadata.timeZone`.
    6.  Merges `created_at`, `region` with answer columns.
    7.  **Data Cleaning**:
        *   Q3: Consolidates 'Regulatory/compliance concerns' into 'Regulatory/compliance or technical readiness concerns'.
        *   Q5: Maps 'Emergent consciousness' (a rare response) to 'No opinion'.
    8.  Removes duplicate rows.
    9.  Strips leading/trailing whitespace from answers for Q1-Q10 and converts them to `pd.Categorical`.
*   **Outputs**:
    *   `data/clean_survey.parquet`: Tidy data for quantitative analysis.
    *   `data/q11.txt`: Extracted, cleaned free-text responses from Q11.

---

# 4 Exploratory Analysis & Visualization (`scripts/02_explore_v2.py`)

*   **Input**: `data/clean_survey.parquet`.
*   **Visualizations (saved to `figs/`)**:
    1.  **Response Distributions (Fig 1a-j, Fig 1k grid, Fig 1l heatmap)**:
        *   Ten 100% horizontal bar charts (one for each Q1-Q10), showing proportion of each answer. 'No opinion' handled separately and typically at the bottom. (e.g., `Q1_barh.png`)
        *   A grid plot combining all ten bar charts (`all_questions_grid_improved.png`).
        *   A heatmap showing proportions of all answers across all questions (`all_questions_heatmap.png`).
    2.  **Associations (Fig 2a mosaic, Fig 2b Cramér's V)**:
        *   Mosaic plot for Q1 × Q3 (Timeline belief vs. Deployment status) with abbreviated labels (`Q1xQ3_mosaic_supplementary_s1.png`).
        *   Heatmap of Cramér's V for all Q1-Q10 pairs to show strength of association (`cramers_v_heatmap_figure2.png`).
*   **Outputs**:
    *   Image files in `figs/`.
    *   `results/all_questions_results.json`: JSON file containing the normalized counts (proportions) for each answer of Q1-Q10.
*   **Style**: Uses `seaborn` palettes (`tab10`), `Roboto` font if available.

---

# 5 Inferential Statistics & Modeling (`scripts/03_infer_v2.py`)

*   **Input**: `data/clean_survey.parquet`.
*   **Analyses**:

    1.  **Pairwise Associations (Chi-squared & Cramér's V)**:
        *   Performs Pearson Chi-squared tests for independence on all 45 pairs of questions (Q1-Q10).
        *   Calculates Cramér's V (with bias correction) for effect size.
        *   Effect size categories for Cramér's V: negligible (<0.10), small (0.10-0.29), medium (0.30-0.49), large (≥0.50).
        *   Benjamini-Hochberg FDR correction is applied to p-values due to multiple comparisons (45 tests).
        *   *Outputs*:
            *   `results/pairwise_tests.csv`: Detailed results for each pair (chi2, p-value, dof, Cramér's V, effect size, significance).
            *   `results/pairwise_matrices.json`: P-values, Cramér's V, and effect sizes in matrix format.
            *   `results/pairwise_summary.json`: Overall summary (total tests, significant count, effect size distribution, strongest pairs).

    2.  **Multiple Correspondence Analysis (MCA)**:
        *   Applies MCA to Q1-Q10 to identify latent dimensions of attitudes.
        *   Uses `prince.MCA` with `n_components=10` initially.
        *   Determines number of components to retain based on explaining at least 20% of cumulative inertia (`n_components_20`).
        *   *Outputs*:
            *   `results/mca_inertia.json`: Explained and cumulative inertia per component, and `n_components_20`.
            *   `results/mca_row_coordinates.csv`: Principal coordinates for each respondent on the retained components.

    3.  **Segmentation (K-Modes Clustering)**:
        *   Applies K-Modes clustering to Q1-Q10 responses to identify distinct respondent segments.
        *   Tests `k` from 2 to 5 clusters. `init='Huang'`, `n_init=10`.
        *   Uses elbow method (plotting inertia/cost vs. `k`) to help select the optimal `k`. The script auto-selects `best_k=3` for the paper, as justified in the manuscript.
        *   *Outputs*:
            *   `results/kmodes_results.json`: Cluster labels, inertia, and mode centroids for each tested `k`.
            *   `results/kmodes_elbow.png`: Plot of inertia vs. number of clusters.
            *   `results/kmodes_table1.csv` & `results/kmodes_table1.json` (Table 1 for paper): Shows cluster size and the modal (most frequent) answer for each question within each cluster for the chosen `best_k`.

    4.  **Predictors of Agent Deployment (Logistic Regression)**:
        *   **Outcome**: Binary variable `deploy_now` (1 if Q3 is "We ARE deploying them now", 0 otherwise).
        *   **Predictors**: Categorical responses to Q1, Q2, Q4-Q10 (Q3 is excluded as it forms the outcome). These are one-hot encoded.
        *   **Model**: Fixed-effects Logistic Regression (`statsmodels.api.Logit`).
        *   *Outputs*:
            *   `results/logit_deployment_summary.txt`: Full summary of the logistic regression model.
            *   `results/logit_deployment_oddsratios.csv`: Odds ratios, p-values, and confidence intervals for each predictor.
            *   `results/logit_forest_plot_figure4.png` (Fig 4 for paper): Forest plot visualizing the odds ratios and their CIs.

---

# 6 Qualitative Analysis of Q11 (Free-text Remarks)

*   **Input**: `data/q11.txt` (11 non-empty responses).
*   **Analysis**: Manual thematic summary only, due to small n. No further scripting or computational analysis planned. Key themes and illustrative quotes are summarized in the manuscript Discussion. No supplementary table or automated keyphrase extraction is included.

---

# 7 Manuscript & Visualisation Style

*   **Manuscript**: `manuscript/paper.qmd` (Quarto, PDF output).
*   **Figures**:
    *   Fig 1: Response distributions (from `02_explore_v2.py`, grid: `all_questions_grid_improved.png`).
    *   Fig 2: Associations (Cramér's V heatmap from `02_explore_v2.py`, `cramers_v_heatmap_figure2.png`).
    *   Fig 3: Latent attitudes & clusters (MCA plot - to be generated if not present, and reference to Table 1).
    *   Fig 4: Predictors of deployment (Forest plot of odds ratios from `03_infer_v2.py`, `logit_forest_plot_figure4.png`).
    *   Table 1: K-modes cluster profiles (from `03_infer_v2.py`).
*   **Colors**: Use `matplotlib` default categorical palettes (e.g., `tab10`). Ensure color-blind safety (avoid pure red/green juxtapositions).
*   **Fonts**: `Roboto` if available, otherwise system default sans-serif.
*   **Figure Size**: Default `(6,4)` inches, larger for heatmaps/grids (`(8,6)` or `(6,6)` for MCA biplot).

---

# 8 Repository Structure
preprint2/
├── Dockerfile
├── LICENSE
├── README.md
├── data/
│ ├── LICENSE.txt
│ ├── clean_survey.parquet
│ ├── q11.txt
│ └── raw/
│ ├── questions.json
│ └── survey_responses_rows_20250512.csv
├── environment.yml
├── figs/ (auto-generated by scripts)
├── manuscript/
│ └── paper.qmd
├── results/ (auto-generated by scripts)
├── scripts/
│ ├── 01_prepare.py
│ ├── 02_explore_v2.py
│ └── 03_infer_v2.py
├── specs.md
└── todo.md

---

# 9 Common Pitfalls & Mitigations

| Risk                              | Mitigation                                                                                                                            | Status / Plan |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Small free‑text n (Q11) ⇒ over‑claiming | Scope Q11 to illustrative, anecdotal insights. Clearly state `n=11` (or fewer after cleaning).                                        | Guideline for paper writing |
| Over‑fitting clusters / arbitrary `k` | Report inertia plot (`kmodes_elbow.png`). Justify choice of `k` (e.g., via elbow inspection, silhouette scores if applicable to k-modes). Provide code to reproduce with different `k`. | Elbow plot generated. Justification in paper needed. |
| χ² invalid if expected counts < 5 | For pairs with low expected counts, note this limitation. Fisher's Exact Test could be an alternative for 2x2 tables or small N.        | Check expected counts in `03_infer_v2.py` output (not explicitly done now). |
| Multiple testing inflation (pairwise tests) | Apply Benjamini-Hochberg FDR correction to the 45 p-values from pairwise tests. Report both raw and adjusted p-values if space allows. | **To be implemented in `03_infer_v2.py`**. |
| Interpretation of MCA dimensions  | Carefully examine variable contributions to MCA dimensions. Biplots (variables and/or individuals) can aid interpretation.            | MCA coordinates saved. Biplot generation for paper needed. |
| Logistic Regression Assumptions   | Check for multicollinearity among predictors (e.g., VIFs). Ensure sufficient data points per predictor category.                      | **To be implemented/checked in `03_infer_v2.py`**. |

---