import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
from pathlib import Path
import json
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap

# --- Config ---
DATA_PATH = 'data/clean_survey.parquet'
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
QUESTIONS = [str(i) for i in range(1, 11)]
ALPHA = 0.05  # significance level
# Effect size thresholds for Cramér's V (df=1)
EFFECT_SIZES = {
    'small': 0.10,
    'medium': 0.30,
    'large': 0.50
}
PALETTE = sns.color_palette("pastel")  # Define pastel palette

# --- Load data ---
df = pd.read_parquet(DATA_PATH)

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Pairwise Tests ---
results = []
for q1, q2 in combinations(QUESTIONS, 2):
    # Create contingency table
    table = pd.crosstab(df[q1], df[q2])

    # Calculate chi-square and p-value
    chi2, p_value, dof, expected = chi2_contingency(table)
    low_expected = (expected < 5).any()
    test_used = 'chi2'
    fisher_p = None
    if low_expected and table.shape == (2, 2):
        # Use Fisher's Exact Test for 2x2 tables with low expected counts
        try:
            _, fisher_p = fisher_exact(table)
            p_value = fisher_p
            test_used = 'fisher'
        except Exception:
            # fallback to chi2 if Fisher's test fails
            pass

    # Calculate Cramér's V
    cv = cramers_v(df[q1], df[q2])

    # Determine effect size category
    if cv >= EFFECT_SIZES['large']:
        effect = 'large'
    elif cv >= EFFECT_SIZES['medium']:
        effect = 'medium'
    elif cv >= EFFECT_SIZES['small']:
        effect = 'small'
    else:
        effect = 'negligible'

    # Store results
    results.append({
        'q1': f'Q{q1}',
        'q2': f'Q{q2}',
        'chi2': float(chi2),
        'p_value': float(p_value),
        'dof': int(dof),
        'cramers_v': float(cv),
        'effect_size': effect,
        'significant': p_value < ALPHA,
        'low_expected': bool(low_expected),
        'test_used': test_used
    })

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# --- Benjamini-Hochberg FDR correction ---
reject, pvals_corrected, _, _ = multipletests(
    results_df['p_value'], alpha=ALPHA, method='fdr_bh'
)
results_df['p_value_fdr'] = pvals_corrected
results_df['significant_fdr'] = reject

# Save detailed results
results_df.to_csv(RESULTS_DIR / 'pairwise_tests.csv', index=False)

# Create summary matrices for visualization
questions = [f'Q{q}' for q in QUESTIONS]
n_questions = len(questions)

# Initialize matrices
p_values = pd.DataFrame(1.0, index=questions, columns=questions)
p_values_fdr = pd.DataFrame(1.0, index=questions, columns=questions)
cramers = pd.DataFrame(0.0, index=questions, columns=questions)
effects = pd.DataFrame('', index=questions, columns=questions)

# Fill matrices
for result in results_df.to_dict('records'):
    q1, q2 = result['q1'], result['q2']
    p_values.loc[q1, q2] = result['p_value']
    p_values.loc[q2, q1] = result['p_value']
    p_values_fdr.loc[q1, q2] = result['p_value_fdr']
    p_values_fdr.loc[q2, q1] = result['p_value_fdr']
    cramers.loc[q1, q2] = cramers.loc[q2, q1] = result['cramers_v']
    effects.loc[q1, q2] = effects.loc[q2, q1] = result['effect_size']

# Fill diagonals
for q in questions:
    p_values.loc[q, q] = 0.0
    p_values_fdr.loc[q, q] = 0.0
    cramers.loc[q, q] = 1.0
    effects.loc[q, q] = 'perfect'

# Save matrices
matrices = {
    'p_values': p_values.to_dict(),
    'p_values_fdr': p_values_fdr.to_dict(),
    'cramers_v': cramers.to_dict(),
    'effect_sizes': effects.to_dict()
}

with open(RESULTS_DIR / 'pairwise_matrices.json', 'w') as f:
    json.dump(matrices, f, indent=2)

# Print summary
print("\nSummary of Pairwise Tests:")
print(f"Total number of tests: {len(results)}")
print(
    f"Significant relationships (p < {ALPHA}): "
    f"{sum(results_df['significant'])}"
)
print("\nEffect sizes:")
for size in ['negligible', 'small', 'medium', 'large']:
    count = sum(results_df['effect_size'] == size)
    print(f"  {size}: {count}")

# Save summary
summary = {
    'n_tests': len(results_df),
    'n_significant': int(sum(results_df['significant'])),
    'n_significant_fdr': int(sum(results_df['significant_fdr'])),
    'effect_sizes': {
        size: int(sum(results_df['effect_size'] == size))
        for size in ['negligible', 'small', 'medium', 'large']
    },
    'strongest_pairs': results_df.nlargest(3, 'cramers_v')[
        ['q1', 'q2', 'cramers_v', 'effect_size']
    ].to_dict('records')
}

with open(RESULTS_DIR / 'pairwise_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nResults saved to results/")

# --- Multiple Correspondence Analysis (MCA) ---
try:
    import prince
except ImportError:
    raise ImportError("Please install the 'prince' package for MCA: pip install prince")

# Prepare data for MCA (Q1-Q10, as categorical)
mca_df = df[QUESTIONS].astype('category')

mca = prince.MCA(n_components=10, random_state=42)
mca = mca.fit(mca_df)
print('MCA attributes:', dir(mca))

# Explained inertia (variance) per component
explained_inertia = mca.percentage_of_variance_
cum_inertia = mca.cumulative_percentage_of_variance_

# Find number of components to reach at least 20% inertia
# This finds the smallest number of components whose cumulative inertia >= 0.20
n_components_20 = (
    np.argmax(cum_inertia >= 0.20) + 1
)

# Get row principal coordinates for retained components
row_coords = mca.row_coordinates(mca_df).iloc[:, :n_components_20]
row_coords.index = df.index

# Save coordinates and inertia
row_coords.to_csv(RESULTS_DIR / 'mca_row_coordinates.csv')
with open(RESULTS_DIR / 'mca_inertia.json', 'w') as f:
    json.dump({
        'explained_inertia': explained_inertia.tolist(),
        'cum_inertia': cum_inertia.tolist(),
        'n_components_20': int(n_components_20)
    }, f, indent=2)

print(
    "\nMCA complete. "
    f"{n_components_20} components explain at least 20% inertia."
)
print("Row coordinates and inertia saved to results/.")

# --- K-modes clustering (2–5 clusters, elbow method) ---
try:
    from kmodes.kmodes import KModes
except ImportError:
    raise ImportError("Please install the 'kmodes' package: pip install kmodes")

k_range = range(2, 6)
k_modes_results = {}

# Prepare data (Q1-Q10 as strings)
kmodes_df = df[QUESTIONS].astype(str)

for k in k_range:
    km = KModes(n_clusters=k, init='Huang', n_init=5, random_state=42)
    clusters = km.fit_predict(kmodes_df)
    inertia = km.cost_
    k_modes_results[k] = {
        'labels': clusters.tolist(),
        'inertia': inertia,
        'modes': km.cluster_centroids_.tolist()
    }
    # Save cluster assignments for each k
    df[f'cluster_k{k}'] = clusters

# Save all k-modes results
with open(RESULTS_DIR / 'kmodes_results.json', 'w') as f:
    json.dump(k_modes_results, f, indent=2)

# Save cluster assignments for best k (elbow method: lowest k before inertia flattens)
# For the paper, we fix best_k = 3 based on elbow plot (see justification file)
best_k = 3
best_labels = np.array(k_modes_results[best_k]['labels'])

table1 = []
for cluster in range(best_k):
    mask = best_labels == cluster
    cluster_size = mask.sum()
    row = {'cluster': cluster, 'size': int(cluster_size)}
    for q in QUESTIONS:
        # Most common answer in this cluster
        mode = df.loc[mask, q].mode()
        row[f'Q{q}_mode'] = mode.iloc[0] if not mode.empty else None
    table1.append(row)

table1_df = pd.DataFrame(table1)
table1_df.to_csv(RESULTS_DIR / 'kmodes_table1.csv', index=False)
with open(RESULTS_DIR / 'kmodes_table1.json', 'w') as f:
    json.dump(table1, f, indent=2)

print(f"\nTable 1 (cluster size and dominant answers for k={best_k}) saved to results/.")

# --- MCA Biplot with Cluster Overlay (Figure 3) ---
FIGS_DIR = Path('figs')
FIGS_DIR.mkdir(exist_ok=True)

if 'prince' in globals() and 'row_coords' in globals() and 'best_labels' in globals() and 'mca' in globals():
    if row_coords.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=row_coords.iloc[:, 0],
            y=row_coords.iloc[:, 1],
            hue=best_labels,
            palette=PALETTE[:best_k], # Use defined pastel palette
            legend='full'
        )
        plt.title(f'MCA Biplot by Cluster (k={best_k})')
        dim1_inertia = mca.percentage_of_variance_[0]
        dim2_inertia = mca.percentage_of_variance_[1]
        plt.xlabel(f'Dim 1 ({dim1_inertia:.2f}% inertia)')
        plt.ylabel(f'Dim 2 ({dim2_inertia:.2f}% inertia)')
        plt.axhline(0, color='grey', lw=0.5, linestyle='--')
        plt.axvline(0, color='grey', lw=0.5, linestyle='--')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'mca_biplot_clusters.png', dpi=300)
        plt.close()
        print("MCA biplot saved to figs/mca_biplot_clusters.png")
    else:
        print(f"Skipping MCA biplot: Only {row_coords.shape[1]} dimension(s) found, but 2 are needed for the biplot.")
else:
    print("Skipping MCA biplot: Missing variables ('prince', 'row_coords', 'best_labels', or 'mca').")

# --- Logistic Regression: Predicting current deployment (Q3) ---
# Q3: 'We ARE deploying them now' as the positive class
df['deploy_now'] = (df['3'] == 'We ARE deploying them now').astype(int)

# Define predictors (Q1, Q2, Q4-Q10)
# Drop 'No opinion' and use first level as reference for dummy variables
predictor_cols = []
reference_levels = {}
for q_num in ['1', '2'] + [str(i) for i in range(4, 11)]:
    if q_num in df and df[q_num].notna().any():
        temp_series = df[q_num][df[q_num] != 'No opinion'].astype('category')
        if not temp_series.empty:
            reference_level = temp_series.cat.categories[0]
            reference_levels[f'Q{q_num}'] = reference_level
            dummies = pd.get_dummies(df[q_num], prefix=f'Q{q_num}', dummy_na=False)

            ref_col_name = f'Q{q_num}_{reference_level}'
            if ref_col_name in dummies.columns:
                dummies = dummies.drop(columns=[ref_col_name])

            no_opinion_col_name = f'Q{q_num}_No opinion'
            if no_opinion_col_name in dummies.columns:
                dummies = dummies.drop(columns=[no_opinion_col_name])

            # Specific fix for Q6 problematic category
            if q_num == '6':
                problematic_q6_col = 'Q6_50+ (massive specialization networks)'
                if problematic_q6_col in dummies.columns:
                    dummies = dummies.drop(columns=[problematic_q6_col])
                    print(f"INFO: Explicitly dropping problematic predictor '{problematic_q6_col}' for Q6.")

            predictor_cols.extend(dummies.columns)
            df = pd.concat([df, dummies], axis=1)
        else:
            print(f"Warning: Question Q{q_num} has no valid levels after excluding 'No opinion'. Skipping.")
    else:
        print(f"Warning: Question Q{q_num} not found in DataFrame or is all NaN. Skipping.")

if not predictor_cols:
    print("Error: No valid predictor columns were generated. Cannot proceed with logistic regression.")
    logit_model = None
    logit_results = None
else:
    final_predictors = [p for p in predictor_cols if p not in QUESTIONS and p not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]
    formula = 'deploy_now ~ ' + ' + '.join([f'Q("{p}")' for p in final_predictors])

    X_for_vif = df[final_predictors].copy()
    for col in X_for_vif.columns:
        if X_for_vif[col].dtype == 'bool':  # VIF doesn't like boolean
            X_for_vif[col] = X_for_vif[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(X_for_vif[col]):
            print(f"Warning: Column {col} is not numeric for VIF. Skipping VIF for this column or it might fail.")

    X_for_vif_dropna = X_for_vif.dropna()

    if not X_for_vif_dropna.empty and X_for_vif_dropna.shape[1] > 1:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_for_vif_dropna.columns
        try:
            vif_data["VIF"] = [
                variance_inflation_factor(X_for_vif_dropna.values, i)
                for i in range(X_for_vif_dropna.shape[1])
            ]
            vif_data.to_csv(RESULTS_DIR / 'logit_deployment_vif.csv', index=False)
            print("\nVIF data saved to results/logit_deployment_vif.csv")
            if (vif_data["VIF"] > 5).any():
                print("Warning: High multicollinearity detected (VIF > 5).")
        except Exception as e:
            print(f"Error calculating VIF: {e}. Check predictor columns for issues (e.g. perfect collinearity, non-numeric data).")
            print(f"X_for_vif_dropna dtypes:\n{X_for_vif_dropna.dtypes}")
            print(f"X_for_vif_dropna head:\n{X_for_vif_dropna.head()}")
    else:
        print("Skipping VIF calculation: Not enough data or predictors after handling NaNs or only one predictor.")

    try:
        logit_model = smf.logit(formula, data=df).fit(disp=0)  # disp=0 to suppress convergence messages
        logit_results = logit_model.summary()
        print("\nLogistic Regression Results:")
        print(logit_results)

        # Save detailed results (coefficients, p-values, CIs)
        params = logit_model.params
        conf_int = logit_model.conf_int()
        p_values_logit = logit_model.pvalues
        odds_ratios = np.exp(params)
        odds_ratios_conf_int = np.exp(conf_int)

        results_summary = pd.DataFrame({
            'coefficient': params,
            'p_value': p_values_logit,
            'conf_int_lower': conf_int.iloc[:, 0],
            'conf_int_upper': conf_int.iloc[:, 1],
            'odds_ratio': odds_ratios,
            'or_conf_int_lower': odds_ratios_conf_int.iloc[:, 0],
            'or_conf_int_upper': odds_ratios_conf_int.iloc[:, 1]
        })
        results_summary.to_csv(
            RESULTS_DIR / 'logit_deployment_coefs.csv',
            index_label='variable'
        )
        # Save full summary to a text file if summary2() is available and works
        try:
            with open(RESULTS_DIR / 'logit_deployment_summary.txt', 'w') as f:
                f.write(logit_model.summary2().as_text())
        except Exception as e:
            print(f"Could not save summary2: {e}")
            try:
                 with open(RESULTS_DIR / 'logit_deployment_summary.txt', 'w') as f:
                    f.write(str(logit_model.summary())) # Fallback to summary()
            except Exception as e_sum:
                 print(f"Could not save summary: {e_sum}")

        print("Logistic regression coefficients and summary saved to results/.")

    except Exception as e:
        print(f"Error during logistic regression: {e}")
        print(f"Formula used: {formula}")
        # print(f"Data dtypes:\n{df[final_predictors + ['deploy_now']].info()}")
        logit_model = None
        logit_results = None

# --- Forest Plot for Logistic Regression Odds Ratios (Figure 4) ---
if logit_model and 'results_summary' in locals():
    # Filter out intercept for the plot
    plot_data = results_summary.drop('Intercept', errors='ignore').copy()
    # Exclude rows where odds_ratio might be inf or NaN after filtering
    plot_data = plot_data[np.isfinite(plot_data['odds_ratio'])]
    plot_data = plot_data.sort_values(by='odds_ratio')

    # Error bars are OR - lower_ci, upper_ci - OR
    plot_data['err_lower'] = plot_data['odds_ratio'] - plot_data['or_conf_int_lower']
    plot_data['err_upper'] = plot_data['or_conf_int_upper'] - plot_data['odds_ratio']
    # Ensure errors are not negative (can happen with very wide CIs or OR close to 0)
    plot_data['err_lower'] = np.maximum(plot_data['err_lower'], 0)
    plot_data['err_upper'] = np.maximum(plot_data['err_upper'], 0)

    errors = [plot_data['err_lower'].values, plot_data['err_upper'].values]

    # Check for convergence issues
    has_convergence_issues = False
    if hasattr(logit_model, 'mle_retvals'):
        if logit_model.mle_retvals['converged'] is False:
            has_convergence_issues = True
    # Also check for very large odds ratios as sign of issues
    if (plot_data['odds_ratio'] > 20).any():
        has_convergence_issues = True

    # Clean up labels for y-axis (remove Q(...) and QX_ prefixes)
    def clean_label(label):
        # Remove formula escaping Q("...")
        if label.startswith('Q("') and label.endswith('")'):
            label = label[3:-2]
        # Remove specific type suffix like [T.True]
        if label.endswith('[T.True]'):
            label = label[:-8]
        # Remove general QX_ prefix
        parts = label.split('_', 1)
        if len(parts) > 1 and parts[0].startswith('Q') and parts[0][1:].isdigit():
            label = parts[1]
        # Replace underscores with spaces for readability
        label = label.replace('_', ' ')
        return label

    cleaned_labels = [clean_label(idx) for idx in plot_data.index]

    if not plot_data.empty:
        plt.figure(figsize=(10, max(4, len(plot_data) * 0.4 + 1)))

        # Plot the point estimates
        plt.errorbar(
            plot_data['odds_ratio'],
            np.arange(len(plot_data)),
            xerr=errors,
            fmt='o',
            color='black',
            ecolor='gray',
            capsize=3,
            linestyle='None',
            markersize=5,
            zorder=3  # Make sure points are on top
        )

        # Add semi-transparent confidence interval areas
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            lower_ci = row['or_conf_int_lower']
            upper_ci = row['or_conf_int_upper']
            plt.barh(
                i,
                width=upper_ci-lower_ci,
                left=lower_ci,
                height=0.3,
                color='skyblue',
                alpha=0.3,
                zorder=1  # Make sure this is below the points
            )

        # Add reference line at OR=1
        plt.axvline(1, color='red', linestyle='--', lw=1, zorder=2)

        plt.yticks(np.arange(len(plot_data)), cleaned_labels)
        plt.xlabel('Odds Ratio (95% CI)', fontsize=12)

        # Check for extreme values that might be truncated
        has_extreme_values = False
        extreme_threshold = 15.0  # Define what's considered extreme

        # Add title with convergence warning if needed
        title = 'Forest Plot: Odds Ratios for Current Agent Deployment'
        if has_convergence_issues:
            title += '\n⚠️ Warning: Model convergence issues detected'
        if (plot_data['odds_ratio'] > extreme_threshold).any() or (plot_data['or_conf_int_upper'] > extreme_threshold).any():
            title += '\n(Some extreme values may be truncated)'
            has_extreme_values = True
        plt.title(title, fontsize=14)

        plt.xscale('log')

        min_ci = plot_data['or_conf_int_lower'].min()
        max_ci = plot_data['or_conf_int_upper'].max()
        default_xlim = (0.1, 10.0)
        final_xlim = list(default_xlim)

        if np.isfinite(min_ci) and np.isfinite(max_ci) and min_ci > 0 and max_ci > 0:
            if max_ci / min_ci < 1000:  # Avoid extreme ranges due to outliers
                final_xlim[0] = max(0.01, min_ci * 0.5)
                final_xlim[1] = max(10.0, min(20.0, max_ci * 1.2))
        final_xlim[0] = max(0.01, final_xlim[0])

        plt.xlim(final_xlim)

        # Add significance stars and annotations for extreme values
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            # Add significance stars
            if row['p_value'] < 0.001:
                plt.text(final_xlim[1]*1.05, i, '***', va='center', fontsize=10)
            elif row['p_value'] < 0.01:
                plt.text(final_xlim[1]*1.05, i, '**', va='center', fontsize=10)
            elif row['p_value'] < 0.05:
                plt.text(final_xlim[1]*1.05, i, '*', va='center', fontsize=10)

            # Add numerical labels for extreme values
            if has_extreme_values and (row['odds_ratio'] > extreme_threshold or
                                       row['or_conf_int_upper'] > extreme_threshold):
                if row['odds_ratio'] > extreme_threshold:
                    plt.text(
                        final_xlim[1]*0.9,
                        i,
                        f"OR: {row['odds_ratio']:.1f}",
                        va='center',
                        ha='right',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )

        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'logit_forest_plot.png', dpi=300)
        plt.close()
        print("Forest plot for logistic regression saved to figs/logit_forest_plot.png")

        # --- Additional Forest Plot for Significant Predictors Only (Figure 4b) ---
        # Filter for only significant predictors (p < 0.05)
        sig_data = plot_data[plot_data['p_value'] < 0.05].copy()

        if not sig_data.empty:
            # Get cleaned labels for significant predictors only
            sig_labels = [clean_label(idx) for idx in sig_data.index]

            # Calculate error bars for significant predictors
            sig_errors = [sig_data['err_lower'].values, sig_data['err_upper'].values]

            plt.figure(figsize=(10, max(4, len(sig_data) * 0.6 + 1)))

            # Plot the point estimates
            plt.errorbar(
                sig_data['odds_ratio'],
                np.arange(len(sig_data)),
                xerr=sig_errors,
                fmt='o',
                color='black',
                ecolor='gray',
                capsize=3,
                linestyle='None',
                markersize=5,
                zorder=3
            )

            # Add semi-transparent confidence interval areas
            for i, (idx, row) in enumerate(sig_data.iterrows()):
                lower_ci = row['or_conf_int_lower']
                upper_ci = row['or_conf_int_upper']
                plt.barh(
                    i,
                    width=upper_ci-lower_ci,
                    left=lower_ci,
                    height=0.3,
                    color='skyblue',
                    alpha=0.3,
                    zorder=1
                )

            # Add reference line at OR=1
            plt.axvline(1, color='red', linestyle='--', lw=1, zorder=2)

            plt.yticks(np.arange(len(sig_data)), sig_labels)
            plt.xlabel('Odds Ratio (95% CI)', fontsize=12)

            # Add title
            title = 'Significant Predictors of Current Agent Deployment'
            if has_extreme_values:
                title += '\n(p < 0.05, some extreme values may be truncated)'
            plt.title(title, fontsize=14)

            plt.xscale('log')

            # Use the same x-axis limits for consistency
            plt.xlim(final_xlim)

            # Add significance stars
            for i, (idx, row) in enumerate(sig_data.iterrows()):
                if row['p_value'] < 0.001:
                    plt.text(final_xlim[1]*1.05, i, '***', va='center', fontsize=10)
                elif row['p_value'] < 0.01:
                    plt.text(final_xlim[1]*1.05, i, '**', va='center', fontsize=10)
                elif row['p_value'] < 0.05:
                    plt.text(final_xlim[1]*1.05, i, '*', va='center', fontsize=10)

                # Add numerical labels for extreme values
                if has_extreme_values and (row['odds_ratio'] > extreme_threshold or
                                           row['or_conf_int_upper'] > extreme_threshold):
                    if row['odds_ratio'] > extreme_threshold:
                        plt.text(
                            final_xlim[1]*0.9,
                            i,
                            f"OR: {row['odds_ratio']:.1f}",
                            va='center',
                            ha='right',
                            fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                        )

            plt.tight_layout()
            plt.savefig(FIGS_DIR / 'logit_forest_plot_significant.png', dpi=300)
            plt.close()
            print("Forest plot of significant predictors saved to figs/logit_forest_plot_significant.png")
        else:
            print("No significant predictors found for the reduced forest plot.")
    else:
        print("Skipping forest plot: No data to plot after filtering.")
else:
    print("Skipping forest plot: Logistic regression model or results not available.")

# --- Create combined summary figure ---
if 'cramers' in globals() and logit_model and 'pairwise_tests.csv' in [f.name for f in RESULTS_DIR.glob('*.csv')]:
    plt.figure(figsize=(15, 10))

    # 1. Top-left: Top 3 strongest Cramér's V associations
    plt.subplot(2, 2, 1)
    # Find top 3 associations
    cramers_flat = cramers.copy()
    np.fill_diagonal(cramers_flat.values, 0)  # Remove self-associations
    cramers_flat = cramers_flat.stack().sort_values(ascending=False)
    top3 = cramers_flat.head(3)

    # Create a simple bar chart
    plt.bar(
        [f"Q{a}-Q{b}" for a, b in top3.index],
        top3.values,
        color=PALETTE[:3],
        edgecolor='grey'
    )
    plt.ylabel("Cramér's V", fontsize=12)
    plt.title("Top 3 Question Associations", fontsize=14)
    plt.ylim(0, 1)

    # Annotate bars
    for i, v in enumerate(top3.values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    # 2. Top-right: Significant predictors from logistic regression
    if logit_model and 'results_summary' in locals():
        plt.subplot(2, 2, 2)
        sig_predictors = sig_data.copy()

        if not sig_predictors.empty:
            sig_predictors = sig_predictors.sort_values('odds_ratio', ascending=False)
            # Use clean labels for readability
            labels = [clean_label(idx) for idx in sig_predictors.index]

            bars = plt.barh(
                np.arange(len(sig_predictors)),
                sig_predictors['odds_ratio'],
                color=PALETTE[:len(sig_predictors)],
                edgecolor='grey'
            )
            plt.axvline(1, color='red', linestyle='--', lw=1)
            plt.yticks(np.arange(len(sig_predictors)), labels)
            plt.xlabel('Odds Ratio', fontsize=12)
            plt.title('Significant Predictors of Deployment', fontsize=14)
            plt.xscale('log')

            # Add significance stars
            for i, (idx, row) in enumerate(sig_predictors.iterrows()):
                if row['p_value'] < 0.001:
                    stars = '***'
                elif row['p_value'] < 0.01:
                    stars = '**'
                else:  # < 0.05
                    stars = '*'
                plt.text(
                    sig_predictors['odds_ratio'].max() * 1.1,
                    i,
                    stars,
                    va='center',
                    fontsize=10
                )
        else:
            plt.text(0.5, 0.5, "No significant predictors found",
                    ha='center', va='center', transform=plt.gca().transAxes)

    # 3. Bottom-left: Distribution of Q1 (coding automation timeline)
    plt.subplot(2, 2, 3)
    counts = df['1'].value_counts(normalize=True, dropna=False)
    if 'No opinion' in counts.index:
        no_opinion = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion})])
    else:
        counts = counts.sort_values(ascending=False)

    # Create wrapped labels for better readability
    labels = counts.index
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in labels]

    bars = plt.bar(
        np.arange(len(counts)),
        counts.values,
        color=PALETTE[:len(counts)],
        edgecolor='grey'
    )
    plt.xticks(np.arange(len(counts)), wrapped_labels, rotation=45, ha='right')
    plt.ylabel('Proportion', fontsize=12)
    plt.title('Q1: AI Code Generation Timeline', fontsize=14)

    # Add direct labels on bars
    for i, v in enumerate(counts.values):
        if v >= 0.05:  # Only add label if bar is tall enough
            plt.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

    # 4. Bottom-right: Distribution of Q3 (deployment status)
    plt.subplot(2, 2, 4)
    counts = df['3'].value_counts(normalize=True, dropna=False)
    if 'No opinion' in counts.index:
        no_opinion = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion})])
    else:
        counts = counts.sort_values(ascending=False)

    # Create wrapped labels for better readability
    labels = counts.index
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in labels]

    bars = plt.bar(
        np.arange(len(counts)),
        counts.values,
        color=PALETTE[:len(counts)],
        edgecolor='grey'
    )
    plt.xticks(np.arange(len(counts)), wrapped_labels, rotation=45, ha='right')
    plt.ylabel('Proportion', fontsize=12)
    plt.title('Q3: Current Deployment Status', fontsize=14)

    # Add direct labels on bars
    for i, v in enumerate(counts.values):
        if v >= 0.05:  # Only add label if bar is tall enough
            plt.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

    plt.suptitle('Survey Key Findings: AI Code Generation & Deployment', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(FIGS_DIR / 'key_findings_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Key findings summary visualization saved to figs/key_findings_summary.png")
else:
    print("Skipping combined figure: Missing required data.")

# --- Mixed-effects model for deployment (Q3) ---
# The paper mentions this was illustrative, so we may not need to fully run/save plots
# Q3: 'We ARE deploying them now' as the positive class
# df_mixed = df.copy()