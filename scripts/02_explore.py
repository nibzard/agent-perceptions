import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.font_manager as fm
from pathlib import Path
import json
import textwrap

# --- Config ---
DATA_PATH = 'data/clean_survey.parquet'
FIGS_DIR = Path('figs')
FIGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
QUESTIONS = [str(i) for i in range(1, 11)]
PALETTE = sns.color_palette('pastel')  # Use pastel palette
FIGSIZE = (6, 4)
MCA_FIGSIZE = (6, 6)
FONT = 'Roboto'

# Try to set Roboto font if available
if FONT in [f.name for f in fm.fontManager.ttflist]:
    plt.rcParams['font.family'] = FONT
    plt.rcParams['font.size'] = 10  # Default font size
else:
    plt.rcParams['font.size'] = 10  # Default font size

# --- Load data ---
df = pd.read_parquet(DATA_PATH)

# --- 1. Ten 100% horizontal bar charts (Q1-Q10) ---
for q in QUESTIONS:
    counts = df[q].value_counts(normalize=True, dropna=False)
    # Separate 'No opinion' and sort others by value descending
    if 'No opinion' in counts.index:
        no_opinion = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion})])
    else:
        counts = counts.sort_values(ascending=False)
    # Invert the order for barh (so largest is at top, 'No opinion' at bottom)
    counts = counts[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(counts) + 1.5))
    bars = counts.plot.barh(ax=ax, color=PALETTE[:len(counts)], edgecolor='grey')
    ax.set_xlabel('Proportion', fontsize=12)
    ax.set_ylabel(f'Q{q} Answer', fontsize=12)
    ax.set_title(f'Q{q}: Distribution of Answers', fontsize=14)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Set x-axis to max value + 15% padding instead of 0-1 range
    max_val = counts.max()
    x_max = min(1.0, max_val * 1.15)  # Cap at 1.0 if needed
    ax.set_xlim(0, x_max)

    # Place labels inside bars when possible
    for i, v in enumerate(counts):
        # Place inside if bar width allows, otherwise outside
        if v > 0.1:  # Threshold for inside/outside placement
            ax.text(v/2, i, f'{v:.1%}', va='center', ha='center',
                    color='black', fontweight='bold', fontsize=9)
        else:
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=9)

    # Wrap long y-axis labels
    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=30)) for label in labels]
    ax.set_yticklabels(wrapped_labels)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / f'Q{q}_barh.png', dpi=300)
    plt.close()

# --- 1b. Grid of 10 bar charts (Q1-Q10) in one image ---
fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True)
for idx, q in enumerate(QUESTIONS):
    ax = axes[idx // 5, idx % 5]
    counts = df[q].value_counts(normalize=True, dropna=False)
    # Separate 'No opinion' and sort others by value descending
    if 'No opinion' in counts.index:
        no_opinion = counts['No opinion']
        counts = counts.drop('No opinion')
        counts = counts.sort_values(ascending=False)
        counts = pd.concat([counts, pd.Series({'No opinion': no_opinion})])
    else:
        counts = counts.sort_values(ascending=False)
    counts = counts[::-1]
    bars = counts.plot.barh(ax=ax, color=PALETTE[:len(counts)], edgecolor='grey')
    ax.set_xlabel('Proportion', fontsize=10)
    ax.set_title(f'Q{q}', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    # Set x-axis to max value + 15% padding instead of 0-1 range
    max_val = counts.max()
    x_max = min(1.0, max_val * 1.15)  # Cap at 1.0 if needed
    ax.set_xlim(0, x_max)

    # Place labels inside bars when possible
    for i, v in enumerate(counts):
        if v > 0.1:  # Threshold for inside/outside placement
            ax.text(v/2, i, f'{v:.1%}', va='center', ha='center',
                    color='black', fontweight='bold', fontsize=7)
        else:
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=7)

    # Wrap long y-axis labels in grid
    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=25)) for label in labels]
    ax.set_yticklabels(wrapped_labels)

plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Add space between subplots
plt.suptitle('All Survey Questions (Q1-Q10)', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent title overlap
plt.savefig(FIGS_DIR / 'all_questions_grid.png', dpi=300)
plt.close()

# --- 1c. Heatmap of answer proportions for all questions (Q1-Q10) ---
# Get all unique answer categories across all questions
answer_categories = sorted(
    set(val for q in QUESTIONS for val in df[q].dropna().unique())
)
heatmap_data = []
for q in QUESTIONS:
    counts = df[q].value_counts(normalize=True, dropna=False)
    row = [counts.get(cat, 0) for cat in answer_categories]
    heatmap_data.append(row)
heatmap_df = pd.DataFrame(
    heatmap_data, index=[f'Q{q}' for q in QUESTIONS], columns=answer_categories
)

# Increased height, width depends on number of categories
plt.figure(figsize=(max(10, 0.6 * len(answer_categories)), 10))
sns.heatmap(
    heatmap_df, annot=True, fmt='.2f', cmap='Blues',
    annot_kws={"size": 8},  # Control annotation font size
    linewidths=.5,  # Add lines between cells
    cbar_kws={'label': 'Proportion'}
)
plt.xticks(rotation=45, ha='right', fontsize=9)  # Rotate x-axis labels
plt.yticks(fontsize=10)
plt.title('Survey Answer Proportions (Q1–Q10)', fontsize=14)
plt.ylabel('Question', fontsize=12)
plt.xlabel('Answer', fontsize=12)
plt.tight_layout()  # Adjust layout to make room for labels
plt.savefig(FIGS_DIR / 'all_questions_heatmap.png', dpi=300)
plt.close()

# --- Save all values from plots as JSON ---
# Collect proportions for each question and answer
results = {}
for q in QUESTIONS:
    counts = df[q].value_counts(normalize=True, dropna=False)
    # Ensure all answer categories are included
    all_answers = sorted(set(val for val in df[q].dropna().unique()))
    q_results = {str(ans): float(counts.get(ans, 0)) for ans in all_answers}
    # Also include 'No opinion' if present in any question
    if 'No opinion' not in q_results and 'No opinion' in df[q].unique():
        q_results['No opinion'] = float(counts.get('No opinion', 0))
    results[f'Q{q}'] = q_results

# Save to JSON file
with open(RESULTS_DIR / 'all_questions_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# --- 2. Mosaic plot Q1 × Q3 ---
# Abbreviated/wrapped labels for Q1
q1_labels = {
    "Never - creativity can't be automated": "Never -\ncreativity",
    '5-10 years for most programming tasks': '5-10 yrs\nmost prog',
    'Already happening for simple tasks': 'Already\nhappening',
    'Machines will write 99% of code by 2030': '99% by\n2030',
    'No opinion': 'No opinion',
}
# Abbreviated/wrapped labels for Q3
q3_labels = {
    'We ARE deploying them now': 'Deploying\nnow',
    'Regulatory/compliance concerns': 'Regulatory/\ncompliance',
    'Fear of the unknown': 'Fear of\nunknown',
    "Technical leadership doesn't understand the potential": "Leadership\ndoesn't understand",
    'Regulatory/compliance or technical readiness concerns': 'Compliance/\nreadiness',
    'No opinion': 'No opinion',
}

# Prepare a DataFrame with mapped labels
mosaic_df = df[['1', '3']].copy()
mosaic_df['1'] = mosaic_df['1'].map(q1_labels)
mosaic_df['3'] = mosaic_df['3'].map(q3_labels)

# Define colors based on the first variable (Q1)
q1_cats = mosaic_df['1'].astype('category').cat.categories
color_map = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(q1_cats)}


def get_props(key):
    # Key is a tuple, e.g., ('Already\nhappening', 'Deploying\nnow')
    # Color based on the first element of the key (Q1 answer)
    q1_answer = key[0]
    return {'color': color_map.get(q1_answer, '#dddddd'), 'edgecolor': 'white'}


fig, ax = plt.subplots(figsize=(8, 6))
# Apply the properties function to set colors and gap between tiles
mosaic(mosaic_df, ['1', '3'], ax=ax, properties=get_props, gap=0.02)
ax.set_title('Mosaic Plot: Q1 × Q3 (Pastel Colors, Abbreviated Labels)')
plt.tight_layout()
plt.savefig(FIGS_DIR / 'Q1xQ3_mosaic.png', dpi=300)
plt.close()

# --- 3. Cramér's V heatmap (Q1-Q10) ---

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


cramers = pd.DataFrame(index=QUESTIONS, columns=QUESTIONS, dtype=float)
for i in QUESTIONS:
    for j in QUESTIONS:
        if i == j:
            cramers.loc[i, j] = 1.0
        else:
            cramers.loc[i, j] = cramers_v(df[i], df[j])

fig, ax = plt.subplots(figsize=MCA_FIGSIZE)
sns.heatmap(
    cramers.astype(float),
    annot=True,
    fmt='.2f',
    cmap='Blues',
    ax=ax,
    cbar_kws={'label': "Cramér's V"}
)
ax.set_title("Cramér's V Heatmap (Q1–Q10)")
plt.tight_layout()
plt.savefig(FIGS_DIR / 'cramers_v_heatmap.png', dpi=300)
plt.close()

print('Exploratory visualisations saved to figs/.')