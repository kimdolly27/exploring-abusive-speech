"""
Script for error analysis and visualization of ternary classification predictions.

Includes:
- Confusion matrices and classification reports
- Mismatched subtask_a vs abuse label filtering
- IMP prediction overlap analysis across models
- subtask_b distribution per confusion matrix cell
- Tweet length comparisons for specific error cases
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Create output directory
os.makedirs('./error_analysis', exist_ok=True)

# Load prediction files
df_targeted = pd.read_csv("./predictions/ternary-CoT-targeted-P1-t0.00_predictions.csv")
df_ihc = pd.read_csv("./predictions/ternary-CoT-IHC_labels-P1-t0.00_predictions.csv")
df_base = pd.read_csv("./predictions/ternary-base-P4-t0.00_predictions.csv")

# Normalize labels
for df in [df_targeted, df_ihc, df_base]:
    df['abuse'] = df['abuse'].str.strip().str.upper()
    df['prediction'] = df['prediction'].str.strip().str.upper()

labels = ['EXP', 'IMP', 'NOTABU']

# Confusion matrix and classification report
cm = confusion_matrix(df_targeted['abuse'], df_targeted['prediction'], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - CoT Targeted (0 shots)")
plt.tight_layout()
plt.savefig("./error_analysis/preconfusion_matrix_targeted_0shots.jpg", dpi=300)
plt.show()

print("Classification Report - CoT Targeted (0 shots)")
print(classification_report(df_targeted['abuse'], df_targeted['prediction'], labels=labels, digits=2))

# Mismatched subtask_a cases
case1 = df_targeted[(df_targeted['subtask_a'] == 'OFF') & (df_targeted['abuse'] == 'NOTABU')]
case2 = df_targeted[(df_targeted['subtask_a'] == 'NOT') & (df_targeted['abuse'].isin(['EXP', 'IMP']))]
mismatched_cases = pd.concat([case1, case2])
mismatched_cases.to_csv("./error_analysis/mismatched_subtask_a_vs_abuse.csv", index=False)

# Confusion matrix for mismatched cases
mismatched_cases['abuse'] = mismatched_cases['abuse'].str.strip().str.upper()
mismatched_cases['prediction'] = mismatched_cases['prediction'].str.strip().str.upper()
cm_mismatch = confusion_matrix(mismatched_cases['abuse'], mismatched_cases['prediction'], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mismatch, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Reannotation Mismatches")
plt.tight_layout()
plt.savefig("./error_analysis/confusion_matrix_mismatched.jpg", dpi=300)
plt.show()

print("Classification Report - Mismatched Examples")
print(classification_report(mismatched_cases['abuse'], mismatched_cases['prediction'], labels=labels))

# Get IMP ID sets
def get_imp_ids(df):
    return set(df[(df['abuse'] == 'IMP') & (df['prediction'] == 'IMP')]['id'])

imp_targeted = get_imp_ids(df_targeted)
imp_ihc = get_imp_ids(df_ihc)
imp_base = get_imp_ids(df_base)

# Save overlap subsets
def save_subset(subset_ids, filename, source_df):
    df_subset = source_df[source_df['id'].isin(subset_ids)]
    df_subset.to_csv(f'./error_analysis/{filename}.csv', index=False)

save_subset(imp_targeted & imp_ihc, 'imp_targeted_and_ihc', df_targeted)
save_subset(imp_targeted & imp_base, 'imp_targeted_and_base', df_targeted)
save_subset(imp_ihc & imp_base, 'imp_ihc_and_base', df_ihc)
save_subset(imp_targeted & imp_ihc & imp_base, 'imp_all_three', df_targeted)

print("IMP totals:")
print(f"Targeted: {len(imp_targeted)} | Base: {len(imp_base)} | IHC: {len(imp_ihc)}")
print("IMP overlaps:")
print(f"Targeted ∩ IHC: {len(imp_targeted & imp_ihc)}")
print(f"Targeted ∩ Base: {len(imp_targeted & imp_base)}")
print(f"IHC ∩ Base: {len(imp_ihc & imp_base)}")
print(f"All three: {len(imp_targeted & imp_ihc & imp_base)}")
print("IMP unique:")
print(f"Targeted only: {len(imp_targeted - imp_ihc - imp_base)}")
print(f"Base only: {len(imp_base - imp_targeted - imp_ihc)}")
print(f"IHC only: {len(imp_ihc - imp_targeted - imp_base)}")

# subtask_b distribution per confusion cell
df_targeted['subtask_b'] = df_targeted['subtask_b'].astype(str).str.strip().str.upper()
valid_b = df_targeted[df_targeted['subtask_b'].isin(['UNT', 'TIN'])]

rows = []
for true_label in labels:
    for pred_label in labels:
        cell = valid_b[
            (valid_b['abuse'] == true_label) &
            (valid_b['prediction'] == pred_label)
        ]
        total = len(cell)
        tin = cell['subtask_b'].value_counts().get('TIN', 0)
        unt = cell['subtask_b'].value_counts().get('UNT', 0)
        rows.append({
            'True_Label': true_label,
            'Pred_Label': pred_label,
            'Total': total,
            'TIN': tin,
            'UNT': unt,
            'TIN_%': round(tin / total * 100, 2) if total else 0,
            'UNT_%': round(unt / total * 100, 2) if total else 0
        })

df_b_dist = pd.DataFrame(rows)
print("subtask_b distribution by confusion cell")
print(df_b_dist)

# Plot TIN/UNT percentages for confusion pairs (excluding NOTABU as true label)
plot_df = df_b_dist[df_b_dist['True_Label'] != 'NOTABU'].copy()
plot_df['Cell'] = plot_df['True_Label'] + " → " + plot_df['Pred_Label']
x = range(len(plot_df))

fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.4
ax.bar([i - bar_width/2 for i in x], plot_df['TIN_%'], width=bar_width, label='TIN %')
ax.bar([i + bar_width/2 for i in x], plot_df['UNT_%'], width=bar_width, label='UNT %')
ax.set_xticks(x)
ax.set_xticklabels(plot_df['Cell'], rotation=45, ha='right')
ax.set_ylabel('Percentage')
ax.set_title('TIN vs. UNT per prediction cell')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./error_analysis/subtask_b_distribution_per_cell.png', dpi=300)
plt.show()

# Save specific error case subsets
imp_notabu = df_targeted[(df_targeted['abuse'] == 'IMP') & (df_targeted['prediction'] == 'NOTABU')]
exp_imp = df_targeted[(df_targeted['abuse'] == 'EXP') & (df_targeted['prediction'] == 'IMP')]

imp_notabu.to_csv('./error_analysis/IMP_predicted_as_NOTABU.csv', index=False)
exp_imp.to_csv('./error_analysis/EXP_predicted_as_IMP.csv', index=False)

# Compare tweet lengths for those error cases
def compare_lengths(file_path, label):
    if os.path.exists(file_path):
        subset_df = pd.read_csv(file_path)
        subset_df['tweet_length'] = subset_df['tweet'].str.len()
        df_targeted['tweet_length'] = df_targeted['tweet'].str.len()
        stats_subset = subset_df['tweet_length'].agg(['mean', 'median', 'std', 'count']).rename(label)
        stats_targeted = df_targeted['tweet_length'].agg(['mean', 'median', 'std', 'count']).rename('CoT_all')
        print(f"\nTweet length comparison: {label}")
        print(pd.concat([stats_subset, stats_targeted], axis=1))

compare_lengths('./error_analysis/IMP_predicted_as_NOTABU.csv', 'IMP→NOTABU')
compare_lengths('./error_analysis/EXP_predicted_as_IMP.csv', 'EXP→IMP')