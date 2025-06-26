"""
Dataset Statistics and Class Distributions

This script loads and analyzes the OLID AbuseEval datasets (HIC, Main/Train, Test),
printing abuse class distributions and subtask label counts.
"""

import pandas as pd
from transformers import AutoTokenizer

def print_abuse_distribution(df, label):
    """
    Print count and percentage distribution of the 'abuse' column.

    Args:
        df (pd.DataFrame): Input dataset with an 'abuse' column.
        label (str): Label for identifying the dataset in output.
    """
    total = len(df)
    abuse_counts = df['abuse'].value_counts(dropna=False)
    abuse_percentages = df['abuse'].value_counts(normalize=True, dropna=False) * 100

    distribution = pd.DataFrame({
        'Count': abuse_counts,
        'Percentage': abuse_percentages.round(2)
    })

    print(f"\nAbuse Distribution - {label}")
    print(distribution)
    print(f"Total rows: {total}")
    print("-" * 60)

def print_distribution(df, name):
    """
    Print value counts for subtask_a, subtask_b, and subtask_c columns.

    Args:
        df (pd.DataFrame): Input dataset with subtask columns.
        name (str): Label for identifying the dataset in output.
    """
    label_map = {
        'subtask_a': 'Subtask A – Offensive Language Detection',
        'subtask_b': 'Subtask B – Type of Offensive Language',
        'subtask_c': 'Subtask C – Target of Offense'
    }

    print(f"\nSubtask Distribution - {name}")
    for subtask in ['subtask_a', 'subtask_b', 'subtask_c']:
        if subtask in df.columns:
            print(f"\n{label_map[subtask]}:")
            print(df[subtask].value_counts(dropna=False))

def print_subtask_a_imp_exp(df, name):
    """
    Print subtask_a counts only for rows where abuse is IMP or EXP.

    Args:
        df (pd.DataFrame): Input dataset with 'abuse' and 'subtask_a' columns.
        name (str): Label for identifying the dataset in output.
    """
    if {'abuse', 'subtask_a'}.issubset(df.columns):
        filtered_df = df[df['abuse'].isin(['IMP', 'EXP'])]
        print(f"\nSubtask A (IMP or EXP only) - {name}")
        print(filtered_df['subtask_a'].value_counts(dropna=False))

# File paths
hic_path = "./data/train/olid_abuseeval_hic_train.tsv"
main_path = "./data/train/olid_abuseeval_train.tsv"
test_path = "./data/test/olid_abuseeval_test.tsv"

# Load and analyze HIC file
print(f"\nLoading: {hic_path}")
df_hic = pd.read_csv(hic_path, sep='\t')
print_abuse_distribution(df_hic, "HIC File (Full)")

# Print distribution from first occurrence of id == 1
start_index_hic = df_hic.index[df_hic['id'] == 1].min()
if not pd.isna(start_index_hic):
    df_hic_from_id1 = df_hic.loc[start_index_hic:]
    print_abuse_distribution(df_hic_from_id1, "HIC File (From ID == 1)")

    print("\nFirst 5 rows from ID == 1 onward:")
    print(df_hic_from_id1.head(5))

# Load and analyze Main (Train) file
print(f"\nLoading: {main_path}")
train_df = pd.read_csv(main_path, sep='\t')
print_abuse_distribution(train_df, "Main (Train) File")

# Load and analyze Test file
print(f"\nLoading: {test_path}")
test_df = pd.read_csv(test_path, sep='\t')
print_abuse_distribution(test_df, "Test File")

# Show first few rows
print("\nFirst rows of Main (Train) File:")
print(train_df.head())

print("\nFirst rows of Test File:")
print(test_df.head())

# Print subtask distributions
print_distribution(train_df, "Main (Train) File")
print_distribution(test_df, "Test File")

# Analyze token lengths using a tokenizer
print("\nAnalyzing token lengths using BERT tokenizer...")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Or replace with "GroNLP/hateBERT"

# Tokenize and count tokens
token_lengths = train_df['tweet'].apply(lambda x: len(tokenizer.tokenize(str(x))))

# Count how many tweets have more than 128 tokens
count_over_128 = (token_lengths > 128).sum()
percent_over_128 = (count_over_128 / len(token_lengths)) * 100

print(f"Tweets > 128 tokens: {count_over_128} ({percent_over_128:.2f}%)")