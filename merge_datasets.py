"""
Prepare training and test datasets by merging OLID, AbuseEval, and HIC datasets.

- Creates a unified train set (OLID + HIC)
- Creates a test set by combining OLID labels and AbuseEval annotations
- Outputs: TSV files for training and testing

NOTE: Make sure the input files exist at the specified paths before running this script.
Expected input files:
- OLID: ./data/OLIDv1/
- AbuseEval: ./data/AbuseEval-master/
- HIC: ./data/implicit-hate-corpus/
"""

import pandas as pd

# File paths

olid_train_path = "./data/OLIDv1/olid-training-v1.0.tsv"
abuseeval_train_path = "./data/AbuseEval-master/data/abuseval_labels/abuseval_offenseval_train.tsv"
hic_path = './data/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv'
intermediate_output_path = "./data/train/olid_abuseeval_train.tsv"
final_output_path = './data/train/olid_abuseeval_hic_train.tsv'

olid_test_path = "./data/OLIDv1/testset-levela.tsv"
labels_a_path = "./data/OLIDv1/labels-levela.csv"
labels_b_path = "./data/OLIDv1/labels-levelb.csv"
labels_c_path = "./data/OLIDv1/labels-levelc.csv"
abuseeval_test_path = "./data/AbuseEval-master/data/abuseval_labels/abuseval_offenseval_test.tsv"
output_test_path = "./data/test/olid_abuseeval_test.tsv"

def merge_olid_and_abuseeval(olid_path, abuseeval_path):
    """
    Merge OLID and AbuseEval train sets on 'id'.
    
    Args:
        olid_path (str): Path to OLID training file.
        abuseeval_path (str): Path to AbuseEval training file.
    
    Returns:
        pd.DataFrame: Merged training DataFrame with 'abuse' label.
    """
    olid_df = pd.read_csv(olid_path, sep='\t')
    abuseeval_df = pd.read_csv(abuseeval_path, sep='\t')
    common_ids = set(olid_df['id']).intersection(abuseeval_df['id'])
    olid_df = olid_df[olid_df['id'].isin(common_ids)]
    abuseeval_df = abuseeval_df[abuseeval_df['id'].isin(common_ids)]
    return pd.merge(olid_df, abuseeval_df[['id', 'abuse']], on='id', how='left')

def prepare_hic_data(hic_file_path):
    """
    Format HIC data to match OLID/AbuseEval structure.

    Args:
        hic_file_path (str): Path to HIC dataset.
    
    Returns:
        pd.DataFrame: Processed HIC data with 'id', 'tweet', 'abuse'.
    """
    hic_df = pd.read_csv(hic_file_path, sep='\t')
    label_map = {
        'implicit_hate': 'IMP',
        'explicit_hate': 'EXP',
        'not_hate': 'NOTABU'
    }
    hic_df['class'] = hic_df['class'].map(label_map)
    hic_df.insert(0, 'id', [str(i) for i in range(1, len(hic_df) + 1)])
    hic_df.rename(columns={'post': 'tweet', 'class': 'abuse'}, inplace=True)
    return hic_df[['id', 'tweet', 'abuse']]

def create_test_set(olid_test_path, labels_a_path, labels_b_path, labels_c_path, abuseeval_test_path, output_path):
    """
    Merge OLID test set with labels A/B/C and AbuseEval 'abuse' label.

    Args:
        olid_test_path (str): Path to OLID test tweets.
        labels_a_path (str): Subtask A labels.
        labels_b_path (str): Subtask B labels.
        labels_c_path (str): Subtask C labels.
        abuseeval_test_path (str): AbuseEval test labels.
        output_path (str): Output file path for merged test set.
    """
    olid_test_df = pd.read_csv(olid_test_path, sep='\t')
    labels_a = pd.read_csv(labels_a_path, header=None, names=['id', 'subtask_a'])
    labels_b = pd.read_csv(labels_b_path, header=None, names=['id', 'subtask_b'])
    labels_c = pd.read_csv(labels_c_path, header=None, names=['id', 'subtask_c'])

    olid_test_df['subtask_a'] = olid_test_df['id'].map(labels_a.set_index('id')['subtask_a'])
    olid_test_df['subtask_b'] = olid_test_df['id'].map(labels_b.set_index('id')['subtask_b'])
    olid_test_df['subtask_c'] = olid_test_df['id'].map(labels_c.set_index('id')['subtask_c'])

    abuseeval_test_df = pd.read_csv(abuseeval_test_path, sep='\t')
    common_ids = set(olid_test_df['id']).intersection(abuseeval_test_df['id'])
    olid_test_df = olid_test_df[olid_test_df['id'].isin(common_ids)]
    abuseeval_test_df = abuseeval_test_df[abuseeval_test_df['id'].isin(common_ids)]

    olid_test_df['abuse'] = olid_test_df['id'].map(abuseeval_test_df.set_index('id')['abuse'])
    olid_test_df.to_csv(output_path, sep='\t', index=False)
    print(f"{output_path} saved.")

def main():
    """
    Run data preparation steps:
    - Merge OLID and AbuseEval train sets
    - Format and append HIC data
    - Create merged test set
    """
    # Merge OLID + AbuseEval train
    olid_abuseeval_df = merge_olid_and_abuseeval(olid_train_path, abuseeval_train_path)
    olid_abuseeval_df.to_csv(intermediate_output_path, sep='\t', index=False)
    print(f"{intermediate_output_path} saved.")

    # Format HIC and append to train
    hic_df = prepare_hic_data(hic_path)
    for col in ['subtask_a', 'subtask_b', 'subtask_c']:
        hic_df[col] = None

    columns = ['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c', 'abuse']
    olid_abuseeval_df = olid_abuseeval_df.reindex(columns=columns)
    hic_df = hic_df.reindex(columns=columns)

    final_df = pd.concat([olid_abuseeval_df, hic_df], ignore_index=True)
    final_df.to_csv(final_output_path, sep='\t', index=False)
    print(f"{final_output_path} saved.")

    # Create test set
    create_test_set(
        olid_test_path,
        labels_a_path,
        labels_b_path,
        labels_c_path,
        abuseeval_test_path,
        output_test_path
    )

if __name__ == "__main__":
    main()