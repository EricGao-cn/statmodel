import pandas as pd
import numpy as np

from scipy import stats
from pprint import pprint

def entropy_weight(df, score_columns, valid_columns, consider_sample_size=False, log_sample_size=False):
    """
    Calculate weights for each score using the entropy weighting method.

    Args:
        df (pd.DataFrame): DataFrame containing the scores and validity indicators.
        score_columns (list): List of column names for the scores.
        valid_columns (list): List of column names for the validity indicators.
        consider_sample_size (bool): Whether to consider sample size in weighting.
        log_sample_size (bool): Whether to apply logarithm to sample sizes before normalization.

    Returns:
        dict: A dictionary containing the weights for each score.
    """
    # Step 1: Data Preprocessing
    df = df.copy()  # Avoid modifying the original DataFrame
    
    # Create a copy to store valid data counts for each column
    valid_data_counts = {}
    
    for i, score_col in enumerate(score_columns):
        valid_col = valid_columns[i]
        df[score_col] = df.apply(lambda row: row[score_col] if row[valid_col] == 1 else None, axis=1)
        # Count valid data points for each column
        valid_data_counts[score_col] = df[score_col].count()

    # Drop rows where all scores are NaN
    df = df.dropna(subset=score_columns, how='all').copy()
    df = df.fillna(0)

    # Step 2: Normalize the data
    def normalize(col):
        return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else np.zeros_like(col)

    df_normalized = df[score_columns].apply(normalize)

    # Step 3: Calculate the proportion of the i-th sample for the j-th indicator
    def calculate_proportion(col):
        total = col.sum()
        return col / total if total != 0 else np.zeros_like(col)

    df_proportions = df_normalized.apply(calculate_proportion)

    # Step 4: Calculate the entropy for each indicator
    def calculate_entropy(col):
        col = col[col > 0]
        return -np.sum(col * np.log(col)) / np.log(len(col)) if len(col) > 0 else 0

    entropy_values = df_proportions.apply(calculate_entropy)

    # Step 5: Calculate the redundancy degree
    redundancy = 1 - entropy_values

    # Step 6: Calculate the weights
    if consider_sample_size:
        # Convert valid data counts to a Series with the same index as redundancy
        sample_size_series = pd.Series([valid_data_counts[col] for col in redundancy.index], index=redundancy.index)
        
        # Apply logarithm to sample sizes if requested
        if log_sample_size:
            # Add 1 to avoid log(1)=0 issues
            sample_size_series = np.log(sample_size_series + 1)
        
        # Normalize sample sizes
        sample_size_weight = sample_size_series / sample_size_series.sum()
        
        # Combine entropy-based redundancy with sample size weights
        combined_weights = redundancy * sample_size_weight
        weights = combined_weights / combined_weights.sum()
    else:
        # Original calculation without sample size consideration
        weights = redundancy / redundancy.sum()

    return weights.to_dict()

# Read the Excel file
# Read the Excel file
file_path = "data/all_data_cleaned_with_scores_sorted_valid_only.xlsx"
df = pd.read_excel(file_path)

score_columns = ["sentiment_score", "storytelling_score", "character_performance_score", "production_score"]
valid_columns = ["valid_comment", "storytelling_valid", "character_performance_valid", "production_valid"]

# 不考虑样本大小的权重
weights_wo_sample_size = entropy_weight(df, score_columns, valid_columns)

# 考虑样本大小的权重（线性）
weights_wi_sample_size = entropy_weight(df, score_columns, valid_columns, consider_sample_size=True)

# 考虑对数化后的样本大小的权重
weights_wi_log_sample_size = entropy_weight(df, score_columns, valid_columns, consider_sample_size=True, log_sample_size=True)

print("\n不考虑样本大小的熵权值:")
pprint(weights_wo_sample_size)
print("\n考虑线性样本大小的熵权值:")
pprint(weights_wi_sample_size)
print("\n考虑对数样本大小的熵权值:")
pprint(weights_wi_log_sample_size)
