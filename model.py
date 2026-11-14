"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    mask = np.isnan(X)
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0  # Replace all-NaN columns with 0
    X[mask] = np.take(col_means, np.where(mask)[1])
    return X

# Step 2 - compute_iqr_bounds
import numpy as np

def compute_iqr_bounds(X, k=1.5):
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    lower_bounds = q25 - k * iqr
    upper_bounds = q75 + k * iqr
    return lower_bounds, upper_bounds

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    return (numerator / (denominator + eps))

# Step 5 - append_column
def append_column(X, col):
    return np.column_stack((X, col))

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    sorted_labels = np.unique(labels)
    sorted_labels.sort()
    for i, label in enumerate(sorted_labels):
        labels[labels == label] = i
    X = np.zeros((len(labels), len(sorted_labels)))
    for i, label in enumerate(labels):
        X[i][int(label)] = 1
    return X

# Step 7 - fit_standardizer
import numpy as np

def fit_standardizer(X: np.ndarray):
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X.mean(axis=0), std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    return np.column_stack([np.ones(X.shape[0]), X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    return np.random.RandomState(seed).permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    return np.split(indices, [int(len(indices) * train_ratio), int(len(indices) * (train_ratio + val_ratio))])

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    return (X[indices], y[indices])
    # TODO: Select the rows of X and y at the given indices.
    pass

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

