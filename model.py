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

# Step 13 - ols_fit
def ols_fit(X, y):
    A = X.T @ X
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    return X @ theta.T

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_true - y_pred))

# Step 16 - root_mean_squared_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    return (np.sum((y_true - y_pred) ** 2) / len(y_true)) ** 0.5

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_mean = np.mean(y_true)
    SS_tot = np.sum((y_true - SS_mean) ** 2)
    if SS_tot == 0:
        return 0
    return 1 - (SS_res / SS_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    r = y_true - y_pred
    return {
        "mean": np.mean(r),
        "std": np.std(r),
        "median_abs": np.median(np.abs(r)),
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    lower_bounds, upper_bounds = compute_iqr_bounds(X, iqr_k)

    return clip_columns(X, lower_bounds,upper_bounds)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    ratio_feature = make_ratio_feature(X_num[:, ratio_num_idx], X_num[:, ratio_den_idx])
    X = append_column(X_num, ratio_feature)
    if cat_labels is not None:
        # print("One-hot encoding categorical labels...", one_hot_encode(cat_labels))
        X = append_column(X, one_hot_encode(cat_labels))
    return X

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    indices = make_shuffled_indices(X.shape[0], seed)
    part_indices = partition_indices(indices, train_ratio, val_ratio)
    # print(part_indices)
    X_train, y_train = subset_xy(X, y, part_indices[0])
    X_val, y_val = subset_xy(X, y, part_indices[1])
    X_test, y_test = subset_xy(X, y, part_indices[2])
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

