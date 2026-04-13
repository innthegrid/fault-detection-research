# src/data_utils/scalers.py
import numpy as np


def min_max_scaler(data, min_val=None, max_val=None):
    """
    Normalizes data to [0, 1] range (Required by TimeGAN).

    Args:
        data: input array
        min_val, max_val: optional precomputed stats (for consistency across splits)

    Returns:
        norm_data: normalized data
        min_val: minimum value used
        max_val: maximum value used
    """
    data = np.asarray(data, dtype=np.float32)

    if min_val is None or max_val is None:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

    denom = max_val - min_val
    denom = np.where(denom == 0, 1e-7, denom)

    norm_data = (data - min_val) / denom

    return norm_data, min_val, max_val


def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardizes KPI observations using Z-score (Used by FCVAE).

    Args:
        values: 1D array
        mean, std: optional precomputed statistics
        excludes: boolean mask for values to ignore when computing stats

    Returns:
        standardized values, mean, std
    """
    values = np.asarray(values, dtype=np.float32)

    # ---------------------------
    # Input validation
    # ---------------------------
    if values.ndim != 1:
        raise ValueError("`values` must be a 1-D array")

    if (mean is None) != (std is None):
        raise ValueError("`mean` and `std` must be both None or both provided")

    if excludes is not None:
        excludes = np.asarray(excludes, dtype=bool)
        if excludes.shape != values.shape:
            raise ValueError(
                f"Shape mismatch: excludes {excludes.shape} vs values {values.shape}"
            )

    # ---------------------------
    # Compute statistics
    # ---------------------------
    if mean is None:
        if excludes is not None:
            valid = values[~excludes]
        else:
            valid = values

        if len(valid) == 0:
            # Edge case: all values excluded
            mean = 0.0
            std = 1.0
        else:
            mean = np.mean(valid)
            std = np.std(valid)

    # ---------------------------
    # Safe standard deviation
    # ---------------------------
    if std == 0 or np.isnan(std):
        std = 1e-8

    standardized = (values - mean) / std

    return standardized, mean, std
