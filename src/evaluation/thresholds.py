# src/evaluation/thresholds.py
import numpy as np
import pandas as pd

from .metrics.detection_utils import point_adjust, calc_p2p


# ==========================================================
# Unsupervised Threshold
# ==========================================================
def get_percentile_threshold(train_scores, percentile=99.0):
    """
    Unsupervised threshold using training distribution.

    Args:
        train_scores: array-like anomaly scores
        percentile: cutoff percentile

    Returns:
        threshold value
    """
    train_scores = np.asarray(train_scores, dtype=np.float64)
    train_scores = train_scores[np.isfinite(train_scores)]

    if len(train_scores) == 0:
        raise ValueError("Empty or invalid train_scores")

    return np.percentile(train_scores, percentile)


# ==========================================================
# Supervised Threshold Search (Best F1)
# ==========================================================
def find_optimal_threshold(scores, labels, grain=1000, adjust=True):
    """
    Finds threshold maximizing F1 score.

    Improvements:
        - Uses quantile-based search (more stable than linear)
        - Handles NaNs safely
        - Consistent label binarization

    Args:
        scores: anomaly scores (higher = more anomalous)
        labels: ground truth (0 or 1)
        grain: number of candidate thresholds
        adjust: apply point-adjust or not

    Returns:
        best_threshold, best_f1
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    # Remove invalid values
    valid_mask = np.isfinite(scores)
    scores = scores[valid_mask]
    labels = labels[valid_mask]

    if len(scores) == 0:
        raise ValueError("No valid scores for thresholding")

    # Quantile-based threshold candidates (more robust)
    quantiles = np.linspace(0.0, 0.999, grain)
    thresholds = np.quantile(scores, quantiles)

    best_f1 = -1.0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        if adjust:
            predict, actual = point_adjust(scores, labels, threshold)
        else:
            predict = (scores >= threshold).astype(int)
            actual = labels.astype(int)

        f1, _, _, _, _, _, _ = calc_p2p(predict, actual)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


# ==========================================================
# Dynamic Threshold (Moving Average)
# ==========================================================
def apply_moving_average_threshold(scores, window_size=50, sigma=3):
    """
    Dynamic threshold using rolling statistics.

    Args:
        scores: anomaly scores
        window_size: smoothing window
        sigma: std multiplier

    Returns:
        threshold array (same length as scores)
    """
    scores = np.asarray(scores, dtype=np.float64)

    s = pd.Series(scores)

    rolling_mean = s.rolling(window=window_size, min_periods=1).mean()
    rolling_std = s.rolling(window=window_size, min_periods=1).std()

    rolling_std = rolling_std.fillna(0.0)

    threshold = rolling_mean + sigma * rolling_std

    return threshold.values
