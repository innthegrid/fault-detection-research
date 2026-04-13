# src/evaluation/metrics/detection_utils.py
import numpy as np


# ==========================================================
# Utilities
# ==========================================================
def _sanitize_inputs(score, label):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(int)

    mask = np.isfinite(score)
    return score[mask], label[mask]


def _get_segments(label):
    """
    Returns start/end indices of anomaly segments.
    """
    label = label.astype(int)
    diff = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts, ends


# ==========================================================
# Point Adjust
# ==========================================================
def point_adjust(score, label, threshold):
    """
    Segment-level detection:
    If any point in a segment is detected, entire segment is detected.
    """
    score, label = _sanitize_inputs(score, label)

    predict = (score >= threshold).astype(int)
    actual = label.astype(int)

    starts, ends = _get_segments(actual)

    for start, end in zip(starts, ends):
        if np.any(predict[start:end]):
            predict[start:end] = 1

    return predict, actual


# ==========================================================
# Metrics
# ==========================================================
def calc_p2p(predict, actual):
    """
    Precision / Recall / F1
    """
    predict = predict.astype(int)
    actual = actual.astype(int)

    tp = np.sum(predict * actual)
    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))
    fn = np.sum((1 - predict) * actual)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return f1, precision, recall, tp, tn, fp, fn


# ==========================================================
# Threshold Sweeps
# ==========================================================
def _get_thresholds(score, num=200):
    """
    Robust threshold candidates using quantiles.
    """
    quantiles = np.linspace(0.0, 0.999, num)
    return np.quantile(score, quantiles)


def best_f1_without_pointadjust(score, label):
    """
    Standard point-wise F1
    """
    score, label = _sanitize_inputs(score, label)

    thresholds = _get_thresholds(score)

    best_f1, best_p, best_r, best_pred = 0, 0, 0, None

    for t in thresholds:
        pred = (score >= t).astype(int)
        f1, p, r, _, _, _, _ = calc_p2p(pred, label)

        if f1 > best_f1:
            best_f1, best_p, best_r, best_pred = f1, p, r, pred

    return best_f1, best_p, best_r, best_pred


def best_f1(score, label):
    """
    Point-adjusted F1
    """
    score, label = _sanitize_inputs(score, label)

    thresholds = _get_thresholds(score)

    best_f1, best_p, best_r, best_pred = 0, 0, 0, None

    for t in thresholds:
        pred, actual = point_adjust(score, label, t)
        f1, p, r, _, _, _, _ = calc_p2p(pred, actual)

        if f1 > best_f1:
            best_f1, best_p, best_r, best_pred = f1, p, r, pred

    return best_f1, best_p, best_r, best_pred


# ==========================================================
# Delay F1
# ==========================================================
def delay_f1(score, label, k=7):
    """
    Early detection metric:
    A segment is detected only if detected within k steps of onset.
    """
    score, label = _sanitize_inputs(score, label)

    thresholds = _get_thresholds(score)
    starts, ends = _get_segments(label)

    best_f1, best_p, best_r, best_pred = 0, 0, 0, None

    for t in thresholds:
        raw_pred = (score >= t).astype(int)
        delay_pred = np.zeros_like(raw_pred)

        for start, end in zip(starts, ends):
            window_end = min(start + k, end)

            if np.any(raw_pred[start:window_end]):
                delay_pred[start:end] = 1

        f1, p, r, _, _, _, _ = calc_p2p(delay_pred, label)

        if f1 > best_f1:
            best_f1, best_p, best_r, best_pred = f1, p, r, delay_pred

    return best_f1, best_p, best_r, best_pred
