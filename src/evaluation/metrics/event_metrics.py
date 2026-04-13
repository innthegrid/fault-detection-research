# src/evaluation/metrics/event_metrics.py
import numpy as np
from sklearn.metrics import average_precision_score


# ==========================================================
# Utilities
# ==========================================================
def _sanitize_inputs(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    mask = np.isfinite(scores)
    return scores[mask], labels[mask]


def get_event_segments(label):
    """
    Identifies contiguous anomaly segments.

    Returns:
        List of (start, end) indices
    """
    label = np.asarray(label).astype(int)

    if len(label) == 0:
        return []

    diff = np.diff(np.concatenate([[0], label, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list(zip(starts, ends))


# ==========================================================
# Event Metrics
# ==========================================================
def calculate_event_metrics(scores, labels, threshold):
    """
    Event-based evaluation metrics.

    Metrics:
        - Event Recall
        - Mean Time To Detect (MTTD)
        - AUPRC

    Args:
        scores: anomaly scores (higher = more anomalous)
        labels: binary ground truth (0/1)
        threshold: detection threshold

    Returns:
        dict of metrics
    """
    scores, labels = _sanitize_inputs(scores, labels)

    predictions = (scores >= threshold).astype(int)
    event_segments = get_event_segments(labels)

    if len(event_segments) == 0:
        return {
            "event_recall": 0.0,
            "mttd": 0.0,
            "auprc": 0.0,
            "total_events": 0,
            "detected_events": 0,
        }

    detected_events = 0
    delays = []

    for start, end in event_segments:
        segment_preds = predictions[start:end]

        if np.any(segment_preds):
            detected_events += 1

            # First detection index relative to segment start
            first_idx = np.argmax(segment_preds)
            delays.append(first_idx)
        else:
            # Missed event → penalize with full segment length
            delays.append(end - start)

    event_recall = detected_events / len(event_segments)
    mttd = float(np.mean(delays)) if delays else 0.0

    # AUPRC (safe)
    try:
        if np.sum(labels) == 0:
            auprc = 0.0
        else:
            auprc = float(average_precision_score(labels, scores))
    except Exception:
        auprc = 0.0

    return {
        "event_recall": float(event_recall),
        "mttd": mttd,
        "auprc": auprc,
        "total_events": int(len(event_segments)),
        "detected_events": int(detected_events),
    }
