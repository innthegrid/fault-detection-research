# src/data_utils/time_utils.py
import numpy as np


def complete_timestamp(timestamp, arrays=None):
    """
    Ensures uniform sampling by filling gaps with zeros.

    Args:
        timestamp: 1D array of timestamps
        arrays: list of aligned arrays (same length as timestamp)

    Returns:
        ret_ts: completed timestamps
        ret_missing: mask (1 = missing, 0 = observed)
        ret_arrays: aligned arrays with gaps filled
    """
    timestamp = np.asarray(timestamp, dtype=np.int64)

    if arrays is None:
        arrays = []
    arrays = [np.asarray(array) for array in arrays]

    # ---------------------------
    # Sort timestamps
    # ---------------------------
    src_index = np.argsort(timestamp)
    ts_sorted = timestamp[src_index]

    if len(ts_sorted) <= 1:
        # Edge case: not enough points
        ret_missing = np.zeros_like(ts_sorted, dtype=np.int32)
        return ts_sorted, ret_missing, arrays

    # ---------------------------
    # Infer base interval
    # ---------------------------
    diffs = np.diff(ts_sorted)
    intervals = np.unique(diffs)

    if len(intervals) == 0:
        interval = 1
    else:
        interval = np.min(intervals)

    # Validate intervals
    if any(itv % interval != 0 for itv in intervals):
        raise ValueError("Intervals are not multiples of the minimum interval")

    # ---------------------------
    # Build full timeline
    # ---------------------------
    start, end = ts_sorted[0], ts_sorted[-1]
    ret_ts = np.arange(start, end + interval, interval, dtype=np.int64)

    length = len(ret_ts)

    # Missing mask: default = missing
    ret_missing = np.ones(length, dtype=np.int32)

    # Initialize output arrays
    ret_arrays = [
        np.zeros((length,) + array.shape[1:], dtype=array.dtype) for array in arrays
    ]

    # ---------------------------
    # Map original data into new timeline
    # ---------------------------
    dst_index = ((ts_sorted - start) // interval).astype(np.int32)

    ret_missing[dst_index] = 0

    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    return ret_ts, ret_missing, ret_arrays


def rolling_window(data, window_size, step=1, transpose_for_cnn=False):
    """
    Universal windowing function for TimeGAN and FCVAE.

    Args:
        data: Input array (Length,) or (Length, Features)
        window_size: Length of each sequence
        step: Stride
        transpose_for_cnn: If True -> (N, Features, Time)
                           If False -> (N, Time, Features)

    Returns:
        windows: windowed data
    """
    data = np.asarray(data)

    # Ensure 2D input
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    length, num_features = data.shape

    # ---------------------------
    # Handle insufficient length
    # ---------------------------
    if length < window_size:
        return np.empty((0, window_size, num_features), dtype=data.dtype)

    # ---------------------------
    # Compute number of windows
    # ---------------------------
    n_windows = (length - window_size) // step + 1

    # ---------------------------
    # Efficient indexing
    # ---------------------------
    indexer = np.arange(window_size)[None, :] + step * np.arange(n_windows)[:, None]

    windows = data[indexer]  # (N, window_size, features)

    # ---------------------------
    # Optional transpose (FCVAE)
    # ---------------------------
    if transpose_for_cnn:
        windows = windows.transpose(0, 2, 1)

    return windows
