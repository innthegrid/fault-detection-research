# src/data_utils/loaders.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from ..data_utils import scalers, time_utils


# ==========================================================
# Utility: Load all files from directory or file
# ==========================================================
def load_all_series(data_dir, data_name):
    """
    Supports:
    - Single file (csv/dat)
    - Directory of files
    Returns concatenated 1D numpy array
    """
    full_path = os.path.join(data_dir, data_name)

    if os.path.isfile(full_path):
        df = pd.read_csv(full_path, sep=",", header=None)
        return df.values.flatten().astype(float)

    elif os.path.isdir(full_path):
        all_values = []

        for root, _, files in os.walk(full_path):
            for f in files:
                if f.endswith(".dat") or f.endswith(".csv"):
                    df = pd.read_csv(os.path.join(root, f), sep=",", header=None)
                    all_values.append(df.values.flatten().astype(float))

        if len(all_values) == 0:
            raise ValueError(f"No valid data files found in directory: {full_path}")

        MAX_POINTS = 20000
        return np.concatenate(all_values)[:MAX_POINTS]

    else:
        raise ValueError(f"Invalid path: {full_path}")


# ==========================================================
# Dataset
# ==========================================================
class UnifiedAirDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_name,
        window_size,
        mode="train",
        hparams=None,
        stats=None,
    ):
        self.window = window_size
        self.hp = hparams
        self.mode = mode

        # ---------------------------
        # Load raw signal
        # ---------------------------
        values = load_all_series(data_dir, data_name)

        # ---------------------------
        # Missing mask (before filling)
        # ---------------------------
        missing = np.isnan(values).astype(np.float32)

        # ---------------------------
        # Labels (DEFAULT SAFE BEHAVIOR)
        # ---------------------------
        # Prefer explicit directory naming if available
        if "Healthy" in data_name:
            labels = np.zeros_like(values)
        elif "Fault" in data_name or "Bearing" in data_name:
            labels = np.ones_like(values)
        else:
            # fallback: unknown data treated as normal (safer for anomaly detection)
            labels = np.zeros_like(values)

        # ---------------------------
        # Fill missing values
        # ---------------------------
        values = pd.Series(values).bfill().ffill().fillna(0).values

        # ==========================================================
        # NORMALIZATION
        # ==========================================================
        if hasattr(self.hp, "model_type") and self.hp.model_type == "timegan":
            # Min-max scaling for GAN stability
            if stats is not None:
                t_min, t_max = stats
            else:
                t_min, t_max = values.min(), values.max()

            denom = (t_max - t_min) + 1e-7
            values = (values - t_min) / denom

            self.mean, self.std = t_min, t_max

        else:
            # FCVAE or others → standardized or scaled
            if self.hp.data_pre_mode == 0:
                values, self.mean, self.std = scalers.standardize_kpi(
                    values,
                    mean=stats[0] if stats is not None else None,
                    std=stats[1] if stats is not None else None,
                )
            else:
                if stats is not None:
                    t_min, t_max = stats
                else:
                    t_min, t_max = values.min(), values.max()

                denom = (t_max - t_min) + 1e-8
                values = 2 * (values - t_min) / denom - 1

                self.mean, self.std = t_min, t_max

            # stability clamp
            values = np.clip(values, -40, 40)

            # optional smoothing
            if (
                hasattr(self.hp, "sliding_window_size")
                and self.hp.sliding_window_size > 1
            ):
                sws = self.hp.sliding_window_size
                kernel = np.ones(sws) / sws

                values = np.convolve(values, kernel, mode="valid")
                labels = labels[sws - 1 :]
                missing = missing[sws - 1 :]

        # ==========================================================
        # Apply missing mask AFTER normalization
        # ==========================================================
        values = values.copy()
        values[missing == 1] = 0

        # ==========================================================
        # Disable labels during unsupervised training
        # ==========================================================
        if (
            mode in ["train", "valid"]
            and hasattr(self.hp, "use_label")
            and self.hp.use_label == 0
        ):
            labels[:] = 0

        # ==========================================================
        # Convert to (N, 1)
        # ==========================================================
        values = values.reshape(-1, 1)

        # ==========================================================
        # Windowing
        # ==========================================================
        self.x = time_utils.rolling_window(values, self.window, 1)
        self.y = time_utils.rolling_window(labels, self.window, 1)
        self.z = time_utils.rolling_window(missing, self.window, 1)

        # Safety check
        min_len = min(len(self.x), len(self.y), len(self.z))
        self.x = self.x[:min_len]
        self.y = self.y[:min_len]
        self.z = self.z[:min_len]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].copy(), dtype=torch.float32)
        y = torch.tensor(self.y[idx].copy(), dtype=torch.float32)
        z = torch.tensor(self.z[idx].copy(), dtype=torch.float32)

        # FCVAE expects [1, T]
        if hasattr(self.hp, "model_type") and self.hp.model_type == "fcvae":
            x = x.T

        return x, y, z


# ==========================================================
# DATALOADERS
# ==========================================================
def get_dataloaders(data_dir, train_dir, test_dir, hparams):
    """
    Fully generalized multi-fault pipeline.

    Supports:
    - train_dir: "compressor/Healthy/"
    - test_dir : "compressor/Bearing/" OR any fault folder
    """

    # ---------------------------
    # TRAIN (Healthy only)
    # ---------------------------
    train_ds = UnifiedAirDataset(
        data_dir,
        train_dir,
        hparams.window,
        mode="train",
        hparams=hparams,
    )

    stats = (train_ds.mean, train_ds.std)

    # ---------------------------
    # VALIDATION (Healthy only)
    # ---------------------------
    val_ds = UnifiedAirDataset(
        data_dir,
        train_dir,
        hparams.window,
        mode="valid",
        hparams=hparams,
        stats=stats,
    )

    # ---------------------------
    # TEST = healthy + all faults
    # ---------------------------
    test_healthy = UnifiedAirDataset(
        data_dir,
        train_dir,
        hparams.window,
        mode="test",
        hparams=hparams,
        stats=stats,
    )

    test_faulty = UnifiedAirDataset(
        data_dir,
        test_dir,
        hparams.window,
        mode="test",
        hparams=hparams,
        stats=stats,
    )

    test_ds = ConcatDataset([test_healthy, test_faulty])

    # ---------------------------
    # LOADERS
    # ---------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
    )

    return train_loader, val_loader, test_loader
