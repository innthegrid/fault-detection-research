# src/baselines/run_isolation_forest.py
import os
import argparse
import numpy as np

from sklearn.ensemble import IsolationForest
from src.data_utils.loaders import get_dataloaders


# ==========================================================
# Train Isolation Forest
# ==========================================================
def train_iforest(train_x, contamination=0.01):
    n_samples = train_x.shape[0]

    print(f"[IFOREST] Training on {n_samples} samples...")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train_x)
    return model


# ==========================================================
# Scoring
# ==========================================================
def get_scores(model, test_x):
    scores = model.decision_function(test_x)
    return -scores  # higher = more anomalous


# ==========================================================
# Main
# ==========================================================
def main(args):

    print("\nLoading data...")

    train_loader, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        hparams=args,
    )

    # ----------------------------
    # Convert loaders → numpy
    # ----------------------------
    train_x = []
    for x, _, _ in train_loader:
        train_x.append(x.numpy())

    test_x = []
    test_y = []
    for x, y, _ in test_loader:
        test_x.append(x.numpy())
        test_y.append(y.numpy())

    train_x = np.concatenate(train_x, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    # ----------------------------
    # Flatten for IForest
    # ----------------------------
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

    print(f"Train shape: {train_x.shape}")
    print(f"Test shape: {test_x.shape}")

    # ----------------------------
    # Train + Predict
    # ----------------------------
    model = train_iforest(train_x, contamination=args.contamination)

    scores = get_scores(model, test_x)

    # ----------------------------
    # Save outputs
    # ----------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "iforest_scores.npy"), scores)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_y[:, -1])

    print("\nSaved:")
    print("- iforest_scores.npy")
    print("- test_labels.npy")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()
    main(args)