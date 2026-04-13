# src/evaluation/synthesis.py
# src/evaluation/synthesis.py
import numpy as np
import os

from .metrics.event_metrics import calculate_event_metrics
from .thresholds import find_optimal_threshold
from .visualization.signal_plots import plot_early_warning_trace
from .visualization.latent_plots import (
    project_latent_space,
    plot_latent_clusters,
    plot_degradation_trajectory,
)


def run_final_synthesis(
    model_name,
    scores,
    labels,
    latent_vectors,
    raw_signal,
    save_dir="./results",
):
    """
    End-to-end evaluation + visualization pipeline.

    Args:
        model_name (str): Name of model (e.g., "TimeGAN", "FCVAE")
        scores (np.ndarray): anomaly scores (higher = more anomalous)
        labels (np.ndarray): ground truth (0/1)
        latent_vectors (np.ndarray): latent embeddings
        raw_signal (np.ndarray): original signal
        save_dir (str): output directory

    Returns:
        dict: aggregated evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Starting Synthesis for {model_name} ---")

    # ==========================================================
    # Threshold Selection (Supervised)
    # ==========================================================
    threshold, best_f1 = find_optimal_threshold(scores, labels, adjust=True)
    print(f"Optimal Threshold: {threshold:.4f} (Best Adjusted F1: {best_f1:.4f})")

    # ==========================================================
    # Event-Level Metrics
    # ==========================================================
    metrics = calculate_event_metrics(scores, labels, threshold)
    metrics["best_f1"] = best_f1
    metrics["threshold"] = threshold

    # ==========================================================
    # Latent Projections
    # ==========================================================
    tsne_proj = None
    pca_proj = None

    if latent_vectors is not None and len(latent_vectors) > 0:
        try:
            tsne_proj = project_latent_space(latent_vectors, method="tsne")
            pca_proj = project_latent_space(latent_vectors, method="pca")
        except Exception as e:
            print(f"[Warning] Latent projection failed: {e}")

    # ==========================================================
    # Visualization
    # ==========================================================
    try:
        plot_early_warning_trace(
            raw_signal,
            scores,
            labels,
            threshold,
            model_name=model_name,
            save_path=os.path.join(save_dir, f"{model_name}_early_warning.png"),
        )
    except Exception as e:
        print(f"[Warning] Early warning plot failed: {e}")

    if tsne_proj is not None:
        try:
            plot_latent_clusters(
                tsne_proj,
                labels,
                model_name=model_name,
                save_path=os.path.join(save_dir, f"{model_name}_tsne_clusters.png"),
            )
        except Exception as e:
            print(f"[Warning] TSNE plot failed: {e}")

    if pca_proj is not None:
        try:
            traj_window = min(len(pca_proj), 3000)
            plot_degradation_trajectory(
                pca_proj[-traj_window:],
                np.arange(traj_window),
                save_path=os.path.join(save_dir, f"{model_name}_trajectory.png"),
            )
        except Exception as e:
            print(f"[Warning] Trajectory plot failed: {e}")

    print(f"--- Completed Synthesis for {model_name} ---")

    return metrics


if __name__ == "__main__":
    pass
