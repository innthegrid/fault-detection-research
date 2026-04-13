# src/evaluation/visualization/latent_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.5)


# ==========================================================
# Projection
# ==========================================================
def project_latent_space(latent_vectors, method="tsne", perplexity=30):
    """
    Projects latent vectors to 2D.

    Args:
        latent_vectors: (N, D)
        method: 'tsne' or 'pca'
    """
    latent_vectors = np.asarray(latent_vectors)

    if latent_vectors.ndim != 2 or len(latent_vectors) == 0:
        raise ValueError("latent_vectors must be (N, D) and non-empty")

    n_samples, n_dim = latent_vectors.shape

    # scale first (important for both PCA and TSNE)
    vectors_scaled = StandardScaler().fit_transform(latent_vectors)

    if method == "pca":
        projector = PCA(n_components=2)

    elif method == "tsne":
        # FIX: remove invalid n_jobs, adapt perplexity
        safe_perplexity = min(perplexity, max(5, n_samples // 3))

        projector = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        )
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    print(f"Projecting {n_samples}x{n_dim} → 2D using {method.upper()}")

    return projector.fit_transform(vectors_scaled)


# ==========================================================
# Cluster Plot
# ==========================================================
def plot_latent_clusters(
    projected_vectors,
    labels,
    method="t-SNE",
    model_name="Model",
    save_path=None,
):
    projected_vectors = np.asarray(projected_vectors)
    labels = np.asarray(labels).flatten()

    n = min(len(projected_vectors), len(labels))
    projected_vectors = projected_vectors[:n]
    labels = labels[:n]

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=projected_vectors[:, 0],
        y=projected_vectors[:, 1],
        hue=labels,
        alpha=0.6,
        s=50,
        edgecolor="white",
        linewidth=0.3,
    )

    plt.title(f"{method} Projection: {model_name} Latent Space")
    plt.xlabel(f"{method} Dim 1")
    plt.ylabel(f"{method} Dim 2")

    plt.legend(title="State", loc="best")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()


# ==========================================================
# Trajectory Plot
# ==========================================================
def plot_degradation_trajectory(projected_vectors, timestamp, save_path=None):
    """
    Shows temporal trajectory in latent space.
    """
    projected_vectors = np.asarray(projected_vectors)
    timestamp = np.asarray(timestamp)

    n = min(len(projected_vectors), len(timestamp))
    projected_vectors = projected_vectors[:n]
    timestamp = timestamp[:n]

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        projected_vectors[:, 0],
        projected_vectors[:, 1],
        c=timestamp,
        s=50,
        alpha=0.8,
    )

    # arrows (downsampled for clarity)
    step = max(1, n // 20)

    for i in range(0, n - step, step):
        dx = projected_vectors[i + step, 0] - projected_vectors[i, 0]
        dy = projected_vectors[i + step, 1] - projected_vectors[i, 1]

        plt.arrow(
            projected_vectors[i, 0],
            projected_vectors[i, 1],
            dx,
            dy,
            alpha=0.3,
            width=0.005,
            head_width=0.05,
            head_length=0.05,
        )

    plt.title("Latent Space Degradation Trajectory")
    plt.colorbar(scatter, label="Time")

    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
