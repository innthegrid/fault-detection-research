# src/evaluation/visualization/signal_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.5)


def plot_early_warning_trace(
    raw_signal,
    scores,
    labels,
    threshold,
    model_name="Model",
    save_path=None,
):
    """
    3-tier visualization:
    (1) Raw signal
    (2) Anomaly score + threshold
    (3) Ground truth vs predictions with lead-time annotation
    """
    raw_signal = np.asarray(raw_signal).flatten()
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    length = min(len(raw_signal), len(scores), len(labels))
    raw_signal = raw_signal[:length]
    scores = scores[:length]
    labels = labels[:length]

    time = np.arange(length)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 1]},
    )

    # ==========================================================
    # Raw Signal
    # ==========================================================
    axes[0].plot(
        time,
        raw_signal,
        color="gray",
        alpha=0.6,
        label="Sensor Signal",
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Early Warning Detection Trace: {model_name}", pad=20)
    axes[0].legend(loc="upper right")

    # ==========================================================
    # Scores
    # ==========================================================
    axes[1].plot(time, scores, linewidth=1.5, label="Anomaly Score")
    axes[1].axhline(
        y=threshold,
        linestyle="--",
        label="Threshold",
        alpha=0.8,
    )
    axes[1].fill_between(time, 0, scores, alpha=0.15)
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper right")

    # ==========================================================
    # Labels + Predictions
    # ==========================================================
    preds = (scores >= threshold).astype(int)

    axes[2].fill_between(
        time,
        0,
        1,
        where=(labels > 0.5),
        alpha=0.3,
        label="Ground Truth Fault",
    )

    axes[2].step(
        time,
        preds,
        where="post",
        linewidth=2,
        label="Predicted Alarm",
    )

    # ==========================================================
    # Lead Time Annotation (first detected event)
    # ==========================================================
    actual_diff = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    actual_starts = np.where(actual_diff == 1)[0]

    pred_diff = np.diff(np.concatenate([[0], preds, [0]]))
    pred_starts = np.where(pred_diff == 1)[0]

    if len(actual_starts) > 0 and len(pred_starts) > 0:
        first_fault = actual_starts[0]

        valid_preds = pred_starts[pred_starts <= first_fault + 10]

        if len(valid_preds) > 0:
            first_alarm = valid_preds[0]

            axes[2].annotate(
                "",
                xy=(first_alarm, 0.5),
                xytext=(first_fault, 0.5),
                arrowprops=dict(arrowstyle="<->", lw=1.5),
            )

            axes[2].text(
                (first_alarm + first_fault) / 2,
                0.6,
                "Lead Time",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Healthy", "Fault"])
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_reconstruction_comparison(original, reconstruction, save_path=None):
    """
    Overlay: original vs reconstructed signal with error shading.
    """
    original = np.asarray(original).flatten()
    reconstruction = np.asarray(reconstruction).flatten()

    length = min(len(original), len(reconstruction))
    original = original[:length]
    reconstruction = reconstruction[:length]

    x = np.arange(length)

    plt.figure(figsize=(10, 4))
    plt.plot(x, original, label="Original", alpha=0.6, linewidth=1.5)
    plt.plot(x, reconstruction, linestyle="--", linewidth=1.5, label="Reconstruction")

    plt.fill_between(
        x,
        original,
        reconstruction,
        alpha=0.2,
        label="Reconstruction Error",
    )

    plt.title("Reconstruction Comparison")
    plt.ylabel("Value")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
