# src/evaluate_results.py
import numpy as np
import os
import argparse

from src.evaluation.synthesis import run_final_synthesis


# ==========================================================
# Utilities
# ==========================================================
def safe_load(path, name):
    if not os.path.exists(path):
        print(f"Warning: Missing {name} at {path}")
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None


def validate_shapes(scores, labels, latents, model_name):
    if len(scores) != len(labels):
        raise ValueError(
            f"{model_name}: Score-label length mismatch "
            f"({len(scores)} vs {len(labels)})"
        )

    if latents is None:
        raise ValueError(f"{model_name}: Missing latent vectors.")

    if len(latents) != len(scores):
        raise ValueError(
            f"{model_name}: Latent-score length mismatch "
            f"({len(latents)} vs {len(scores)})"
        )


def ensure_1d(x):
    """
    Forces evaluation inputs into consistent 1D format
    (handles cases where inference outputs are (N,1))
    """
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim > 1:
        return x.reshape(len(x), -1)
    return x


# ==========================================================
# Main
# ==========================================================
def main(args):
    results_dir = args.results_dir
    poster_output = os.path.join(results_dir, "poster_plots")
    os.makedirs(poster_output, exist_ok=True)

    print("Loading shared test metadata...")

    labels = safe_load(os.path.join(results_dir, "test_labels.npy"), "labels")
    raw_signal = safe_load(
        os.path.join(results_dir, "raw_signal_sample.npy"), "raw signal"
    )

    if labels is None or raw_signal is None:
        print("Error: Required metadata missing. Run run_inference.py first.")
        return

    labels = ensure_1d(labels)
    raw_signal = ensure_1d(raw_signal)

    model_results = {}

    # ==========================================================
    # Evaluate models
    # ==========================================================
    for m_name in args.models:
        print(f"\n--- Evaluating {m_name.upper()} ---")

        scores = safe_load(
            os.path.join(results_dir, f"{m_name}_scores.npy"),
            f"{m_name} scores",
        )

        latents = safe_load(
            os.path.join(results_dir, f"{m_name}_latents.npy"),
            f"{m_name} latents",
        )

        if scores is None:
            print(f"Skipping {m_name.upper()} due to missing scores.")
            continue

        scores = ensure_1d(scores)
        latents = ensure_1d(latents)

        try:
            validate_shapes(scores, labels, latents, m_name.upper())

            metrics = run_final_synthesis(
                model_name=m_name.upper(),
                scores=scores,
                labels=labels,
                latent_vectors=latents,
                raw_signal=raw_signal,
                save_dir=poster_output,
            )

            model_results[m_name.upper()] = metrics

        except Exception as e:
            print(f"Error evaluating {m_name.upper()}: {e}")

    # ==========================================================
    # Summary Table
    # ==========================================================
    if model_results:
        print("\n" + "=" * 60)
        print(f"{'MODEL':<12} | {'F1 (ADJ)':<10} | {'AUPRC':<10} | {'MTTD':<10}")
        print("-" * 60)

        for name, m in model_results.items():
            f1 = m.get("best_f1", 0.0)
            auprc = m.get("auprc", 0.0)
            mttd = m.get("mttd", 0.0)

            print(f"{name:<12} | {f1:<10.4f} | {auprc:<10.4f} | {mttd:<10.2f}")

        print("=" * 60)
    else:
        print("No models were successfully evaluated.")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FCVAE + TimeGAN Results")

    # ----------------------------
    # Output directory (NO HARD CODE)
    # ----------------------------
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory containing inference outputs",
    )

    # ----------------------------
    # Models to evaluate (fully flexible)
    # ----------------------------
    parser.add_argument(
        "--models",
        nargs="+",
        default=["timegan", "fcvae"],
        help="List of models to evaluate",
    )

    args = parser.parse_args()
    main(args)
