# src/run_inference.py
import os
import torch
import numpy as np
import argparse
import copy

from src.evaluation.scoring import UnifiedScorer
from src.models.fcvae.model import MyVAE
from src.models.timegan.model import TimeGAN
from src.data_utils.loaders import get_dataloaders


# ==========================================================
# Device
# ==========================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# ==========================================================
# Utility: safe model loading (prevents hparams mutation bugs)
# ==========================================================
def load_model(model_class, ckpt_path, base_args, model_type):
    cfg = copy.deepcopy(base_args)
    cfg.model_type = model_type
    model = model_class.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams=cfg,
    )
    return model.to(device)


# ==========================================================
# Main
# ==========================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading models...")

    # ==========================================================
    # Load models (safe, isolated configs)
    # ==========================================================
    timegan = load_model(TimeGAN, args.timegan_ckpt, args, "timegan")
    fcvae = load_model(MyVAE, args.fcvae_ckpt, args, "fcvae")

    timegan.eval()
    fcvae.eval()

    # ==========================================================
    # Load FULL dataset (healthy + ALL faults automatically)
    # ==========================================================
    print("Loading test dataset (healthy + faults)...")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        hparams=args,
    )

    # ==========================================================
    # Collect batches
    # ==========================================================
    all_x, all_y = [], []

    for batch in test_loader:
        x, y, _ = batch
        all_x.append(x)
        all_y.append(y)

    if len(all_x) == 0:
        raise ValueError("No data loaded from test_loader.")

    test_x = torch.cat(all_x, dim=0).to(device)
    test_y = torch.cat(all_y, dim=0).cpu().numpy()

    print(f"Test data shape (raw): {test_x.shape}")

    # ==========================================================
    # Shape normalization
    # ==========================================================
    if test_x.dim() != 3:
        raise ValueError(f"Unexpected input shape: {test_x.shape}")

    # Dataset convention:
    # FCVAE -> [B, 1, T]
    # TimeGAN -> [B, T, 1]

    if test_x.shape[1] == 1:
        x_fcvae = test_x
        x_timegan = test_x.transpose(1, 2)
    else:
        x_timegan = test_x
        x_fcvae = test_x.transpose(1, 2)

    # ==========================================================
    # Scoring
    # ==========================================================
    scorer = UnifiedScorer(timegan, fcvae, device=device)

    print("Running inference...")

    with torch.no_grad():
        tg_scores, tg_latents = scorer.get_timegan_results(x_timegan)
        vae_scores, vae_latents = scorer.get_fcvae_results(x_fcvae)

    # ==========================================================
    # Labels + raw signal extraction (fault-agnostic)
    # ==========================================================
    labels = test_y[:, -1]

    # robust last-step signal extraction
    if x_timegan.shape[-1] == 1:
        raw_signal = x_timegan[:, -1, 0].detach().cpu().numpy()
    else:
        raw_signal = x_timegan[:, -1].detach().cpu().numpy()

    # ==========================================================
    # Save outputs
    # ==========================================================
    np.save(os.path.join(args.output_dir, "timegan_scores.npy"), tg_scores)
    np.save(os.path.join(args.output_dir, "fcvae_scores.npy"), vae_scores)

    np.save(os.path.join(args.output_dir, "timegan_latents.npy"), tg_latents)
    np.save(os.path.join(args.output_dir, "fcvae_latents.npy"), vae_latents)

    np.save(os.path.join(args.output_dir, "test_labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "raw_signal_sample.npy"), raw_signal)

    print("\nSaved files:")
    print("- timegan_scores.npy")
    print("- fcvae_scores.npy")
    print("- timegan_latents.npy")
    print("- fcvae_latents.npy")
    print("- test_labels.npy")
    print("- raw_signal_sample.npy")

    print(f"\nResults saved to: {args.output_dir}")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Inference Pipeline")

    # ----------------------------
    # Data (NO hardcoded fault names anywhere)
    # ----------------------------
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    # ----------------------------
    # Checkpoints
    # ----------------------------
    parser.add_argument("--timegan_ckpt", type=str, required=True)
    parser.add_argument("--fcvae_ckpt", type=str, required=True)

    # ----------------------------
    # Loader params
    # ----------------------------
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # ----------------------------
    # FCVAE-specific params
    # ----------------------------
    parser.add_argument("--data_pre_mode", type=int, default=0)
    parser.add_argument("--sliding_window_size", type=int, default=1)
    parser.add_argument("--use_label", type=int, default=0)

    # ----------------------------
    # Output
    # ----------------------------
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()

    main(args)
