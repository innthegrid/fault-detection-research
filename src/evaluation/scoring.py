# src/evaluation/scoring.py
import torch
import numpy as np


class UnifiedScorer:
    """
    Unified inference wrapper for TimeGAN and FCVAE.

    Produces:
        - anomaly scores
        - latent representations (for visualization / clustering)
    """

    def __init__(self, timegan_model=None, fcvae_model=None, device="cpu"):
        self.timegan = timegan_model
        self.fcvae = fcvae_model
        self.device = device

        if self.timegan:
            self.timegan.to(device)
        if self.fcvae:
            self.fcvae.to(device)

    # ==========================================================
    # TimeGAN
    # ==========================================================
    def get_timegan_results(self, x, alpha=0.5, batch_size=64):
        """
        Args:
            x: numpy array [N, T, F]
        Returns:
            scores: anomaly scores
            latents: latent embeddings
        """
        assert self.timegan is not None, "TimeGAN model not provided"

        self.timegan.eval()

        all_scores = []
        all_latents = []

        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.FloatTensor(x[i : i + batch_size]).to(self.device)

                # ---- Forward ----
                h = self.timegan.embedder(batch)
                x_tilde = self.timegan.recovery(h)

                # ---- Reconstruction error ----
                recon_error = torch.mean((batch - x_tilde) ** 2, dim=(1, 2))

                # ---- Discriminator score ----
                d_logit = self.timegan.discriminator(h)
                disc_score = 1.0 - torch.sigmoid(d_logit).mean(dim=1).squeeze(-1)

                # ---- Hybrid score ----
                hybrid_score = alpha * recon_error + (1 - alpha) * disc_score

                all_scores.append(hybrid_score.cpu().numpy())
                all_latents.append(torch.mean(h, dim=1).cpu().numpy())

        return np.concatenate(all_scores), np.concatenate(all_latents)

    # ==========================================================
    # FCVAE
    # ==========================================================
    def get_fcvae_results(self, x, z=None, batch_size=64):
        """
        Args:
            x: numpy array [N, T, F] OR [N, 1, T]
            z: missing mask (optional) same shape as x

        Returns:
            scores: anomaly scores
            latents: latent embeddings (mu)
        """
        assert self.fcvae is not None, "FCVAE model not provided"

        self.fcvae.eval()

        all_scores = []
        all_latents = []

        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.FloatTensor(x[i : i + batch_size]).to(self.device)

                # Ensure FCVAE shape: [B, 1, T]
                if batch.dim() == 3 and batch.shape[-1] != 1:
                    batch = batch.permute(0, 2, 1)  # [B, F, T]
                if batch.dim() == 3 and batch.shape[1] != 1:
                    batch = batch[:, :1, :]  # enforce single channel

                # Missing mask
                if z is not None:
                    z_batch = torch.FloatTensor(z[i : i + batch_size]).to(self.device)
                else:
                    z_batch = torch.zeros_like(batch)

                # ---- MCMC inference (test mode) ----
                x_recon, prob = self.fcvae(batch, mode="test", mask=z_batch)

                # ---- Train-mode forward for latent ----
                mu_x, var_x, _, mu, var, _ = self.fcvae(
                    batch, mode="train", mask=(~z_batch.bool()).float()
                )

                # ---- Score ----
                score = -prob[:, :, -1]

                all_scores.append(score.cpu().numpy())
                all_latents.append(mu.cpu().numpy())

        return np.concatenate(all_scores), np.concatenate(all_latents)
