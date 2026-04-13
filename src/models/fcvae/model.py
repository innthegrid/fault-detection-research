# src/models/fcvae/model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

from .CVAE import CVAE
from ...data_utils import augmentations as data_augment
from ...evaluation.metrics.detection_utils import (
    best_f1,
    delay_f1,
    best_f1_without_pointadjust,
)


class MyVAE(LightningModule):
    """Frequency-enhanced CVAE for Anomaly Detection"""

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hp = hparams
        self.vae = CVAE(self.hp)

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x, mode, mask):
        """
        Input:
            FCVAE loader gives [B, 1, T]
        Convert to:
            [B, T, 1] (expected by CVAE)
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.transpose(1, 2)

        return self.vae(x, mode, mask)

    # ==========================================================
    # Loss
    # ==========================================================
    def compute_loss(self, x, y_all, z_all):
        # mask = valid (non-anomaly AND non-missing)
        mask = (~(y_all.bool() | z_all.bool())).float()

        _, _, _, _, _, loss = self.forward(x, "train", mask)
        return loss

    # ==========================================================
    # Training
    # ==========================================================
    def training_step(self, batch, batch_idx):
        x, y, z = batch
        x, y, z = self.batch_data_augmentation(x, y, z)

        loss = self.compute_loss(x, y, z)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        loss = self.compute_loss(x, y, z)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    # ==========================================================
    # Testing
    # ==========================================================
    def test_step(self, batch, batch_idx):
        x, y_all, z_all = batch
        y = y_all[:, -1].unsqueeze(1)

        with torch.no_grad():
            # MCMC inference (returns x_refined, likelihood)
            x_recon, prob = self.forward(x, "test", z_all)

            # reconstruction stats (train mode forward)
            mask = (~z_all.bool()).float()
            mu_x, var_x, _, _, _, _ = self.forward(x, "train", mask)

        # anomaly score = negative log-likelihood (last timestep)
        score = -prob[:, :, -1]

        return {
            "y": y.cpu(),
            "score": score.cpu(),
            "x": x[:, :, -1].cpu(),
            "mu_x": mu_x[:, :, -1].cpu(),
            "x_recon": x_recon[:, :, -1].cpu(),
            "var_x": var_x[:, :, -1].cpu(),
        }

    def test_epoch_end(self, outputs):
        y = torch.cat([o["y"] for o in outputs]).numpy().flatten()
        score = torch.cat([o["score"] for o in outputs]).numpy().flatten()

        x = torch.cat([o["x"] for o in outputs]).numpy()
        mu_x = torch.cat([o["mu_x"] for o in outputs]).numpy()
        x_recon = torch.cat([o["x_recon"] for o in outputs]).numpy()
        var_x = torch.cat([o["var_x"] for o in outputs]).numpy()

        os.makedirs("./results", exist_ok=True)

        np.save("./results/score.npy", score)
        np.save("./results/label.npy", y)

        # dataset-dependent delay window
        if "NAB" in self.hp.data_dir:
            k = 150
        elif "Yahoo" in self.hp.data_dir:
            k = 3
        else:
            k = 7

        # Safe AUC (avoid crash if only one class)
        try:
            auc = roc_auc_score(y, score)
        except:
            auc = float("nan")

        f1_d, prec_d, rec_d, pred_d = delay_f1(score, y, k)
        f1_b, prec_b, rec_b, pred_b = best_f1(score, y)
        f1_raw, prec_raw, rec_raw, pred_raw = best_f1_without_pointadjust(score, y)

        print("\n--- FCVAE Results ---")
        print(f"AUC: {auc:.4f}")
        print(f"Best F1: {f1_b:.4f}")
        print(f"Delay F1: {f1_d:.4f}")
        print(f"Raw F1: {f1_raw:.4f}")

        # Save CSV
        df = pd.DataFrame(
            {
                "x": x.flatten(),
                "mu_x": mu_x.flatten(),
                "recon": x_recon.flatten(),
                "var": var_x.flatten(),
                "label": y,
                "score": score,
                "pred_delay": pred_d,
                "pred_best": pred_b,
            }
        )

        df.to_csv("./results/fcvae_results.csv", index=False)

    # ==========================================================
    # Optimizer
    # ==========================================================
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sched]

    # ==========================================================
    # Data Augmentation
    # ==========================================================
    def batch_data_augmentation(self, x, y, z):
        if self.hp.point_ano_rate > 0:
            x_a, y_a, z_a = data_augment.point_ano(x, y, z, self.hp.point_ano_rate)
            x = torch.cat([x, x_a], dim=0)
            y = torch.cat([y, y_a], dim=0)
            z = torch.cat([z, z_a], dim=0)

        if self.hp.seg_ano_rate > 0:
            x_a, y_a, z_a = data_augment.seg_ano(x, y, z, self.hp.seg_ano_rate)
            x = torch.cat([x, x_a], dim=0)
            y = torch.cat([y, y_a], dim=0)
            z = torch.cat([z, z_a], dim=0)

        x, y, z = data_augment.missing_data_injection(
            x, y, z, self.hp.missing_data_rate
        )

        return x, y, z

    # ==========================================================
    # CLI Args
    # ==========================================================
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MyVAE")

        parser.add_argument("--window", type=int, default=64)
        parser.add_argument("--latent_dim", type=int, default=8)
        parser.add_argument("--learning_rate", type=float, default=5e-4)

        parser.add_argument("--missing_data_rate", type=float, default=0.01)
        parser.add_argument("--point_ano_rate", type=float, default=0.05)
        parser.add_argument("--seg_ano_rate", type=float, default=0.1)

        parser.add_argument("--d_model", type=int, default=256)
        parser.add_argument("--d_inner", type=int, default=512)
        parser.add_argument("--n_head", type=int, default=8)

        parser.add_argument("--use_label", type=int, default=0)

        return parent_parser
