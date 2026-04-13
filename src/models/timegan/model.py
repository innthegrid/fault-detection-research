# src/models/timegan/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .timegan import Embedder, Recovery, Generator, Supervisor, Discriminator


class TimeGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.automatic_optimization = False

        self.embedder = Embedder(
            hparams.dim, hparams.hidden_dim, hparams.num_layer, hparams.module
        )
        self.recovery = Recovery(
            hparams.hidden_dim, hparams.dim, hparams.num_layer, hparams.module
        )
        self.generator = Generator(
            hparams.dim, hparams.hidden_dim, hparams.num_layer, hparams.module
        )
        self.supervisor = Supervisor(
            hparams.hidden_dim, hparams.num_layer, hparams.module
        )
        self.discriminator = Discriminator(
            hparams.hidden_dim, hparams.num_layer, hparams.module
        )

    # -----------------------------
    # Utility Loss
    # -----------------------------
    def _get_moment_loss(self, x, x_hat):
        mean_x = torch.mean(x, dim=0)
        mean_x_hat = torch.mean(x_hat, dim=0)

        std_x = torch.sqrt(torch.var(x, dim=0) + 1e-6)
        std_x_hat = torch.sqrt(torch.var(x_hat, dim=0) + 1e-6)

        loss_v1 = torch.mean(torch.abs(mean_x - mean_x_hat))
        loss_v2 = torch.mean(torch.abs(std_x - std_x_hat))
        return loss_v1 + loss_v2

    # -----------------------------
    # Training Step Dispatcher
    # -----------------------------
    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        if self.current_epoch < self.hparams.pretrain_epochs:
            self._train_embedder(x)

        elif self.current_epoch < (self.hparams.pretrain_epochs * 2):
            self._train_supervisor(x)

        else:
            self._train_joint(x)

    # -----------------------------
    # Phase 1: Embedder
    # -----------------------------
    def _train_embedder(self, x):
        e_opt = self.optimizers()[0]
        e_opt.zero_grad()

        h = self.embedder(x)
        x_tilde = self.recovery(h)

        loss_recon = nn.functional.mse_loss(x, x_tilde)
        loss_e = 10 * torch.sqrt(loss_recon + 1e-7)

        self.manual_backward(loss_e)
        e_opt.step()

        self.log("e_loss", loss_e, prog_bar=True)

    # -----------------------------
    # Phase 2: Supervisor
    # -----------------------------
    def _train_supervisor(self, x):
        _, g_opt, _ = self.optimizers()
        g_opt.zero_grad()

        with torch.no_grad():
            h = self.embedder(x)

        h_hat = self.supervisor(h)

        loss_s = nn.functional.mse_loss(h[:, 1:, :], h_hat[:, :-1, :])
        loss_gs = torch.sqrt(loss_s + 1e-7)

        self.manual_backward(loss_gs)
        g_opt.step()

        self.log("s_loss", loss_gs, prog_bar=True)

    # -----------------------------
    # Phase 3: Joint Training
    # -----------------------------
    def _train_joint(self, x):
        e_opt, g_opt, d_opt = self.optimizers()
        gamma = 1.0

        # -------------------------
        # Generator + Embedder
        # -------------------------
        for _ in range(2):
            g_opt.zero_grad()

            z = torch.rand(x.size(0), x.size(1), self.hparams.dim, device=self.device)

            h = self.embedder(x)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            h_hat_sup = self.supervisor(h)
            x_hat = self.recovery(h_hat)

            # Discriminator outputs
            y_fake = self.discriminator(h_hat)
            y_fake_e = self.discriminator(e_hat)

            # --- Generator losses ---
            g_loss_u = nn.functional.binary_cross_entropy_with_logits(
                y_fake, torch.ones_like(y_fake)
            )
            g_loss_u_e = nn.functional.binary_cross_entropy_with_logits(
                y_fake_e, torch.ones_like(y_fake_e)
            )

            g_loss_s = nn.functional.mse_loss(h[:, 1:, :], h_hat_sup[:, :-1, :])

            g_loss_v = self._get_moment_loss(x, x_hat)

            g_loss = (
                g_loss_u
                + gamma * g_loss_u_e
                + 100 * torch.sqrt(g_loss_s + 1e-7)
                + 100 * g_loss_v
            )

            self.manual_backward(g_loss)
            g_opt.step()

            # ---------------------
            # Embedder update
            # ---------------------
            e_opt.zero_grad()

            h = self.embedder(x)
            x_tilde = self.recovery(h)

            e_loss_t0 = nn.functional.mse_loss(x, x_tilde)

            h_hat_sup = self.supervisor(h)
            g_loss_s = nn.functional.mse_loss(h[:, 1:, :], h_hat_sup[:, :-1, :])

            e_loss = 10 * torch.sqrt(e_loss_t0 + 1e-7) + 0.1 * g_loss_s

            self.manual_backward(e_loss)
            e_opt.step()

        # -------------------------
        # Discriminator
        # -------------------------
        d_opt.zero_grad()

        z = torch.rand(x.size(0), x.size(1), self.hparams.dim, device=self.device)

        with torch.no_grad():
            h_real = self.embedder(x)
            e_hat = self.generator(z)
            h_fake = self.supervisor(e_hat)

        y_real = self.discriminator(h_real)
        y_fake = self.discriminator(h_fake)
        y_fake_e = self.discriminator(e_hat)

        d_loss_real = nn.functional.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real)
        )
        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
            y_fake, torch.zeros_like(y_fake)
        )
        d_loss_fake_e = nn.functional.binary_cross_entropy_with_logits(
            y_fake_e, torch.zeros_like(y_fake_e)
        )

        d_loss = d_loss_real + d_loss_fake + gamma * d_loss_fake_e

        if d_loss.item() > 0.15:
            self.manual_backward(d_loss)
            d_opt.step()

        self.log_dict(
            {
                "g_loss": g_loss.detach(),
                "d_loss": d_loss.detach(),
                "e_loss_joint": e_loss.detach(),
            },
            prog_bar=True,
        )

    # -----------------------------
    # Optimizers
    # -----------------------------
    def configure_optimizers(self):
        e_vars = list(self.embedder.parameters()) + list(self.recovery.parameters())
        g_vars = list(self.generator.parameters()) + list(self.supervisor.parameters())
        d_vars = self.discriminator.parameters()

        return [
            torch.optim.Adam(e_vars, lr=self.hparams.lr),
            torch.optim.Adam(g_vars, lr=self.hparams.lr),
            torch.optim.Adam(d_vars, lr=self.hparams.lr),
        ]

    # -----------------------------
    # Testing
    # -----------------------------
    def test_step(self, batch, batch_idx):
        x, y_all, _ = batch

        y = (y_all[:, -1]).unsqueeze(1)

        with torch.no_grad():
            h = self.embedder(x)
            x_tilde = self.recovery(h)

            rec_score = torch.mean((x - x_tilde) ** 2, dim=-1)

            y_pred = torch.sigmoid(self.discriminator(h))
            disc_score = 1 - y_pred.squeeze(-1)

            combined_score = rec_score + disc_score

        return {
            "y": y.cpu(),
            "score": combined_score.cpu(),
        }

    def test_epoch_end(self, outputs):
        from ...evaluation.metrics.detection_utils import (
            best_f1,
            delay_f1,
            best_f1_without_pointadjust,
        )

        y = torch.cat([o["y"] for o in outputs]).numpy().flatten()
        score = torch.cat([o["score"] for o in outputs]).numpy()
        score = score[:, -1]

        k = 150 if "NAB" in self.hparams.data_dir else 7

        f1_d, _, _, _ = delay_f1(score, y, k)
        f1_b, _, _, _ = best_f1(score, y)
        f1_raw, _, _, _ = best_f1_without_pointadjust(score, y)

        print("\n--- TimeGAN Results ---")
        print(f"Best F1: {f1_b:.4f} | Delay F1: {f1_d:.4f} | Raw F1: {f1_raw:.4f}")
