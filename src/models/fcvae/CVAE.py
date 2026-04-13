# src/models/fcvae/CVAE.py
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .Attention import EncoderLayer_selfattn


class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super().__init__()
        self.hp = hp
        self.num_iter = 0

        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.tensor([max_capacity], dtype=torch.float32)
        self.C_stop_iter = Capacity_max_iter

        # ======================
        # Encoder
        # ======================
        in_dim = 1 + 2 * hp.condition_emb_dim  # input feature dim + condition

        hidden_dims = [100, 100]
        modules = []
        for h in hidden_dims:
            modules.append(nn.Sequential(nn.Linear(in_dim, h), nn.Tanh()))
            in_dim = h

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], hp.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], hp.latent_dim), nn.Softplus()
        )

        # ======================
        # Decoder
        # ======================
        self.decoder_input = nn.Linear(
            hp.latent_dim + 2 * hp.condition_emb_dim,
            hidden_dims[-1],
        )

        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )

        modules.append(nn.Sequential(nn.Linear(hidden_dims[-1], 1)))
        self.decoder = nn.Sequential(*modules)

        self.fc_mu_x = nn.Identity()
        self.fc_var_x = nn.Sequential(nn.Linear(1, 1), nn.Softplus())

        # ======================
        # Attention (Local Features)
        # ======================
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    hp.d_model,
                    hp.d_inner,
                    hp.n_head,
                    hp.d_inner // hp.n_head,
                    hp.d_inner // hp.n_head,
                )
            ]
        )

        self.emb_local = nn.Sequential(
            nn.Linear(2 + hp.kernel_size, hp.d_model),
            nn.Tanh(),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(hp.d_model, hp.condition_emb_dim),
            nn.Tanh(),
        )

        self.emb_global = nn.Sequential(
            nn.Linear(hp.window, hp.condition_emb_dim),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(hp.dropout_rate)

    # ==========================================================
    # Encode / Decode
    # ==========================================================
    def encode(self, x):
        B, T, D = x.shape
        x = x.reshape(B * T, D)

        h = self.encoder(x)
        mu = self.fc_mu(h)
        var = self.fc_var(h) + 1e-6

        mu = mu.view(B, T, -1)
        var = var.view(B, T, -1)

        return mu, var

    def decode(self, z):
        B, T, D = z.shape
        z = z.reshape(B * T, D)

        h = self.decoder_input(z)
        x = self.decoder(h)

        mu_x = self.fc_mu_x(x)
        var_x = self.fc_var_x(x) + 1e-6

        mu_x = mu_x.view(B, T, -1)
        var_x = var_x.view(B, T, -1)

        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-6)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x, mode, y):
        cond = self.get_condition(x)
        cond = self.dropout(cond)

        enc_input = torch.cat([x, cond], dim=-1)

        mu, var = self.encode(enc_input)
        z = self.reparameterize(mu, var)

        dec_input = torch.cat([z, cond], dim=-1)
        mu_x, var_x = self.decode(dec_input)

        if mode in ["train", "valid"]:
            loss = self.loss_func(mu_x, var_x, x, mu, var)
            return mu_x, var_x, z, loss
        else:
            return self.MCMC2(x)

    # ==========================================================
    # Condition Extraction
    # ==========================================================
    def get_condition(self, x):
        # x: [B, T, 1]

        # -------- Global FFT --------
        f_global = torch.fft.rfft(x.squeeze(-1), dim=-1)
        f_global = torch.cat([f_global.real, f_global.imag], dim=-1)
        f_global = self.emb_global(f_global).unsqueeze(1)

        # -------- Local FFT --------
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            stride=(1, self.hp.stride),
        )

        x_unfold = unfold(x.transpose(1, 2).unsqueeze(2))
        x_unfold = x_unfold.transpose(1, 2)

        f_local = torch.fft.rfft(x_unfold, dim=-1)
        f_local = torch.cat([f_local.real, f_local.imag], dim=-1)
        f_local = self.emb_local(f_local)

        for attn in self.atten:
            f_local, _ = attn(f_local)

        f_local = self.out_linear(f_local[:, -1, :]).unsqueeze(1)

        cond = torch.cat([f_global, f_local], dim=-1)
        cond = cond.repeat(1, x.shape[1], 1)

        return cond

    # ==========================================================
    # MCMC Inference
    # ==========================================================
    def MCMC2(self, x):
        cond = self.get_condition(x)
        origin = x.clone()

        for _ in range(10):
            mu, var = self.encode(torch.cat([x, cond], dim=-1))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat([z, cond], dim=-1))

            recon = -0.5 * (torch.log(var_x) + (origin - mu_x) ** 2 / var_x)

            threshold = torch.quantile(
                recon, self.hp.mcmc_rate / 100.0, dim=1, keepdim=True
            )

            mask = (recon < threshold).float()
            x = mu_x * (1 - mask) + origin * mask

        # Monte Carlo estimate
        prob = 0
        mu, var = self.encode(torch.cat([x, cond], dim=-1))

        for _ in range(64):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat([z, cond], dim=-1))
            prob += -0.5 * (torch.log(var_x) + (origin - mu_x) ** 2 / var_x)

        return x, prob / 64

    # ==========================================================
    # Loss
    # ==========================================================
    def loss_func(self, mu_x, var_x, x, mu, var):
        # Reconstruction loss (Gaussian NLL)
        recon_loss = torch.mean(0.5 * (torch.log(var_x) + (x - mu_x) ** 2 / var_x))

        # KL Divergence (standard VAE)
        kld_loss = torch.mean(-0.5 * (1 + torch.log(var) - mu**2 - var))

        if self.loss_type == "B":
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter,
                0,
                self.C_max.item(),
            )
            return recon_loss + self.gamma * (kld_loss - C).abs()

        elif self.loss_type == "D":
            return recon_loss + (self.num_iter / 1000.0) * kld_loss

        return recon_loss + kld_loss
