# src/baselines/lstm_ae.py
import torch
import torch.nn as nn


# ==========================================================
# Model
# ==========================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=1, hidden_dim=64, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        """
        x: (B, T, F)
        """
        _, (h_n, _) = self.encoder(x)

        batch_size, seq_len, _ = x.shape

        # repeat latent state across time
        h = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder(h)
        recon = self.output_layer(dec_out)

        return recon


# ==========================================================
# Trainer
# ==========================================================
def train_lstm_ae(model, dataloader, epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, _, _ in dataloader:
            x = x.to(device)

            optimizer.zero_grad()

            recon = model(x)

            loss = loss_fn(recon, x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[LSTM-AE] Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")


# ==========================================================
# Scoring
# ==========================================================
def get_lstm_ae_scores(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    all_scores = []

    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            recon = model(x)

            # reconstruction error per window
            err = torch.mean((x - recon) ** 2, dim=(1, 2))

            all_scores.append(err.cpu())

    return torch.cat(all_scores).numpy()