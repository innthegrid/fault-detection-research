# src/baselines/run_lstm_ae.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils.loaders import get_dataloaders


# ==========================================================
# Model
# ==========================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=1, hidden_dim=64):
        super().__init__()

        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.out = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)

        b, t, _ = x.shape
        h_rep = h.permute(1, 0, 2).repeat(1, t, 1)

        dec_out, _ = self.decoder(h_rep)
        return self.out(dec_out)


# ==========================================================
# Train
# ==========================================================
def train(model, loader, device, epochs=20, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0

        for x, _, _ in loader:
            x = x.to(device).float()

            recon = model(x)
            loss = loss_fn(recon, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {ep+1}/{epochs} | Loss: {total/len(loader):.6f}")


# ==========================================================
# Scoring
# ==========================================================
def score(model, loader, device):
    model.eval()
    scores = []

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device).float()
            recon = model(x)

            mse = ((x - recon) ** 2).mean(dim=(1, 2))
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores)


# ==========================================================
# Main
# ==========================================================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")

    train_loader, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        hparams=args,
    )

    model = LSTMAutoencoder(n_features=1, hidden_dim=args.hidden_dim).to(device)

    print("\nTraining LSTM AE...")
    train(model, train_loader, device, epochs=args.epochs, lr=args.lr)

    print("\nScoring...")
    scores = score(model, test_loader, device)

    # labels
    labels = []
    for _, y, _ in test_loader:
        labels.append(y.numpy())
    labels = np.concatenate(labels, axis=0)[:, -1]

    # save
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "lstm_ae_scores.npy"), scores)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), labels)

    print("\nSaved:")
    print("- lstm_ae_scores.npy")
    print("- test_labels.npy")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()
    main(args)