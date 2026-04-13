# src/evaluation/metrics/fidelity.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error


# ==========================================================
# Utilities
# ==========================================================
def _sanitize_data(data):
    data = np.asarray(data, dtype=np.float32)
    mask = np.isfinite(data).all(axis=(1, 2))
    return data[mask]


def _get_batch_indices(n, batch_size):
    size = min(batch_size, n)
    return np.random.choice(n, size=size, replace=False)


# ==========================================================
# Model
# ==========================================================
class FidelityMonitor(nn.Module):
    """
    GRU-based evaluator for:
        - Discriminative score
        - Predictive score
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, mode="classify"):
        _, h_n = self.gru(x)
        last_state = h_n[-1]

        if mode == "classify":
            return torch.sigmoid(self.classifier(last_state))
        else:
            return self.regressor(last_state)


# ==========================================================
# Discriminative Score
# ==========================================================
def get_discriminative_score(
    real_data, fake_data, hidden_dim=None, iterations=2000, batch_size=128, seed=42
):
    """
    Can a model distinguish real vs fake?

    Score:
        |0.5 - accuracy| (lower is better)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    real_data = _sanitize_data(real_data)
    fake_data = _sanitize_data(fake_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = real_data.shape[2]
    hidden_dim = hidden_dim or max(1, dim // 2)

    model = FidelityMonitor(dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(iterations):
        idx_r = _get_batch_indices(len(real_data), batch_size)
        idx_f = _get_batch_indices(len(fake_data), batch_size)

        X_real = torch.FloatTensor(real_data[idx_r]).to(device)
        X_fake = torch.FloatTensor(fake_data[idx_f]).to(device)

        optimizer.zero_grad()

        y_real = model(X_real).view(-1, 1)
        y_fake = model(X_fake).view(-1, 1)

        loss_real = criterion(y_real, torch.ones_like(y_real))
        loss_fake = criterion(y_fake, torch.zeros_like(y_fake))

        loss = loss_real + loss_fake
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_real = model(torch.FloatTensor(real_data).to(device)).cpu().numpy()
        y_pred_fake = model(torch.FloatTensor(fake_data).to(device)).cpu().numpy()

    y_pred = np.concatenate([y_pred_real, y_pred_fake]).reshape(-1) > 0.5
    y_true = np.concatenate([np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))])

    acc = accuracy_score(y_true, y_pred)
    return float(abs(0.5 - acc))


# ==========================================================
# Predictive Score
# ==========================================================
def get_predictive_score(
    real_data, fake_data, iterations=5000, batch_size=128, seed=42
):
    """
    Train on synthetic → test on real.

    Score:
        MAE (lower is better)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    real_data = _sanitize_data(real_data)
    fake_data = _sanitize_data(fake_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len, dim = fake_data.shape[1], fake_data.shape[2]

    if dim < 2:
        raise ValueError("Predictive score requires at least 2 features")

    model = FidelityMonitor(dim - 1, max(1, dim // 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    model.train()
    for _ in range(iterations):
        idx = _get_batch_indices(len(fake_data), batch_size)

        # Inputs: all features except last
        batch_x = torch.FloatTensor(fake_data[idx, :-1, :-1]).to(device)

        # Target: next timestep of last feature
        batch_y = torch.FloatTensor(fake_data[idx, 1:, -1:]).to(device)

        optimizer.zero_grad()

        preds = model(batch_x, mode="regress")
        loss = criterion(preds, batch_y[:, -1, :])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.FloatTensor(real_data[:, :-1, :-1]).to(device)
        test_y = torch.FloatTensor(real_data[:, 1:, -1:]).to(device)

        preds = model(test_x, mode="regress").cpu().numpy()

    return float(mean_absolute_error(test_y[:, -1, :].cpu().numpy(), preds))
