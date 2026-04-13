# src/models/timegan/timegan.py
import torch
import torch.nn as nn


# -------------------------------------------------
# Base RNN Block
# -------------------------------------------------
class TimeGAN_SubNetwork(nn.Module):
    """
    Base class for TimeGAN components:
    Handles GRU/LSTM + FC + activation
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        module_name="gru",
        activation=nn.Sigmoid(),
    ):
        super().__init__()

        self.module_name = module_name.lower()

        if self.module_name == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers, batch_first=True
            )
        elif self.module_name == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers, batch_first=True
            )
        else:
            raise ValueError(f"Unsupported module: {module_name}")

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


# -------------------------------------------------
# Embedder (X → H)
# -------------------------------------------------
class Embedder(TimeGAN_SubNetwork):
    def __init__(self, input_dim, hidden_dim, num_layers, module_name):
        super().__init__(
            input_dim, hidden_dim, num_layers, module_name, activation=nn.Sigmoid()
        )


# -------------------------------------------------
# Recovery (H → X)
# -------------------------------------------------
class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, module_name):
        super().__init__()

        module_name = module_name.lower()

        if module_name == "gru":
            self.rnn = nn.GRU(
                hidden_dim, hidden_dim, num_layers, batch_first=True
            )
        elif module_name == "lstm":
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, batch_first=True
            )
        else:
            raise ValueError(f"Unsupported module: {module_name}")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, h):
        out, _ = self.rnn(h)
        out = self.fc(out)
        return self.activation(out)


# -------------------------------------------------
# Generator (Z → E)
# -------------------------------------------------
class Generator(TimeGAN_SubNetwork):
    def __init__(self, z_dim, hidden_dim, num_layers, module_name):
        super().__init__(
            z_dim, hidden_dim, num_layers, module_name, activation=nn.Sigmoid()
        )


# -------------------------------------------------
# Supervisor (H → H_next)
# -------------------------------------------------
class Supervisor(TimeGAN_SubNetwork):
    def __init__(self, hidden_dim, num_layers, module_name):
        super().__init__(
            hidden_dim,
            hidden_dim,
            max(1, num_layers - 1),
            module_name,
            activation=nn.Sigmoid(),
        )


# -------------------------------------------------
# Discriminator (H → Real/Fake)
# -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers, module_name):
        super().__init__()

        module_name = module_name.lower()

        if module_name == "gru":
            self.rnn = nn.GRU(
                hidden_dim, hidden_dim, num_layers, batch_first=True
            )
        elif module_name == "lstm":
            self.rnn = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, batch_first=True
            )
        else:
            raise ValueError(f"Unsupported module: {module_name}")

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        out, _ = self.rnn(h)
        logits = self.fc(out)  # [B, T, 1]
        return logits


# -------------------------------------------------
# Full TimeGAN Wrapper (CRITICAL ADDITION)
# -------------------------------------------------
class TimeGAN(nn.Module):
    """
    Full TimeGAN architecture wrapper for PyTorch.

    Combines:
    - Embedder
    - Recovery
    - Generator
    - Supervisor
    - Discriminator
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        z_dim=None,
        module_name="gru",
    ):
        super().__init__()

        if z_dim is None:
            z_dim = input_dim

        self.embedder = Embedder(
            input_dim, hidden_dim, num_layers, module_name
        )
        self.recovery = Recovery(
            hidden_dim, input_dim, num_layers, module_name
        )
        self.generator = Generator(
            z_dim, hidden_dim, num_layers, module_name
        )
        self.supervisor = Supervisor(
            hidden_dim, num_layers, module_name
        )
        self.discriminator = Discriminator(
            hidden_dim, num_layers, module_name
        )

    # -------------------------------------------------
    # Forward passes (modular)
    # -------------------------------------------------
    def forward_embedder(self, x):
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        return h, x_tilde

    def forward_generator(self, z):
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        x_hat = self.recovery(h_hat)
        return e_hat, h_hat, x_hat

    def forward_discriminator(self, h, h_hat, e_hat):
        y_real = self.discriminator(h)
        y_fake = self.discriminator(h_hat)
        y_fake_e = self.discriminator(e_hat)
        return y_real, y_fake, y_fake_e