# src/baselines/lstm_ae.py
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        
        batch_size, seq_len, _ = x.shape
        h_repeated = h_n.permute(1, 0, 2).repeat(1, seq_len, 1)
        
        out, _ = self.decoder(h_repeated)
        return self.output_layer(out)