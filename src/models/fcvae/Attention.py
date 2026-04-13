import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderLayer_selfattn(nn.Module):
    """Encoder layer with self-attention + feedforward"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn


# --------------------------------------------------
# Multi-Head Attention
# --------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Projections
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # Better initialization
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        """
        q, k, v: [B, T, D]
        """
        B, T_q, _ = q.size()
        B, T_k, _ = k.size()
        B, T_v, _ = v.size()

        residual = q

        # Linear projections
        q = self.w_qs(q).view(B, T_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(B, T_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(B, T_v, self.n_head, self.d_v)

        # Reshape for attention: (B * n_head, T, d_k)
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, T_q, self.d_k)
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, T_k, self.d_k)
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, T_v, self.d_v)

        # Attention
        output, attn = self.attention(q, k, v)

        # Restore shape
        output = output.view(B, self.n_head, T_q, self.d_v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(B, T_q, -1)

        # Final projection
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


# --------------------------------------------------
# Scaled Dot Product Attention
# --------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        """
        q, k, v: [B*n_head, T, d_k]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature

        # Stability improvement
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn


# --------------------------------------------------
# Feed Forward
# --------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=1)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=1)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        residual = x

        # Conv expects [B, D, T]
        x = x.transpose(1, 2)

        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)

        x = x.transpose(1, 2)

        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x
