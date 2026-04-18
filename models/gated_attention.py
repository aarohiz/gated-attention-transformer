import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.last_attn_weights = None  # (B, num_heads, T, T) after last forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attn_weights = attn.detach()

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with a head-specific sigmoid gate applied after SDPA.
    gate = sigmoid(W_g(x))  where W_g: d_model -> num_heads
    Each head's output is element-wise scaled by its own scalar gate value.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # W_g: maps each token's representation to one gate scalar per head
        self.gate_proj = nn.Linear(d_model, num_heads)

        self.last_attn_weights = None  # (B, num_heads, T, T) after last forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attn_weights = attn.detach()

        # (B, H, T, D) before gating
        out = attn @ v  # (B, H, T, D)

        # gate: (B, T, H) -> (B, H, T, 1) so it broadcasts over head_dim
        gate = torch.sigmoid(self.gate_proj(x))          # (B, T, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)       # (B, H, T, 1)
        out = out * gate                                  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
