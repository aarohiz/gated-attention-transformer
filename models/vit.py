import torch
import torch.nn as nn
from models.gated_attention import StandardMultiHeadAttention, GatedMultiHeadAttention

# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "small": {
        "patch_size": 4,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "num_classes": 10,
    },
    "medium": {
        "patch_size": 4,
        "d_model": 384,
        "num_heads": 12,
        "num_layers": 8,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "num_classes": 10,
    },
}

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Splits image into non-overlapping patches and linearly projects each."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, d_model: int = 256):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, d_model)
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, attn: nn.Module, d_model: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Base ViT (shared logic)
# ---------------------------------------------------------------------------


class _ViTBase(nn.Module):
    def __init__(self, cfg: dict, attn_cls):
        super().__init__()
        patch_size = cfg["patch_size"]
        d_model = cfg["d_model"]
        num_heads = cfg["num_heads"]
        num_layers = cfg["num_layers"]
        mlp_ratio = cfg["mlp_ratio"]
        dropout = cfg["dropout"]
        num_classes = cfg["num_classes"]

        self.patch_embed = PatchEmbed(img_size=32, patch_size=patch_size, d_model=d_model)
        num_patches = self.patch_embed.num_patches  # 64 for 32x32 / patch 4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                attn=attn_cls(d_model=d_model, num_heads=num_heads, dropout=dropout, causal=False),
                d_model=d_model,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, d_model)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # classification on cls token

    def get_attn_weights(self) -> list:
        """Returns last_attn_weights from every block as a list of tensors."""
        return [block.attn.last_attn_weights for block in self.blocks]


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class ViT(_ViTBase):
    """ViT using standard multi-head attention."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, attn_cls=StandardMultiHeadAttention)


class GatedViT(_ViTBase):
    """ViT using gated multi-head attention."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, attn_cls=GatedMultiHeadAttention)
