"""A small UNet noise predictor eps_theta(x_t, t, y) for 28x28 images.

Resolution path 28 -> 14 -> 7 -> 14 -> 28 with skip connections. The timestep (sinusoidal
embedding -> MLP) and class label (learned embedding) are summed into one conditioning
vector that every ResBlock adds after its first convolution.

The class embedding table has num_classes + 1 rows: the extra row (index num_classes)
is the "null" label used for classifier-free guidance.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Transformer-style sinusoidal embedding of integer timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions (used only at 7x7 = 49 tokens)."""

    def __init__(self, ch: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = self.norm(x).flatten(2).transpose(1, 2)  # (b, h*w, c)
        out, _ = self.attn(seq, seq, seq, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        ch_mults: tuple = (1, 2, 3),
        num_res_blocks: int = 1,
        attn_levels: tuple = (2,),
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(
            in_ch=in_ch, base_ch=base_ch, ch_mults=tuple(ch_mults), num_res_blocks=num_res_blocks,
            attn_levels=tuple(attn_levels), num_classes=num_classes, dropout=dropout,
        )
        self.num_classes = num_classes
        self.base_ch = base_ch
        emb_dim = base_ch * 4

        self.time_mlp = nn.Sequential(nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder. Record the channel count of every activation pushed onto the skip stack.
        self.down = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, emb_dim, dropout)])
                if level in attn_levels:
                    block.append(SelfAttention(out_ch))
                self.down.append(block)
                ch = out_ch
                skip_chs.append(ch)
            if level != len(ch_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), SelfAttention(ch), ResBlock(ch, ch, emb_dim, dropout)])

        # Decoder: one extra block per level to consume the downsample skip.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout)])
                if level in attn_levels:
                    block.append(SelfAttention(out_ch))
                self.up.append(block)
                ch = out_ch
            if level != 0:
                # No conv after upsampling: the next ResBlock convolves the upsampled map anyway,
                # and a full-resolution conv here would cost ~20% of the network's FLOPs.
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest")]))

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))
        # Start as a zero predictor so early training is stable.
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base_ch)) + self.class_emb(y)

        h = self.in_conv(x)
        hs = [h]
        for block in self.down:
            if isinstance(block[0], ResBlock):
                h = block[0](h, emb)
                for layer in block[1:]:
                    h = layer(h)
            else:
                h = block[0](h)
            hs.append(h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, emb)

        for block in self.up:
            if isinstance(block[0], ResBlock):
                h = block[0](torch.cat([h, hs.pop()], dim=1), emb)
                for layer in block[1:]:
                    h = layer(h)
            else:
                for layer in block:
                    h = layer(h)

        return self.out(h)
