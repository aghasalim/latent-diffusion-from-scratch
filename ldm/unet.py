"""A small UNet epsilon-predictor, used unchanged in pixel space and latent space.

Using the same architecture for both is the point. If the pixel baseline and the
latent model had different backbones, any difference in cost or quality could be
attributed to the backbone rather than to where the diffusion happens.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0)
                          * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        ang = t.float().view(-1, 1) * freqs.view(1, -1)
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, t_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(t_dim, cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(nn.functional.silu(self.n1(x)))
        h = h + self.emb(nn.functional.silu(t))[:, :, None, None]
        h = self.c2(nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Symmetric UNet. Skip channel counts are recorded on the way down and
    consumed on the way up, rather than reconstructed from the multiplier list.

    The first version reconstructed them and was off by one level: at the first
    upsampling step the concatenation produced 96 channels while the block had
    been built for 128. GroupNorm caught it immediately, which is the useful
    thing about a normalisation layer that validates its channel count."""

    def __init__(self, channels: int = 1, base: int = 32, mults=(1, 2), t_dim: int = 64):
        super().__init__()
        self.time = TimeEmbedding(t_dim)
        self.inp = nn.Conv2d(channels, base, 3, padding=1)

        downs, skip_ch, c = [], [base], base
        for m in mults:
            out = base * m
            downs.append(nn.ModuleList([ResBlock(c, out, t_dim),
                                        nn.Conv2d(out, out, 4, stride=2, padding=1)]))
            c = out
            skip_ch.append(c)                       # tensor pushed after downsampling
        self.downs = nn.ModuleList(downs)
        self.mid = ResBlock(c, c, t_dim)

        skip_ch.pop()                               # the mid input is not re-used
        ups = []
        for m in reversed(mults):
            out = base * m
            s = skip_ch.pop()
            ups.append(nn.ModuleList([nn.ConvTranspose2d(c, out, 4, stride=2, padding=1),
                                      ResBlock(out + s, out, t_dim)]))
            c = out
        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(nn.GroupNorm(min(8, c), c), nn.SiLU(),
                                 nn.Conv2d(c, channels, 3, padding=1))

    def forward(self, x, t):
        temb = self.time(t)
        h = self.inp(x)
        skips = [h]
        for block, down in self.downs:
            h = block(h, temb)
            h = down(h)
            skips.append(h)
        h = self.mid(h, temb)
        skips.pop()
        for up, block in self.ups:
            h = up(h)
            h = block(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(h)
