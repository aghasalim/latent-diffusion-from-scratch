"""Stage one: perceptual compression.

The LDM factorisation is that an autoencoder handles compression and a
diffusion model handles composition in the compressed space. The knob is the
downsampling factor f: compress too little and stage two is expensive, too much
and stage two cannot recover detail the decoder already threw away.

This is a KL-regularised autoencoder, the continuous variant from the paper. The
KL term is weighted very low (1e-6 by default), which is deliberate and worth
stating: at full VAE weight the latent collapses toward the prior and
reconstructions blur, and the LDM authors use exactly this trick. The purpose of
the KL here is only to keep the latent scale bounded so the diffusion model sees
roughly unit variance, not to make a good generative model on its own.

No adversarial or perceptual loss. The paper uses both, and their absence is why
reconstructions here are blurry at high f in a way that LPIPS would punish more
than MSE does. That limitation is stated in the README rather than hidden.
"""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout, down=False, up=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(cin, cout, 4, stride=2, padding=1))
    elif up:
        layers.append(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1))
    else:
        layers.append(nn.Conv2d(cin, cout, 3, padding=1))
    layers += [nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    """f is the spatial downsampling factor; z_channels the latent depth."""

    def __init__(self, f: int = 4, z_channels: int = 4, base: int = 32,
                 in_channels: int = 1, resolution: int = 32):
        super().__init__()
        assert f in (1, 2, 4, 8, 16), f
        self.f, self.z_channels, self.resolution = f, z_channels, resolution
        n_down = int(torch.log2(torch.tensor(float(f))).item()) if f > 1 else 0

        enc, c = [_block(in_channels, base)], base
        for i in range(n_down):
            enc.append(_block(c, min(base * 2 ** (i + 1), 128), down=True))
            c = min(base * 2 ** (i + 1), 128)
        enc.append(nn.Conv2d(c, 2 * z_channels, 3, padding=1))     # mean and logvar
        self.encoder = nn.Sequential(*enc)

        dec, c2 = [_block(z_channels, c)], c
        for i in range(n_down):
            nxt = max(base, c2 // 2)
            dec.append(_block(c2, nxt, up=True))
            c2 = nxt
        dec += [_block(c2, base), nn.Conv2d(base, in_channels, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    @property
    def latent_shape(self):
        return (self.z_channels, self.resolution // self.f, self.resolution // self.f)

    def compression_ratio(self, in_channels: int = 1) -> float:
        """Elements in, divided by elements out."""
        pixels = in_channels * self.resolution ** 2
        c, h, w = self.latent_shape
        return pixels / (c * h * w)

    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 8.0)

    def sample_latent(self, mean, logvar):
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_latent(mean, logvar)
        return self.decode(z), mean, logvar

    def loss(self, x, kl_weight: float = 1e-6):
        recon, mean, logvar = self(x)
        rec = ((recon - x) ** 2).mean()
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return rec + kl_weight * kl, {"recon": rec.item(), "kl": kl.item()}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB for images in [-1, 1]."""
    mse = ((a - b) ** 2).mean().item()
    return 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()
