"""DDPM, agnostic to whether it operates on pixels or latents.

Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, with a cosine
schedule. The model predicts eps and the loss is a plain MSE on it.

Sampling supports the full ancestral DDPM chain and the deterministic DDIM
subsequence, because the whole cost comparison in this repo is about number of
function evaluations and DDIM is what makes a small NFE budget meaningful.

Nothing here knows about images. That is what lets the identical object run in
pixel space for the baseline and in latent space for the LDM.
"""
from __future__ import annotations

import math

import torch


def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol and Dhariwal cosine alphas_cumprod."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-8, 1.0)
    betas = (1 - abar[1:] / abar[:-1]).clamp(0.0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class Diffusion:
    def __init__(self, T: int = 400):
        self.T = T
        self.abar = cosine_schedule(T)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def loss(self, model, x0, generator=None):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), generator=generator)
        xt, noise = self.q_sample(x0, t)
        return ((model(xt, t) - noise) ** 2).mean()

    @torch.no_grad()
    def ddim_sample(self, model, shape, nfe: int = 50, eta: float = 0.0,
                    generator=None, device="cpu"):
        """Deterministic DDIM by default. NFE is exactly `nfe` model calls."""
        steps = torch.linspace(self.T - 1, 0, nfe).long()
        x = torch.randn(shape, generator=generator, device=device)
        for i, t in enumerate(steps):
            tb = torch.full((shape[0],), int(t), dtype=torch.long, device=device)
            eps = model(x, tb)
            a_t = self.abar[t]
            a_prev = self.abar[steps[i + 1]] if i + 1 < len(steps) else torch.tensor(1.0)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-3, 3)
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0 + dir_xt
            if eta > 0 and i + 1 < len(steps):
                x = x + sigma * torch.randn(shape, generator=generator, device=device)
        return x
