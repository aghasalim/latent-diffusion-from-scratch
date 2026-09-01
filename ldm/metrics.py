"""Sample quality metrics.

Standard FID uses InceptionV3 features from ImageNet. That is the wrong feature
extractor for 32x32 grayscale digits and pulling it in would be a large
dependency for a number that would not mean what its name implies. So this repo
reports two things and names them honestly:

  cFID   Frechet distance in the feature space of a small CNN trained to
         classify MNIST digits. Same formula as FID, different features. It is
         comparable *within this repo* and not comparable to any published FID.
  sW2    sliced 2-Wasserstein in raw pixel space, which needs no learned
         features at all and cannot be gamed by the classifier.

Reporting both matters because they fail differently: cFID is blind to anything
the classifier ignores, sW2 is blind to semantics.
"""
from __future__ import annotations

import torch
from torch import nn


class Featurizer(nn.Module):
    """Small CNN whose penultimate layer provides the cFID feature space."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim), nn.ReLU())
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        return self.head(self.body(x))

    def features(self, x):
        return self.body(x)


def train_featurizer(x, y, steps=600, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = Featurizer()
    opt = torch.optim.Adam(m.parameters(), lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, x.shape[0], (batch,), generator=g)
        loss = nn.functional.cross_entropy(m(x[i]), y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        acc = (m(x[:2000]).argmax(-1) == y[:2000]).float().mean().item()
    return m, acc


def _frechet(mu1, s1, mu2, s2, eps=1e-6):
    """Frechet distance between two Gaussians, via an eigenvalue sqrt."""
    d = mu1 - mu2
    s1 = s1 + eps * torch.eye(s1.shape[0])
    s2 = s2 + eps * torch.eye(s2.shape[0])
    # sqrtm(s1 @ s2) through a symmetric decomposition of s1
    ev, evec = torch.linalg.eigh(s1)
    half = evec @ torch.diag(ev.clamp(min=0).sqrt()) @ evec.T
    inner = half @ s2 @ half
    iev = torch.linalg.eigvalsh(inner).clamp(min=0)
    return (d @ d + torch.trace(s1) + torch.trace(s2) - 2 * iev.sqrt().sum()).item()


@torch.no_grad()
def cfid(featurizer, real: torch.Tensor, fake: torch.Tensor) -> float:
    fr = featurizer.features(real).double()
    fk = featurizer.features(fake).double()
    return _frechet(fr.mean(0), fr.T.cov(), fk.mean(0), fk.T.cov())


def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_proj: int = 128, seed: int = 0,
              n_quantiles: int = 256, dirs: torch.Tensor | None = None) -> float:
    """Compared on a shared quantile grid so unequal sample counts are fine.

    dirs replaces the random projections. The draw is not identical on every CPU
    architecture, so the check in verify/ passes the directions it has on file
    and holds the rest of the kernel to a fixed answer.
    """
    a, b = a.flatten(1), b.flatten(1)
    if dirs is None:
        g = torch.Generator().manual_seed(seed)
        dirs = torch.randn(n_proj, a.shape[1], generator=g)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pa = (a @ dirs.T).sort(dim=0).values
    pb = (b @ dirs.T).sort(dim=0).values
    q = torch.linspace(0, 1, n_quantiles)

    def grab(p):
        idx = q * (p.shape[0] - 1)
        lo, hi = idx.floor().long(), idx.ceil().long()
        w = (idx - lo.float()).unsqueeze(1)
        return p[lo] * (1 - w) + p[hi] * w

    return ((grab(pa) - grab(pb)) ** 2).mean().sqrt().item()
