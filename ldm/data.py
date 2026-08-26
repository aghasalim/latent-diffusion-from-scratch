"""MNIST, padded to 32x32 so downsampling by 2, 4 and 8 all divide evenly.

28 is not divisible by 8, so an f=8 autoencoder on raw MNIST would need
asymmetric padding somewhere and the compression ratios would stop being clean
powers of two. Padding to 32 costs nothing and makes the ablation legible.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(n: int | None = None, seed: int = 0):
    """Return (train, val) float tensors in [-1, 1], shape (N, 1, 32, 32)."""
    with gzip.open(ROOT / "data" / "train-images-idx3-ubyte.gz") as f:
        raw = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    x = torch.from_numpy(raw.copy()).float() / 255.0
    x = torch.nn.functional.pad(x.unsqueeze(1), (2, 2, 2, 2))     # 28 -> 32
    x = x * 2.0 - 1.0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    x = x[perm]
    if n:
        x = x[:n]
    cut = int(0.95 * x.shape[0])
    return x[:cut], x[cut:]


def labels(n: int | None = None, seed: int = 0):
    with gzip.open(ROOT / "data" / "train-labels-idx1-ubyte.gz") as f:
        raw = np.frombuffer(f.read(), np.uint8, offset=8)
    y = torch.from_numpy(raw.copy()).long()
    g = torch.Generator().manual_seed(seed)
    y = y[torch.randperm(y.shape[0], generator=g)]
    return y[:n] if n else y


def batches(x: torch.Tensor, size: int, generator: torch.Generator):
    i = torch.randint(0, x.shape[0], (size,), generator=generator)
    return x[i]
