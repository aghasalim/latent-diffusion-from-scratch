"""MNIST, padded to 32x32 so downsampling by 2, 4 and 8 all divide evenly.

28 is not divisible by 8, so an f=8 autoencoder on raw MNIST would need
asymmetric padding somewhere and the compression ratios would stop being clean
powers of two. Padding to 32 costs nothing and makes the ablation legible.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")


def ensure_data(timeout: int = 120) -> Path:
    """Download MNIST on first use.

    The archives are 9.6 MB and gitignored rather than committed, so a fresh
    clone and CI both start without them. The README says the data is fetched on
    first use; this is the function that makes that true, which it was not until
    CI failed on three tests with FileNotFoundError.
    """
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    for name in FILES:
        target = d / name
        if target.exists() and target.stat().st_size > 0:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(f"{MIRROR}/{name}", timeout=timeout) as r:
            tmp.write_bytes(r.read())
        tmp.rename(target)          # rename last, so an interrupted download
    return d                        # never leaves a truncated file in place


def load(n: int | None = None, seed: int = 0):
    """Return (train, val) float tensors in [-1, 1], shape (N, 1, 32, 32)."""
    ensure_data()
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
    ensure_data()
    with gzip.open(ROOT / "data" / "train-labels-idx1-ubyte.gz") as f:
        raw = np.frombuffer(f.read(), np.uint8, offset=8)
    y = torch.from_numpy(raw.copy()).long()
    g = torch.Generator().manual_seed(seed)
    y = y[torch.randperm(y.shape[0], generator=g)]
    return y[:n] if n else y


def batches(x: torch.Tensor, size: int, generator: torch.Generator):
    i = torch.randint(0, x.shape[0], (size,), generator=generator)
    return x[i]
