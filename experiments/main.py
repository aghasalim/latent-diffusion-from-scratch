"""Stage-1 rate distortion, then pixel diffusion against latent diffusion.

Three questions, in the order the paper poses them:

  1. What does the autoencoder cost you? Compress harder and stage two gets
     cheaper, but the decoder throws away detail no diffusion model above it can
     recover. That floor is measured directly as reconstruction quality.
  2. Is diffusion in the latent space cheaper per unit of quality than diffusion
     in pixel space? Same UNet, same schedule, same sampler, same step count.
  3. Where is the sweet spot on the compression knob?

    .venv/bin/python -m experiments.main
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from ldm.autoencoder import AutoEncoder, psnr
from ldm.data import batches, labels, load
from ldm.diffusion import Diffusion
from ldm.metrics import cfid, sliced_w2, train_featurizer
from ldm.unet import UNet

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# f -> UNet multipliers, chosen so the latent is never downsampled below 2x2
MULTS = {1: (1, 2), 2: (1, 2), 4: (1, 2), 8: (1,), 16: (1,)}


def train_autoencoder(f, train, val, args, seed):
    torch.manual_seed(seed)
    ae = AutoEncoder(f=f)
    opt = torch.optim.Adam(ae.parameters(), lr=args.ae_lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for step in range(args.ae_steps):
        loss, _parts = ae.loss(batches(train, args.batch, g), kl_weight=args.kl_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        opt.step()
    wall = time.perf_counter() - t0
    ae.eval()
    with torch.no_grad():
        recon, mean, _ = ae(val)
    return ae, {"psnr": psnr(recon, val), "mse": ((recon - val) ** 2).mean().item(),
                "wall_s": wall, "latent_std": mean.std().item()}, recon


def train_diffusion(space, ae, train, args, seed):
    """space is 'pixel' or 'latent'. Identical everything else."""
    torch.manual_seed(seed)
    if space == "latent":
        with torch.no_grad():
            mean, _ = ae.encode(train)
            scale = 1.0 / mean.std()
            data = mean * scale                    # unit variance, as the paper does
        ch, f = ae.z_channels, ae.f
    else:
        data, scale, ch, f = train, 1.0, 1, 1

    net = UNet(channels=ch, mults=MULTS[f])
    diff = Diffusion(args.T)
    opt = torch.optim.Adam(net.parameters(), lr=args.dm_lr)
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.perf_counter()
    for step in range(args.dm_steps):
        loss = diff.loss(net, batches(data, args.batch, g), generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    wall = time.perf_counter() - t0
    net.eval()
    return net, diff, float(scale), wall, data.shape[1:]


@torch.no_grad()
def sample(net, diff, ae, shape, n, nfe, scale, seed):
    t0 = time.perf_counter()
    z = diff.ddim_sample(net, (n, *shape), nfe=nfe,
                         generator=torch.Generator().manual_seed(seed))
    if ae is not None:
        z = ae.decode(z / scale)
    return z.clamp(-1, 1), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fs", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ae-steps", type=int, default=1500)
    ap.add_argument("--dm-steps", type=int, default=2500)
    ap.add_argument("--ae-lr", type=float, default=1e-3)
    ap.add_argument("--dm-lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=1e-6)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--nfes", nargs="+", type=int, default=[10, 25, 50])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    train, val = load(args.n)
    y = labels(args.n)[:train.shape[0]]
    feat, acc = train_featurizer(train, y, steps=1500)
    print(f"featurizer accuracy {acc:.3f} (cFID feature space)")

    stage1, stage2, started = [], [], time.perf_counter()
    real = val[:args.eval_n] if val.shape[0] >= args.eval_n else val
    for seed in args.seeds:
        # ---- pixel baseline
        net, diff, scale, wall, shape = train_diffusion("pixel", None, train, args, seed)
        for nfe in args.nfes:
            imgs, st = sample(net, diff, None, shape, args.eval_n, nfe, scale, seed + 9)
            stage2.append({"seed": seed, "model": "pixel DDPM", "f": 1, "nfe": nfe,
                           "cfid": cfid(feat, real, imgs), "sw2": sliced_w2(real, imgs),
                           "train_s": wall, "sample_s": st,
                           "latent_elems": int(torch.tensor(shape).prod()),
                           "params": sum(p.numel() for p in net.parameters())})
        print(f"  seed {seed}  pixel DDPM              train {wall:5.0f}s  "
              f"cFID@50 {stage2[-1]['cfid']:7.2f}")

        for f in args.fs:
            ae, m, recon = train_autoencoder(f, train, val, args, seed)
            rec_cfid = cfid(feat, real, recon[:args.eval_n])
            stage1.append({"seed": seed, "f": f, "compression": ae.compression_ratio(),
                           "latent_shape": str(ae.latent_shape), **m, "rfid": rec_cfid})
            print(f"  seed {seed}  AE f={f}  {ae.compression_ratio():4.0f}x  "
                  f"PSNR {m['psnr']:5.2f} dB  rFID {rec_cfid:6.2f}  {m['wall_s']:4.0f}s")

            net, diff, scale, wall, shape = train_diffusion("latent", ae, train, args, seed)
            for nfe in args.nfes:
                imgs, st = sample(net, diff, ae, shape, args.eval_n, nfe, scale, seed + 9)
                stage2.append({"seed": seed, "model": f"LDM f={f}", "f": f, "nfe": nfe,
                               "cfid": cfid(feat, real, imgs), "sw2": sliced_w2(real, imgs),
                               "train_s": wall, "sample_s": st,
                               "latent_elems": int(torch.tensor(shape).prod()),
                               "params": sum(p.numel() for p in net.parameters())})
            print(f"  seed {seed}  LDM f={f}                train {wall:5.0f}s  "
                  f"cFID@50 {stage2[-1]['cfid']:7.2f}")

    for fname, rows in (("stage1.csv", stage1), ("stage2.csv", stage2)):
        p = RESULTS / fname
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p.relative_to(ROOT)} ({len(rows)} rows)")
    (RESULTS / "run-meta.json").write_text(json.dumps({
        **vars(args), "featurizer_acc": acc,
        "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
