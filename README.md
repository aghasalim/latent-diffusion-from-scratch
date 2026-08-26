# latent-diffusion-from-scratch

The two-stage recipe behind Stable Diffusion, built and measured: a perceptual autoencoder that throws away imperceptible detail, then a diffusion model that only has to learn what's left.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

Pixel-space diffusion spends most of its capacity modelling high-frequency texture that humans can't see and don't care about. The LDM insight is to factor the problem: a VAE with perceptual and adversarial losses handles *perceptual compression* (8× downsampling, ~64× fewer elements), and the diffusion model handles *semantic composition* in that smaller space. The compute saving is roughly two orders of magnitude, which is the difference between "a lab can train this" and "you can train this."

The interesting engineering is in stage 1. Compress too little and stage 2 is expensive; too much and stage 2 can't recover detail the decoder threw away. That rate–distortion knob is the real subject of this repo.

## Compute warning, read this

This is the most expensive project of the eight. On one consumer GPU:

- **Do:** FFHQ or CelebA-HQ at 128×128, or CIFAR-10, with `f=8`. Budget 100–300 GPU-hours end to end.
- **Don't:** attempt 512×512 text-to-image from scratch. That is a multi-GPU-month run and no amount of prompt engineering changes that.

Scaling down is not cheating — the ablations are what carry the project, and they're valid at 128×128.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- 12GB minimum, 24GB comfortable.

## Results

| Model | Resolution | FID ↓ | GPU-hours | NFE |
|---|---|---:|---:|---:|
| Pixel DDPM (baseline) | 64×64 | TODO | TODO | 100 |
| LDM f=4 | 128×128 | TODO | TODO | 100 |
| LDM f=8 | 128×128 | TODO | TODO | 100 |
| LDM f=16 | 128×128 | TODO | TODO | 100 |

Stage-1 rate–distortion:

| f | Latent shape | rFID ↓ | LPIPS ↓ | PSNR ↑ | Compression |
|---:|---|---:|---:|---:|---:|
| 4 | TODO | TODO | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO | TODO | TODO |

`rFID` is reconstruction FID — the floor no generative model above this autoencoder can beat.

## Waves

```
00 bootstrap + metrics                     (serial)
   ├─ 01 diffusion theory                  ┐
   └─ 02 pixel-space DDPM baseline         ┘ parallel
        └─ 03 the autoencoder (stage 1)    (serial — the hard one)
             ├─ 04 latent diffusion        ┐
             └─ 05 conditioning + CFG      ┘ parallel
                  └─ 06 ablations + writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `ldm/metrics/`, `data/` | — |
| 01 | `notes/00-diffusion.md`, `ldm/ref/` | `scripts/` |
| 02 | `ldm/pixel/`, `train/train_pixel.py` | `ldm/metrics/`, `data/` |
| 03 | `ldm/autoencoder/`, `train/train_ae.py` | `ldm/metrics/`, `data/` |
| 04 | `ldm/diffusion/`, `train/train_ldm.py` | `ldm/autoencoder/`, `ldm/ref/` |
| 05 | `ldm/conditioning/`, `ldm/guidance.py` | `ldm/diffusion/` |
| 06 | `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md). If repo 04 is built, its samplers and flow-matching objective drop in here as an alternative to DDPM — that comparison is worth running.

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).
