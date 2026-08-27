# latent-diffusion-from-scratch

A KL-regularised autoencoder, a DDPM, and the comparison the LDM paper is about:
running diffusion in a compressed latent instead of in pixels. MNIST at 32x32 on
a laptop CPU, three seeds, about 100 minutes end to end.

The headline: at a matched step count, diffusion in a 4x compressed latent
trains **5.3x faster** than the pixel baseline and reaches **6.2x better** sample
quality. At 16x compression it is **15.3x faster** and still 5.1x better.

![quality against cost](results/quality-vs-cost.png)

## The claim being tested

Pixel diffusion spends its capacity modelling texture. The LDM factorisation
splits the problem: an autoencoder handles perceptual compression, and the
diffusion model handles composition in the smaller space. Everything about the
diffusion model is held fixed here, the same UNet, the same cosine schedule, the
same DDIM sampler, the same number of training steps, so the only variable is
where the diffusion happens.

The knob is the downsampling factor f. Compress too little and stage two is
expensive. Compress too much and the decoder throws away detail no diffusion
model above it can recover. Finding where that trade sits is the actual subject.

## Stage one: what the autoencoder costs

Median of 3 seeds, with the seed range for rFID because it is much noisier than
the reconstruction error beside it.

| f | latent | compression | recon PSNR | rFID | rFID range over seeds | AE train |
|---:|---|---:|---:|---:|---|---:|
| 2 | (4, 16, 16) | 1.0x | 38.08 dB | 0.023 | 0.021 to 0.033 | 193 s |
| 4 | (4, 8, 8) | 4.0x | 31.65 dB | 0.034 | 0.018 to 0.107 | 245 s |
| 8 | (4, 4, 4) | 16.0x | 26.63 dB | 0.139 | 0.078 to 0.172 | 299 s |

![rate distortion](results/rate-distortion.png)

rFID is reconstruction quality measured in the same feature space as the
generation metric. It is the floor: no model trained on top of this autoencoder
can produce samples better than the autoencoder's own reconstructions.

**The range column is the reason this section makes only one claim.** PSNR is
tight, spanning 0.23 dB at f=2 and 0.15 dB at f=8, so the distortion ordering is
solid. rFID is not: at f=4 the three seeds span 0.018 to 0.107, a factor of 5.9,
and that interval sits entirely across f=2's. So **f=2 and f=4 are not separated
by rFID** and it would be wrong to read the medians 0.023 and 0.034 as a rise.

f=8 is a different matter. Its range, 0.078 to 0.172, does not overlap f=2's at
all, so the floor genuinely rises there, and that is where compression starts to
cost something real. One claim, and the numbers that support it are in the table
rather than behind a median.

f=2 is a useful control. It has four latent channels at half resolution, which
works out to 1.0x compression, so it is a latent space that is not actually
smaller. It still beats pixel diffusion, which says part of the benefit comes
from the latent being a smoother, more Gaussian space to diffuse in, not only
from having fewer elements.

## Stage two: quality against cost

Median of 3 seeds. Identical UNet, schedule and sampler throughout.

| model | elements per sample | train | cFID@10 | cFID@25 | cFID@50 | sW2@50 |
|---|---:|---:|---:|---:|---:|---:|
| pixel DDPM | 1024 | 671 s | 192.90 | 113.67 | 83.71 | 0.1759 |
| LDM f=2 | 1024 | 233 s | 62.06 | 50.21 | 44.29 | 0.1102 |
| **LDM f=4** | 256 | 128 s | 25.31 | 15.38 | **13.59** | **0.0802** |
| LDM f=8 | 64 | 44 s | 18.63 | 17.09 | 16.54 | 0.1008 |

![cost breakdown](results/cost-breakdown.png)

Three things worth pulling out.

**f=4 wins on quality, f=8 wins on cost.** f=4 reaches the best cFID at every
sampling budget above 10 steps. f=8 is three times cheaper again and only
slightly worse, and it is *better* than f=4 at 10 NFE, because a smaller latent
is easier to integrate in few steps. If you had a fixed sampling budget of 10
steps you would pick f=8; at 50 you would pick f=4.

**The pixel baseline is not converged and the latent models nearly are.** That is
not a flaw in the comparison, it is the comparison. Given the same 1200 steps,
the model working on 1024 elements per sample is still far from done while the
one working on 64 has essentially stopped improving. Its cFID falls 192.90 to
83.71 across sampling budgets and is still dropping steeply.

**sW2 and cFID mostly agree, and disagree at f=8.** cFID puts f=8 second, sW2
puts it third behind f=2. The two metrics look at different things: cFID uses
learned digit features, sW2 works on raw pixels. f=8 produces recognisable digits
with blurrier strokes, which the classifier forgives and pixel space does not.
Reporting one number would have hidden that.

## Metrics, named honestly

Standard FID uses InceptionV3 features trained on ImageNet, which is the wrong
feature extractor for 32x32 grayscale digits, and pulling it in would produce a
number that does not mean what its name implies. So this repo reports two things
and calls them what they are:

- **cFID**: the FID formula in the feature space of a small CNN trained here to
  classify MNIST, which reaches 94.5% accuracy. Comparable within this repo,
  not comparable to any published FID.
- **sW2**: sliced 2-Wasserstein in raw pixel space, no learned features at all.

They fail differently, which is why both are here.

## What I got wrong

**The UNet skip connections were misaligned and I did not notice from the
shapes.** The first version reconstructed skip channel counts from the multiplier
list rather than recording them on the way down. At the first upsampling step the
concatenation produced 96 channels into a block built for 128. GroupNorm raised
immediately, which is the useful thing about a normalisation layer that
validates its channel count: a plain convolution would have broadcast something
plausible and trained badly forever.

**I set out expecting compression to be a pure trade and it is not, at least not
here.** The assumption going in was that f=2 would be closest to pixel quality
and each step of compression would cost a little. Instead every latent model beat
pixel diffusion at matched steps, including the one at 1.0x compression that is
not smaller at all. The saving is not only about having fewer elements to
denoise.

**The comparison is at matched training steps, not matched wall clock.** That
choice favours the latent models, since a step on 64 elements is much cheaper
than a step on 1024. A matched wall clock comparison would give the pixel model
5 to 15 times more steps and would be the fairer test of "is the latent space
better" as opposed to "is it cheaper". I did not run it and the tables should be
read with that in mind.

## Limitations

- No perceptual or adversarial loss in stage one. The paper uses both, and their
  absence is why reconstructions here are blurry at high f in a way LPIPS would
  punish more than MSE does.
- MNIST at 32x32 is a long way from 256x256 faces. The mechanism should transfer;
  the numbers will not.
- 1200 diffusion steps is short. The pixel baseline in particular is undertrained.
- No conditioning, no classifier-free guidance, no cross attention. This is
  unconditional generation only.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m experiments.main --seeds 0 1 2 --fs 2 4 8
```

```bash
python -m bench.figures
```

The sweep takes about 100 minutes on an M4 CPU and writes `results/stage1.csv`
and `results/stage2.csv`. Figures read those files and never re-run an
experiment, so a plot cannot disagree with a number in this README.

MNIST is downloaded to `data/` on first use.

## Layout

```
ldm/data.py         MNIST padded to 32x32 so f=2,4,8 all divide evenly
ldm/autoencoder.py  KL-regularised autoencoder, the compression knob
ldm/unet.py         epsilon predictor, identical in both spaces
ldm/diffusion.py    cosine schedule, DDPM training, DDIM sampling
ldm/metrics.py      cFID and sliced W2, both named for what they are
experiments/main.py the sweep
tests/              32 tests
```

## Sources

- **Rombach, Blattmann, Lorenz, Esser, Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.** [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) The factorisation this repo tests, and the f ablation in section 4.1.
- **Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.** [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) The training objective.
- **Song, Meng, Ermon. Denoising Diffusion Implicit Models. ICLR 2021.** [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) DDIM, which is what makes a small NFE budget meaningful.
- **Nichol, Dhariwal. Improved Denoising Diffusion Probabilistic Models. ICML 2021.** [arXiv:2102.09672](https://arxiv.org/abs/2102.09672) The cosine schedule used here.
- **Esser, Rombach, Ommer. Taming Transformers for High-Resolution Image Synthesis. CVPR 2021.** [arXiv:2012.09841](https://arxiv.org/abs/2012.09841) The first-stage autoencoder design, including why the KL weight is tiny.
- **Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS 2017.** [arXiv:1706.08500](https://arxiv.org/abs/1706.08500) The FID formula, applied here to different features and renamed accordingly.
- MNIST from the CVDF mirror of LeCun's dataset.

Related: [rectified-flow-from-scratch](https://github.com/aghasalim/rectified-flow-from-scratch)
is the same generative problem with a straight interpolant instead of a diffusion path.

## Conventions

Shared rules in [`CONVENTIONS.md`](CONVENTIONS.md). Rule 8, no number that did not
come from a measurement, and rule 15, say what was not measured, are why the
limitations section is as long as it is.

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

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

MIT, see [LICENSE](LICENSE).
