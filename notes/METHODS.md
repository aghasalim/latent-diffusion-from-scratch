# Methods and detail

Long form detail moved out of the README.


## Stage two: quality against cost


Median of 3 seeds. Identical UNet, schedule and sampler throughout.

| model | elements per sample | train | cFID@10 | cFID@25 | cFID@50 | sW2@50 |
|---|---:|---:|---:|---:|---:|---:|
| pixel DDPM | 1024 | 671 s | 192.90 | 113.67 | 83.71 | 0.1759 |
| LDM f=2 | 1024 | 233 s | 62.06 | 50.21 | 44.29 | 0.1102 |
| **LDM f=4** | 256 | 128 s | 25.31 | 15.38 | **13.59** | **0.0802** |
| LDM f=8 | 64 | 44 s | 18.63 | 17.09 | 16.54 | 0.1008 |

![cost breakdown](../results/cost-breakdown.png)

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
