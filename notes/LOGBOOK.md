# Logbook

## 2026-08-27, the UNet skip connections were misaligned, and GroupNorm caught it
**Tried:** first forward pass of the UNet at 32x32 with mults=(1,2).
**Measured:** `RuntimeError: Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [128] and input of shape [4, 96, 16, 16]`.
**Concluded:** I reconstructed the skip channel counts from the multiplier list on the way up instead of recording them on the way down, and was off by one level. Worth noting what saved me: GroupNorm validates its channel count, so it raised at the exact layer. A plain convolution would have accepted whatever it was given, produced a plausible tensor, and trained badly forever. Rewrote the UNet to push channel counts onto a list during the downward pass and pop them on the way up, and there is now a shape test parametrised over five channel/resolution/multiplier combinations.

## 2026-08-27, stage one rate distortion, three seeds
**Tried:** KL-regularised autoencoders at f=2, 4 and 8 on MNIST padded to 32x32, 1000 steps each.
**Measured:** f=2 gives 1.0x compression at 38.08 dB PSNR and rFID 0.02; f=4 gives 4.0x at 31.65 dB and 0.03; f=8 gives 16.0x at 26.63 dB and 0.14.
**Concluded:** the floor is what matters here. rFID is the best any generative model above the autoencoder can do, and it stays essentially flat from f=2 to f=4 then rises 5x at f=8. So the first 4x of compression is close to free and the next 4x is not. Also worth recording that f=2 with four latent channels at half resolution is exactly 1.0x compression, which makes it an accidental but useful control: a latent space that is not smaller.

## 2026-08-27, latent diffusion beats pixel diffusion by more than the element count explains
**Tried:** identical UNet, cosine schedule, DDIM sampler and 1200 training steps, run in pixel space and in each latent space. 3 seeds, 5957 s total on an M4 CPU.
**Measured:** pixel DDPM 671 s training, cFID@50 83.71, sW2 0.1759. LDM f=4 128 s and 13.59, so 5.3x faster and 6.2x better. LDM f=8 44 s and 16.54, 15.3x faster and 5.1x better. LDM f=2, which is 1.0x compression, still gets 44.29 at 233 s.
**Concluded:** the part I did not expect is f=2. It operates on exactly as many elements as the pixel model, 1024, and still beats it by 1.9x on cFID while training 2.9x faster. So the benefit is not only "fewer things to denoise": the latent is a smoother and more nearly Gaussian space, and the autoencoder has already removed the high frequency detail the diffusion model would otherwise spend capacity on. Two more things to keep: f=8 beats f=4 at 10 NFE and loses at 50, because a smaller latent is easier to integrate in few steps, so the right f depends on the sampling budget you plan to use. And cFID and sW2 disagree about f=8, which is the argument for reporting both: blurry strokes are forgiven by a digit classifier and not by pixel space.

## 2026-08-27, the comparison favours the latent models and I should say so
**Tried:** nothing, this is a note about the design.
**Measured:** a training step on 64 elements is far cheaper than one on 1024, and the tables hold step count fixed rather than wall clock.
**Concluded:** matched steps is the standard way to isolate "where does diffusion happen" from "how much compute did you spend", but it does flatter the latent models. A matched wall clock run would give the pixel baseline 5 to 15 times more steps and is the fairer test of whether the latent space is genuinely better as opposed to merely cheaper. I did not run it, the README says so, and it is the first thing I would do next.
