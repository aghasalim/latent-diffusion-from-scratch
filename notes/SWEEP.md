# The sweep, as it ran

`results/stage1.csv` and `results/stage2.csv` are the output of one invocation
of `experiments/main.py`. This page is the configuration of that invocation,
with each value traced to the line that set it or the column that recorded it.
Nothing here is remembered. If a value is not in the tree it is not on this page.

## The invocation

```
python -m experiments.main --seeds 0 1 2 --fs 2 4 8 --n 15000 --ae-steps 1000 --dm-steps 1200 --eval-n 750
```

That is the command in the README under "Running it", and
`results/run-meta.json` records the same values, because `experiments/main.py:157`
dumps the parsed arguments into it at the end of the run. The argparse defaults
at `experiments/main.py:98`, `:100`, `:101` and `:106` are larger (20000, 1500,
2500 and 1000). The published numbers come from the smaller run on the command
line, not from the defaults.

## When, on what

| | value | source |
|---|---|---|
| date | 2026-08-27 | author date of commit `3731945`, which added both CSVs; the logbook entries for the run carry the same date |
| device | cpu | `results/run-meta.json` `device` |
| machine | M4 laptop | README, "about 100 minutes on an M4 CPU"; `notes/LOGBOOK.md`, "5957 s total on an M4 CPU" |
| torch | 2.13.0 | `results/run-meta.json` `torch` |

## How long

| | seconds | source |
|---|---:|---|
| whole run | 5957.0 | `results/run-meta.json` `wall_clock_s`, the clock started at `experiments/main.py:116` |
| 9 autoencoders | 2212.4 | `results/stage1.csv` column `wall_s`, summed |
| 12 diffusion models | 3229.0 | `results/stage2.csv` column `train_s`, one value per (seed, model), summed |
| 36 sampling passes | 429.7 | `results/stage2.csv` column `sample_s`, summed |

Those three columns account for 5871 s. The 86 s they do not cover has no
column: it is the featurizer at `experiments/main.py:113`, the encoding of the
training set at `experiments/main.py:61`, and the cFID and sW2 calls, none of
which are timed.

Training time per diffusion model, from `train_s`, min to max over the three
seeds:

| model | train_s | params | source |
|---|---|---:|---|
| pixel DDPM | 669.9 to 676.2 | 485,313 | `stage2.csv` rows with `model` = pixel DDPM |
| LDM f=2 | 232.3 to 232.7 | 487,044 | `stage2.csv` rows with `f` = 2 |
| LDM f=4 | 127.3 to 127.7 | 487,044 | `stage2.csv` rows with `f` = 4 |
| LDM f=8 | 43.7 to 44.2 | 117,028 | `stage2.csv` rows with `f` = 8 |

The `params` column is the UNet only, counted at `experiments/main.py:127` and
`:146`. f=8 is smaller because `MULTS` at `experiments/main.py:33` gives it one
resolution level instead of two.

## Seeds

Three, 0, 1 and 2, from the `seed` column in both CSVs and `seeds` in
`run-meta.json`. Each one fans out into several generators:

| what | seed | where |
|---|---|---|
| autoencoder init and batches | seed | `experiments/main.py:37`, `:40` |
| diffusion init | seed | `experiments/main.py:58` |
| diffusion batches and noise | seed + 1 | `experiments/main.py:71` |
| DDIM starting noise | seed + 9 | `experiments/main.py:122`, `:141` |
| MNIST shuffle | 0, fixed | `ldm/data.py:42`, `:50` |
| featurizer | 0, fixed | `ldm/metrics.py:42`, `:43` |

So the featurizer and the data split are shared by all three seeds, and only the
models and their noise change between them.

## Settings

Every value below is in `results/run-meta.json` and is read from `args` in
`experiments/main.py` at the line given.

| setting | value | read at |
|---|---:|---|
| `n` | 15000 | `experiments/main.py:111` |
| `batch` | 128 | `:43`, `:74` |
| `ae_steps` | 1000 | `:42` |
| `dm_steps` | 1200 | `:73` |
| `ae_lr` | 1e-3 | `:39` |
| `dm_lr` | 2e-4 | `:70` |
| `kl_weight` | 1e-6 | `:43` |
| `T` | 400 | `:69` |
| `eval_n` | 750 | `:117`, `:122`, `:133`, `:141` |
| `nfes` | 10, 25, 50 | `:121`, `:140` |
| `fs` | 2, 4, 8 | `:131` |

Fixed in the code rather than on the command line:

| setting | value | where |
|---|---:|---|
| optimiser | Adam | `experiments/main.py:39`, `:70` |
| gradient clip | 1.0 | `experiments/main.py:46`, `:77` |
| latent scaling | 1 / std of the encoded training set | `experiments/main.py:62` |
| featurizer steps | 1500 | `experiments/main.py:113`, overriding the 600 at `ldm/metrics.py:42` |
| featurizer eval set | first 2000 training images | `ldm/metrics.py:57` |
| train / val split | 95 / 5 | `ldm/data.py:55` |

With `n` = 15000 that split is 14250 training images and 750 validation images,
so `eval_n` = 750 is the whole validation set, and the real side of every cFID
is the same 750 images for every row. The featurizer landed on 0.945 accuracy,
`featurizer_acc` in `run-meta.json`.

## What it wrote

- `results/stage1.csv`, 9 rows, one per (seed, f), written at `experiments/main.py:150`
- `results/stage2.csv`, 36 rows, one per (seed, model, nfe), same line
- `results/run-meta.json`, `experiments/main.py:157`

## What it did not write

No weights. There is no `torch.save` anywhere in `experiments/` or `ldm/`; the
only saves in the repo are the figure writes in `bench/figures.py` and the frame
cache at `bench/figures.py:218`. The 21 networks the sweep trained went out of
scope when their loop iteration ended, and the CSVs are what is left of them.
This is why the animation has to retrain the f=4 pair at seed 0
(`bench/figures.py:194` to `:201`) rather than load it, and why that cache,
`results/denoise-frames.npz`, is in `.gitignore`.

No loss curves either. The loops at `experiments/main.py:42` and `:73` keep no
per step record, so there is nothing to plot beyond the final metrics already in
the tables.
