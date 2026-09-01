"""Write the reference vectors the C, Rust and Java checks are held against.

The three kernels in this repository that produce a published number and are
pure arithmetic are the cosine schedule, the DDIM update, and the two metrics.
They are implemented once, in PyTorch, so nothing has ever disagreed with them.
This dumps their inputs and outputs so an implementation in another language can
be required to land on the same values.

The inputs are synthetic and seeded, not MNIST, because what is being checked is
the arithmetic and not the data. The eps predictor the DDIM check uses is a fixed
closed form rather than the trained UNet, for the same reason: the sampler update
is the part with algebra in it, and the model is a black box on both sides.

Run from the repository root with a torch install:

    python verify/export_golden.py [output directory]

With no argument it rewrites verify/golden/. verify/verify.sh passes a temporary
directory instead and diffs the result, so that the check that the golden
vectors still match ldm/ cannot pass by quietly regenerating them.

It rewrites verify/golden/. The inputs are printed at 9 significant digits,
which round trips float32 exactly. The two reference values are float64 and are
printed at 17, so a reimplementation is held against the value and not against
the rounding of the file it reads.

This is run once and the result is committed. It is not rerun as a check:
torch.randn does not give the same draw on every CPU architecture, so the same
seed produces different inputs on a different machine. verify/check_golden.py is
the check, and it reads the inputs from the files rather than redrawing them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldm.diffusion import Diffusion, cosine_schedule
from ldm.metrics import _frechet, sliced_w2

GOLDEN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "verify" / "golden"
T = 400          # results/run-meta.json
NFE = 50         # the largest sampling budget in the tables
N_PROJ = 128     # the sliced_w2 default, which is what produced sW2 in results/


def write(name: str, comment: str, rows, digits: int = 9) -> None:
    lines = [f"# {comment}"]
    for row in rows:
        lines.append(" ".join(f"{float(v):.{digits}g}" for v in row))
    (GOLDEN / name).write_text("\n".join(lines) + "\n")
    print(f"  {name:<22} {len(lines) - 1} rows")


def eps_model(x, t):
    """A stand in for the trained UNet: deterministic, bounded, elementwise."""
    return torch.tanh(0.8 * x + 0.002 * t.view(-1, 1, 1, 1).float())


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}")

    abar = cosine_schedule(T)
    write("schedule.txt", f"ldm.diffusion.cosine_schedule({T}), alphas_cumprod",
          [[v] for v in abar.tolist()])

    d = Diffusion(T)
    steps = torch.linspace(d.T - 1, 0, NFE).long()
    write("ddim-steps.txt", f"the DDIM timestep subsequence at nfe={NFE}",
          [[int(v)] for v in steps.tolist()])

    shape = (4, 1, 4, 4)
    g = torch.Generator().manual_seed(0)
    x_init = torch.randn(shape, generator=g)
    g = torch.Generator().manual_seed(0)
    x_final = d.ddim_sample(eps_model, shape, nfe=NFE, eta=0.0, generator=g)
    write("ddim-init.txt", f"x_T, shape {shape}, one sample per row",
          x_init.flatten(1).tolist())
    write("ddim-final.txt", "x_0 after ddim_sample with eps = tanh(0.8x + 0.002t)",
          x_final.flatten(1).tolist())

    # Frechet distance inputs. Feature vectors out of ldm.metrics.Featurizer are
    # post ReLU, so the golden ones are too: nonnegative and rank deficient,
    # which is the case an eigenvalue based matrix square root has to survive.
    torch.manual_seed(7)
    dim, n = 16, 150
    basis = torch.randn(dim, dim)
    real = torch.relu(torch.randn(n, dim) @ basis + 0.5)
    fake = torch.relu(torch.randn(n, dim) @ basis * 1.15 + 0.2)
    fr, fk = real.double(), fake.double()
    value = _frechet(fr.mean(0), fr.T.cov(), fk.mean(0), fk.T.cov())
    write("frechet-real.txt", f"{n} feature vectors of dim {dim}", real.tolist())
    write("frechet-fake.txt", f"{n} feature vectors of dim {dim}", fake.tolist())
    write("frechet-value.txt", "ldm.metrics._frechet of the two above", [[value]],
          digits=17)
    print(f"  frechet = {value!r}")

    # Sliced W2 inputs. The directions are the ones sliced_w2 draws for itself,
    # exported so another implementation can be held to the same slices rather
    # than to torch's generator.
    torch.manual_seed(11)
    da, db = 64, 64
    a = torch.randn(64, 1, 8, 8)
    b = torch.randn(80, 1, 8, 8) * 1.2 + 0.15
    gg = torch.Generator().manual_seed(0)
    dirs = torch.randn(N_PROJ, da, generator=gg)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    value = sliced_w2(a, b, n_proj=N_PROJ, seed=0)
    write("sw2-a.txt", f"{a.shape[0]} samples flattened to {da}", a.flatten(1).tolist())
    write("sw2-b.txt", f"{b.shape[0]} samples flattened to {db}", b.flatten(1).tolist())
    write("sw2-dirs.txt", f"the {N_PROJ} unit directions sliced_w2 draws at seed 0",
          dirs.tolist())
    write("sw2-value.txt", "ldm.metrics.sliced_w2 of the two above", [[value]],
          digits=17)
    print(f"  sliced_w2 = {value!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
