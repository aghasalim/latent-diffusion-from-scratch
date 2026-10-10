"""Score the cFID featurizer on held-out images, without rerunning the sweep.

The run behind results/ recorded `featurizer_acc` in run-meta.json, but that was
measured on 2000 of the featurizer's own training images. The featurizer is
trained with a fixed seed on the same split, so this retrains it exactly as
experiments/main.py does, checks it lands on the recorded training-row figure,
and writes its accuracy on the validation images to results/featurizer.json.

    .venv/bin/python scripts/featurizer_acc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldm.data import labels, load
from ldm.metrics import train_featurizer

STEPS = 1500  # what experiments/main.py:113 trains the featurizer for


def main() -> int:
    meta = json.loads((ROOT / "results" / "run-meta.json").read_text())
    train, val = load(meta["n"])
    y = labels(meta["n"])
    y_tr, y_val = y[:train.shape[0]], y[train.shape[0]:]
    feat, val_acc = train_featurizer(train, y_tr, val, y_val, steps=STEPS)
    with torch.no_grad():
        train_rows_acc = (feat(train[:2000]).argmax(-1) == y_tr[:2000]).float().mean().item()
    if abs(train_rows_acc - meta["featurizer_acc"]) > 1e-6:
        print(f"retrained featurizer gets {train_rows_acc:.4f} on the training rows, "
              f"run-meta.json recorded {meta['featurizer_acc']:.4f}: not the same model")
        return 1
    out = {"val_acc": val_acc, "val_n": int(val.shape[0]), "train_rows_acc": train_rows_acc,
           "steps": STEPS, "n": meta["n"], "torch": torch.__version__}
    (ROOT / "results" / "featurizer.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"validation accuracy {val_acc:.4f} on {out['val_n']} images "
          f"(training rows {train_rows_acc:.4f}, as recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
