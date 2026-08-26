"""Fail if a number in README.md no longer matches results/."""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def median_by(rows, key_cols, val, places):
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault(tuple(r[c] for c in key_cols), []).append(float(r[val]))
    return {k: f"{statistics.median(v):.{places}f}" for k, v in groups.items()}


def main() -> int:
    s1 = list(csv.DictReader((ROOT / "results" / "stage1.csv").open()))
    s2 = list(csv.DictReader((ROOT / "results" / "stage2.csv").open()))
    body = (ROOT / "README.md").read_text()

    claims = []
    for col, places in (("psnr", 2), ("rfid", 2), ("compression", 1)):
        for k, v in median_by(s1, ["f"], col, places).items():
            claims.append((f"stage1 f={k[0]} {col}", v))
    # cFID is quoted at every sampling budget; sW2 only at the largest, which is
    # what the results table states. Checking sW2 everywhere would fail on
    # numbers the README never claims.
    for k, v in median_by(s2, ["model", "nfe"], "cfid", 2).items():
        claims.append((f"stage2 {k[0]} nfe={k[1]} cfid", v))
    for k, v in median_by(s2, ["model", "nfe"], "sw2", 4).items():
        if k[1] == str(max(int(r["nfe"]) for r in s2)):
            claims.append((f"stage2 {k[0]} nfe={k[1]} sw2", v))
    for k, v in median_by(s2, ["model"], "train_s", 0).items():
        claims.append((f"stage2 {k[0]} train_s", v))

    failures = [f"{lab} should read {txt}, not found" for lab, txt in claims
                if not re.search(r"(?<![\d.])" + re.escape(txt) + r"(?!\d)", body)]

    print(f"checked {len(claims)} figures against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        # only report the ones the README is expected to quote
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
