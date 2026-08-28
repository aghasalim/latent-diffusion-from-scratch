"""Fail if a number in README.md no longer matches results/."""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _group(rows, key_cols, val):
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault(tuple(r[c] for c in key_cols), []).append(float(r[val]))
    return groups


def median_by(rows, key_cols, val, places):
    return {k: f"{statistics.median(v):.{places}f}"
            for k, v in _group(rows, key_cols, val).items()}


def extremes_by(rows, key_cols, val, places):
    """Min and max per group, for the seed-range columns the README quotes.

    A median alone hides how repeatable a number is. rFID at f=4 spans a factor
    of 5.9 across three seeds, wide enough that f=2 and f=4 are not separated,
    so the range is a claim in its own right and gets checked like one.
    """
    out = {}
    for k, v in _group(rows, key_cols, val).items():
        out[k + ("min",)] = f"{min(v):.{places}f}"
        out[k + ("max",)] = f"{max(v):.{places}f}"
    return out


def main() -> int:
    s1 = list(csv.DictReader((ROOT / "results" / "stage1.csv").open()))
    s2 = list(csv.DictReader((ROOT / "results" / "stage2.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()

    claims = []
    for col, places in (("psnr", 2), ("rfid", 3), ("compression", 1)):
        for k, v in median_by(s1, ["f"], col, places).items():
            claims.append((f"stage1 f={k[0]} {col}", v))
    # the rFID range column
    for k, v in extremes_by(s1, ["f"], "rfid", 3).items():
        claims.append((f"stage1 f={k[0]} rfid {k[1]}", v))
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
    # What this does and does not cover, so the green line is not read as more
    # than it is: each figure is recomputed from results/ and looked for in the
    # prose. It cannot catch a wrong number that happens to appear somewhere,
    # it does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/LOGBOOK.md.
    print("this checks quoted figures against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
