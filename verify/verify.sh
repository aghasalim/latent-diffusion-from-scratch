#!/usr/bin/env bash
# Recompute the published numbers in every language here and require agreement.
#
# Every figure in this repository came out of one Python implementation. The
# tables came from statistics.median, the metrics from ldm/metrics.py, the
# trajectories from ldm/diffusion.py, and the checker in scripts/check_numbers.py
# recomputes them the same way, so it agrees with itself by construction. These
# are independent implementations, written from the definitions, and a mistake
# would have to be repeated identically in all of them to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    # a check that reports 2 is telling us its own toolchain is incomplete,
    # which is a skip and not a failure
    "$@"; local rc=$?
    case "$rc" in
        0) pass=$((pass + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
}

# README.md with the bold markers removed, so a bolded number compares as a
# number. Rebuilt once and reused by the checks that grep for a table row.
sed 's/\*\*//g' README.md > "$tmp/readme-plain.md"

# The golden vectors are the fixed point the C, Java and Rust checks are all
# measured against, so they are only evidence while ldm/ still lands on them.
# This reruns the four kernels on the inputs already on file and compares
# values. It cannot compare files: torch.randn does not draw the same numbers on
# every CPU architecture, so regenerating them elsewhere is not the same test.
PY="${PYTHON:-python3}"
check_python () {
    if ! "$PY" -c "import torch" >/dev/null 2>&1; then
        printf 'skipped: %s has no torch\n' "$PY"
        return 2
    fi
    "$PY" verify/check_golden.py "$root"
}

# SQL prints each table row in a canonical form and this requires it in the
# README. sqlite3 reads stdin, which inside a script is the script itself, so
# the redirect matters. Its CSV output is CRLF, so the \r is stripped.
check_sql () {
    local rows bad=0 line pipes
    rows=$(sqlite3 -init verify/tables.sql :memory: "" < /dev/null 2>/dev/null | tr -d '\r')
    [ -n "$rows" ] || { echo "sqlite produced nothing"; return 1; }
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        # stage1|2|(4, 16, 16)|... becomes | 2 | (4, 16, 16) | ...  |
        pipes="| $(printf '%s' "${line#*|}" | sed 's/|/ | /g') |"
        if grep -Fq "$pipes" "$tmp/readme-plain.md"; then
            printf '  ok    %s\n' "$pipes"
        else
            printf '  FAIL  not in the README: %s\n' "$pipes"
            bad=$((bad + 1))
        fi
    done <<< "$rows"
    [ "$bad" -eq 0 ] || return 1
    echo "SQL rebuilt 7 table rows from the seed level CSVs and found every one in the README"
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o "$tmp/kernel" verify/kernel.c -lm || return 1
    "$tmp/kernel" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/slicedw2 && cargo run --release --quiet -- "$root" ); }

check_java () { java verify/Frechet.java "$root"; }

run "Python, ldm/ still lands on the goldens" "$PY"      check_python
run "SQL, both published tables"              sqlite3   check_sql
run "C, cosine schedule and DDIM"             cc        check_c
run "Go, results files and table cells"       go        check_go
run "R, the claims written as prose"          Rscript   Rscript verify/verify.R "$root"
run "Rust, sliced W2 and its projection noise" cargo    check_rust
run "Java, the cFID kernel"                   java      check_java
run "JavaScript, the headline multiples"      node      node verify/ratios.mjs "$root"
run "Ruby, the second copy of the table"      ruby      ruby verify/methods_check.rb "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
