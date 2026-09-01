# The claims the README makes in words rather than in table cells.
#
# scripts/check_numbers.py says what it does not cover: it looks for quoted
# figures and cannot check a claim written as prose. Everything the stage one
# section argues is written that way. The medians are in the table, but the
# reason only one claim is made from them is an argument about seed spread and
# interval overlap, and nothing checked it.
#
# So this recomputes, from results/stage1.csv and results/stage2.csv:
#
#   the three PSNR spreads and which is widest
#   the two closest seed gaps between neighbouring f
#   the rFID spread factor at f=4
#   which rFID intervals overlap and which do not
#   whether the pixel and latent cFID intervals overlap at every budget
#
# and requires the numbers it lands on to be the ones written in the prose.
#
# With three seeds a bootstrap would be theatre, so the separation claims are
# tested the only exact way available: complete separation of two groups of
# three has an exact one sided permutation p of 1/choose(6,3), which is printed
# beside each one so the reader can see how weak three seeds are even when the
# intervals do not touch.
#
# Base R, no packages.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

s1 <- read.csv(file.path(root, "results", "stage1.csv"))
s2 <- read.csv(file.path(root, "results", "stage2.csv"))
readme <- paste(readLines(file.path(root, "README.md"), warn = FALSE), collapse = "\n")

failures <- 0
EXACT_P <- 1 / choose(6, 3)

# says: recompute a number, print it, and require the given text in the README.
says <- function(label, text) {
    ok <- grepl(text, readme, fixed = TRUE)
    failures <<- failures + !ok
    cat(sprintf("  %-46s README says %-22s %s\n", label, text,
                if (ok) "ok" else "FAIL, not in the README"))
}

by_f <- function(df, colname, f) df[[colname]][df$f == f]

cat("stage one, seed spread\n")
spreads <- sapply(c(2, 4, 8), function(f) diff(range(by_f(s1, "psnr", f))))
names(spreads) <- c("2", "4", "8")
for (f in c("2", "4", "8"))
    says(sprintf("PSNR spread at f=%s is %.4f dB", f, spreads[[f]]),
         sprintf("%.2f dB", spreads[[f]]))

widest <- names(which.max(spreads))
cat(sprintf("  %-46s %s\n", "widest PSNR spread is at f=",
            paste0(widest, if (widest == "2") ", as the README says" else
                   ", but the README says f=2  FAIL")))
if (widest != "2") failures <- failures + 1

# The closest seeds between neighbouring levels: the worst case gap, which is
# what makes the distortion ordering safe rather than the medians.
gap <- function(hi, lo) min(by_f(s1, "psnr", hi)) - max(by_f(s1, "psnr", lo))
g24 <- gap(2, 4)
g48 <- gap(4, 8)
cat("\nstage one, distortion ordering at the closest seeds\n")
says(sprintf("f=2 over f=4 worst case is %.4f dB", g24), sprintf("%.1f dB", g24))
says(sprintf("f=4 over f=8 worst case is %.4f dB", g48), sprintf("%.1f dB", g48))
cat(sprintf("  %-46s exact one sided permutation p = %.3f\n",
            "both are complete separations of 3 against 3", EXACT_P))
if (g24 <= 0 || g48 <= 0) {
    cat("  FAIL: the PSNR levels are not ordered at their closest seeds\n")
    failures <- failures + 1
}

cat("\nstage one, rFID intervals\n")
iv <- lapply(c(2, 4, 8), function(f) range(by_f(s1, "rfid", f)))
names(iv) <- c("2", "4", "8")
for (f in c("2", "4", "8"))
    says(sprintf("rFID range at f=%s", f), sprintf("%.3f to %.3f", iv[[f]][1], iv[[f]][2]))

factor4 <- iv[["4"]][2] / iv[["4"]][1]
says(sprintf("rFID at f=4 spans a factor of %.4f", factor4),
     sprintf("factor of %.1f", factor4))

overlaps <- function(a, b) a[1] <= b[2] && b[1] <= a[2]
ov24 <- overlaps(iv[["2"]], iv[["4"]])
ov28 <- overlaps(iv[["2"]], iv[["8"]])
cat(sprintf("  %-46s %s\n", "f=2 and f=4 rFID intervals overlap",
            if (ov24) "yes, so no rise can be read  ok" else "no  FAIL"))
cat(sprintf("  %-46s %s\n", "f=2 and f=8 rFID intervals overlap",
            if (!ov28) "no, so the floor genuinely rises  ok" else "yes  FAIL"))
failures <- failures + !ov24 + ov28

cat("\nstage two, does any latent model overlap the pixel baseline\n")
cfid_range <- function(model, nfe)
    range(s2$cfid[s2$model == model & s2$nfe == nfe])
for (nfe in c(10, 25, 50)) {
    px <- cfid_range("pixel DDPM", nfe)
    for (m in c("LDM f=2", "LDM f=4", "LDM f=8")) {
        r <- cfid_range(m, nfe)
        ov <- overlaps(px, r)
        expect_overlap <- m == "LDM f=2"
        ok <- ov == expect_overlap
        failures <- failures + !ok
        cat(sprintf("  %-11s nfe=%2d  [%7.2f, %7.2f] against pixel [%7.2f, %7.2f]  %s  %s\n",
                    m, nfe, r[1], r[2], px[1], px[2],
                    if (ov) "overlaps    " else "separated   ",
                    if (ok) "ok" else "FAIL"))
    }
}
cat(sprintf("  %-46s %s\n", "f=2 against pixel is the weakest comparison",
            "as the README says, it is the only one that overlaps"))

# The one place the README concedes a latent model loses to a pixel seed.
best_px <- min(s2$cfid[s2$model == "pixel DDPM" & s2$nfe == 50])
f2_50 <- s2$cfid[s2$model == "LDM f=2" & s2$nfe == 50]
beat_all <- all(best_px < f2_50)
cat(sprintf("\n  %-46s %.2f against %s  %s\n",
            "best pixel run at 50 NFE beats all three f=2 runs",
            best_px, paste(sprintf("%.2f", sort(f2_50)), collapse = " "),
            if (beat_all) "ok" else "FAIL"))
failures <- failures + !beat_all

# The median claim that survives all of the above.
med_px <- sapply(c(10, 25, 50), function(n) median(s2$cfid[s2$model == "pixel DDPM" & s2$nfe == n]))
med_f2 <- sapply(c(10, 25, 50), function(n) median(s2$cfid[s2$model == "LDM f=2" & s2$nfe == n]))
better <- all(med_f2 < med_px)
cat(sprintf("  %-46s %s  %s\n", "f=2 beats pixel on the median at every budget",
            paste(sprintf("%.2f<%.2f", med_f2, med_px), collapse = " "),
            if (better) "ok" else "FAIL"))
failures <- failures + !better

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces every stage one spread claim and every interval claim\n")
