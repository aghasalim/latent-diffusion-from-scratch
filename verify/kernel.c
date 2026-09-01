/* Recompute the cosine schedule and the DDIM update in C.
 *
 * ldm/diffusion.py is the only implementation of either. Every number in the
 * README comes out of a sampler that used it, so an error in the schedule or in
 * the DDIM step would move every cFID and every sW2 in the tables together, and
 * nothing would look wrong. This is a second implementation, written from the
 * formulas rather than from the Python, and held against vectors that
 * verify/export_golden.py dumped straight out of the PyTorch one.
 *
 * The eps predictor is the closed form the exporter used, not the trained UNet.
 * The model is a black box on both sides; the algebra around it is the part
 * being checked.
 *
 * PyTorch works in float32 here and this works in double, so the two cannot be
 * required to agree bit for bit. The tolerances below are set well inside what
 * a real mistake would produce and are printed with the measured difference
 * next to them.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* glibc hides M_PI under -std=c99, so carry the constant rather than reaching
 * for a feature test macro. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define T_STEPS 400
#define NFE 50
#define SAMPLES 4
#define ELEMS 16

/* float32 round trips through the schedule, so the only difference left is the
 * libm cos, which is well under this. */
#define SCHEDULE_TOL 1e-7
/* 50 sequential steps of float32 arithmetic against double. */
#define DDIM_TOL 1e-4

static int read_numbers(const char *path, double *out, int max)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    char line[65536];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        for (char *tok = strtok(line, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
            if (n >= max) {
                fprintf(stderr, "%s has more than %d values\n", path, max);
                fclose(f);
                return -1;
            }
            out[n++] = strtod(tok, NULL);
        }
    }
    fclose(f);
    return n;
}

static double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Nichol and Dhariwal, as ldm/diffusion.py computes it: build alphas_cumprod
 * from the cosine, difference it into betas, clamp, then cumulative product
 * back. The round trip is not a no-op because of the clamp, which is why this
 * follows the same route rather than using the cosine directly. */
static void cosine_schedule(double s, float *abar)
{
    static double f[T_STEPS + 1], raw[T_STEPS + 1];
    for (int i = 0; i <= T_STEPS; i++) {
        double u = (double)i / T_STEPS;
        double c = cos((u + s) / (1.0 + s) * M_PI / 2.0);
        f[i] = c * c;
    }
    for (int i = 0; i <= T_STEPS; i++)
        raw[i] = clampd(f[i] / f[0], 1e-8, 1.0);

    double acc = 1.0;
    for (int i = 0; i < T_STEPS; i++) {
        double beta = clampd(1.0 - raw[i + 1] / raw[i], 0.0, 0.999);
        acc *= 1.0 - beta;
        abar[i] = (float)acc;
    }
}

static double eps_model(double x, int t)
{
    return tanh(0.8 * x + 0.002 * (double)t);
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[4096];
    static double buf[T_STEPS + 64];
    int failures = 0;

#define LOAD(name, dst, cap)                                        \
    snprintf(path, sizeof path, "%s/verify/golden/%s", root, name); \
    int n_##dst = read_numbers(path, dst, cap);                     \
    if (n_##dst < 0)                                                \
        return 2;

    /* 1. the schedule */
    LOAD("schedule.txt", buf, T_STEPS + 64)
    if (n_buf != T_STEPS) {
        fprintf(stderr, "schedule.txt has %d values, expected %d\n", n_buf, T_STEPS);
        return 2;
    }
    static float abar[T_STEPS];
    cosine_schedule(0.008, abar);

    double worst = 0.0;
    int worst_at = 0;
    for (int i = 0; i < T_STEPS; i++) {
        double d = fabs((double)abar[i] - buf[i]);
        if (d > worst) {
            worst = d;
            worst_at = i;
        }
    }
    printf("cosine schedule, %d alphas_cumprod   max |d| %.2e at t=%d  tol %.0e  %s\n",
           T_STEPS, worst, worst_at, SCHEDULE_TOL,
           worst <= SCHEDULE_TOL ? "ok" : "FAIL");
    failures += worst > SCHEDULE_TOL;

    /* 2. the timestep subsequence, which is integer and must match exactly */
    static double steps_golden[NFE];
    LOAD("ddim-steps.txt", steps_golden, NFE)
    if (n_steps_golden != NFE) {
        fprintf(stderr, "ddim-steps.txt has %d values, expected %d\n",
                n_steps_golden, NFE);
        return 2;
    }
    int steps[NFE], step_bad = 0;
    /* torch.linspace on CPU is float32 here, and it does not evaluate
     * start + i*step across the whole range: the first half is built forward
     * from start and the second half backward from end. That matters. Exact
     * arithmetic puts index 21 at 228.0, float32 puts it at 227.99998, and
     * .long() truncates, so the sampler actually visits t=227 and not t=228.
     * Computing this the obvious way in double disagrees with the run that
     * produced results/ at one of the fifty steps, so this reproduces what
     * torch does rather than what the formula says. */
    for (int i = 0; i < NFE; i++) {
        float st = (float)((0.0 - (double)(T_STEPS - 1)) / (NFE - 1));
        float v = i < NFE / 2 ? (float)(T_STEPS - 1) + (float)i * st
                              : 0.0f - (float)(NFE - 1 - i) * st;
        steps[i] = (int)v; /* torch .long() truncates toward zero */
        step_bad += steps[i] != (int)steps_golden[i];
    }
    printf("DDIM timesteps, nfe=%d                %d of %d differ                      %s\n",
           NFE, step_bad, NFE, step_bad == 0 ? "ok" : "FAIL");
    failures += step_bad != 0;

    /* 3. the sampler update */
    static double init[SAMPLES * ELEMS], final[SAMPLES * ELEMS];
    LOAD("ddim-init.txt", init, SAMPLES * ELEMS)
    LOAD("ddim-final.txt", final, SAMPLES * ELEMS)
    if (n_init != SAMPLES * ELEMS || n_final != SAMPLES * ELEMS) {
        fprintf(stderr, "ddim vectors are %d and %d, expected %d\n",
                n_init, n_final, SAMPLES * ELEMS);
        return 2;
    }

    static double x[SAMPLES * ELEMS];
    memcpy(x, init, sizeof x);
    for (int i = 0; i < NFE; i++) {
        int t = steps[i];
        double a_t = abar[t];
        double a_prev = i + 1 < NFE ? abar[steps[i + 1]] : 1.0;
        double sa = sqrt(a_t), s1ma = sqrt(1.0 - a_t);
        double dir = sqrt(a_prev < 1.0 ? 1.0 - a_prev : 0.0);
        double sap = sqrt(a_prev);
        for (int k = 0; k < SAMPLES * ELEMS; k++) {
            double eps = eps_model(x[k], t);
            double x0 = clampd((x[k] - s1ma * eps) / sa, -3.0, 3.0);
            x[k] = sap * x0 + dir * eps;
        }
    }

    worst = 0.0;
    worst_at = 0;
    for (int k = 0; k < SAMPLES * ELEMS; k++) {
        double d = fabs(x[k] - final[k]);
        if (d > worst) {
            worst = d;
            worst_at = k;
        }
    }
    printf("DDIM sample, %d steps on %d elements  max |d| %.2e at %d   tol %.0e  %s\n",
           NFE, SAMPLES * ELEMS, worst, worst_at, DDIM_TOL,
           worst <= DDIM_TOL ? "ok" : "FAIL");
    failures += worst > DDIM_TOL;

    if (failures) {
        printf("\n%d of 3 kernels disagree with PyTorch\n", failures);
        return 1;
    }
    printf("\nC reproduces the schedule, the timesteps and the DDIM trajectory\n");
    return 0;
}
