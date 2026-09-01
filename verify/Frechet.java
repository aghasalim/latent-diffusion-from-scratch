// The cFID kernel, reimplemented against the golden vectors.
//
// cFID is the headline metric of this repository: every quality number in both
// tables is a Frechet distance out of ldm/metrics.py _frechet. That function
// has one implementation, and the part of it that can go quietly wrong is the
// matrix square root, which is done through an eigendecomposition of the first
// covariance rather than a general sqrtm. An error there would not raise, it
// would return a plausible number, and every cFID in the repository would move
// together.
//
// So the whole thing is written again here from the formula in Heusel et al,
// with its own cyclic Jacobi eigensolver rather than a library call, and held
// against verify/golden/frechet-value.txt, which came straight out of PyTorch.
//
// Run: java verify/Frechet.java <root>

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class Frechet {

    // Both sides are double precision, so the only difference left is the order
    // of the floating point operations inside two different eigensolvers acting
    // on the same badly conditioned covariance. That is worth about 1e-9 here,
    // and the tolerance is set an order of magnitude outside it. A real mistake
    // in the square root moves the value by whole units, not by 1e-9.
    static final double TOL = 1e-7;
    static final double EPS = 1e-6;   // ldm/metrics.py _frechet, the ridge on both covariances

    static double[][] read(Path p) throws IOException {
        List<double[]> rows = new ArrayList<>();
        for (String line : Files.readAllLines(p)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] tok = line.trim().split("\\s+");
            double[] r = new double[tok.length];
            for (int i = 0; i < tok.length; i++) r[i] = Double.parseDouble(tok[i]);
            rows.add(r);
        }
        return rows.toArray(new double[0][]);
    }

    static double[] mean(double[][] x) {
        double[] m = new double[x[0].length];
        for (double[] row : x) for (int j = 0; j < m.length; j++) m[j] += row[j];
        for (int j = 0; j < m.length; j++) m[j] /= x.length;
        return m;
    }

    /** Unbiased covariance, the 1/(n-1) torch.cov uses by default. */
    static double[][] cov(double[][] x) {
        int n = x.length, d = x[0].length;
        double[] m = mean(x);
        double[][] c = new double[d][d];
        for (double[] row : x)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    c[i][j] += (row[i] - m[i]) * (row[j] - m[j]);
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) c[i][j] /= (n - 1);
        return c;
    }

    /**
     * Cyclic Jacobi on a symmetric matrix. a is overwritten with the diagonal
     * form; v comes back holding the eigenvectors in its columns. Written out
     * rather than called from a library because a library call would be the
     * same trust in someone else's linear algebra that this check exists to
     * avoid duplicating.
     */
    static double[] jacobi(double[][] a, double[][] v) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) v[i][j] = 0.0;
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            if (off < 1e-30) break;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta == 0.0) t = 1.0;
                    double c = 1 / Math.sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < n; k++) {
                        double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = v[k][p], vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        double[] ev = new double[n];
        for (int i = 0; i < n; i++) ev[i] = a[i][i];
        return ev;
    }

    static double[][] mul(double[][] a, double[][] b) {
        int n = a.length, m = b[0].length, k = b.length;
        double[][] c = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int p = 0; p < k; p++) {
                double aip = a[i][p];
                for (int j = 0; j < m; j++) c[i][j] += aip * b[p][j];
            }
        return c;
    }

    static double frechet(double[][] real, double[][] fake) {
        int d = real[0].length;
        double[] m1 = mean(real), m2 = mean(fake);
        double[][] s1 = cov(real), s2 = cov(fake);
        for (int i = 0; i < d; i++) { s1[i][i] += EPS; s2[i][i] += EPS; }

        // half = V sqrt(L) V^T, the symmetric square root of s1
        double[][] work = new double[d][d], vec = new double[d][d];
        for (int i = 0; i < d; i++) work[i] = s1[i].clone();
        double[] ev = jacobi(work, vec);
        double[][] scaled = new double[d][d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                scaled[i][j] = vec[i][j] * Math.sqrt(Math.max(ev[j], 0.0));
        double[][] vt = new double[d][d];
        for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) vt[i][j] = vec[j][i];
        double[][] half = mul(scaled, vt);

        double[][] inner = mul(mul(half, s2), half);
        // symmetrise, since the product is symmetric only up to rounding
        for (int i = 0; i < d; i++)
            for (int j = i + 1; j < d; j++) {
                double m = 0.5 * (inner[i][j] + inner[j][i]);
                inner[i][j] = m;
                inner[j][i] = m;
            }
        double[] iev = jacobi(inner, new double[d][d]);

        double out = 0.0;
        for (int i = 0; i < d; i++) out += (m1[i] - m2[i]) * (m1[i] - m2[i]);
        for (int i = 0; i < d; i++) out += s1[i][i] + s2[i][i];
        double tr = 0.0;
        for (double e : iev) tr += Math.sqrt(Math.max(e, 0.0));
        return out - 2 * tr;
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");
        Path g = root.resolve("verify").resolve("golden");
        double[][] real = read(g.resolve("frechet-real.txt"));
        double[][] fake = read(g.resolve("frechet-fake.txt"));
        double want = read(g.resolve("frechet-value.txt"))[0][0];

        double got = frechet(real, fake);
        double delta = Math.abs(got - want);
        boolean ok = delta <= TOL;
        System.out.printf("Frechet distance on %d x %d features against %d x %d%n",
                real.length, real[0].length, fake.length, fake[0].length);
        System.out.printf("  PyTorch  %.15f%n  Java     %.15f%n", want, got);
        System.out.printf("  |d| %.2e   relative %.2e   tol %.0e   %s%n",
                delta, delta / Math.abs(want), TOL, ok ? "ok" : "FAIL");
        if (!ok) {
            System.out.println("\nJava does not reproduce the cFID kernel");
            System.exit(1);
        }
        System.out.println("\nJava reproduces the cFID kernel with its own eigensolver");
    }
}
