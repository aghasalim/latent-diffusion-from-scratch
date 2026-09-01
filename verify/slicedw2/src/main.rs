//! The sliced Wasserstein kernel, reproduced and then given an error bar.
//!
//! sW2 is the second metric in both published tables and the only one that does
//! not go through a learned featuriser. It is a Monte Carlo estimate: 128 random
//! directions, and a different 128 would give a different number. The repository
//! reports it to four decimal places and never says how much of the fourth
//! decimal is projection noise, because measuring that means recomputing the
//! metric thousands of times and the Python sweep could not afford it.
//!
//! Two things happen here. First the metric is reimplemented from the definition
//! and required to reproduce verify/golden/sw2-value.txt on the exact directions
//! PyTorch drew. Then the same metric is run again on 800 independent draws of
//! 128 fresh directions from an xorshift generator of its own, which gives the
//! spread the published figure sits inside, and a much larger single estimate
//! that says where the 128 direction estimate is aiming.
//!
//! No crates.
//!
//! Run: cargo run --release -- <root>

use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

const N_PROJ: usize = 128; // ldm/metrics.py sliced_w2 default, and what produced results/
const N_QUANTILES: usize = 256;
const REPLICATES: usize = 800;
const BIG_PROJ: usize = 20000;
/// PyTorch runs this kernel in float32 and this runs in f64, so the two are held
/// to a relative tolerance rather than to equality.
const REL_TOL: f64 = 1e-6;

fn read_matrix(path: &Path) -> Vec<Vec<f64>> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect()
}

/// xorshift64*, which is plenty for direction sampling and is written out so
/// this check carries no dependency at all.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self) -> f64 {
        // (0, 1), open at both ends so the log below is finite
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn unit_direction(&mut self, d: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..d).map(|_| self.normal()).collect();
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }
}

/// Sorted projections of every sample onto one direction.
fn project_sorted(x: &[Vec<f64>], dir: &[f64]) -> Vec<f64> {
    let mut p: Vec<f64> = x
        .iter()
        .map(|row| row.iter().zip(dir).map(|(a, b)| a * b).sum())
        .collect();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    p
}

/// Linear interpolation onto the shared quantile grid, which is what lets two
/// sample counts be compared at all.
fn on_grid(sorted: &[f64], out: &mut [f64]) {
    let n = sorted.len();
    for (k, o) in out.iter_mut().enumerate() {
        let q = k as f64 / (N_QUANTILES - 1) as f64;
        let idx = q * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let w = idx - lo as f64;
        *o = sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }
}

fn sliced_w2(a: &[Vec<f64>], b: &[Vec<f64>], dirs: &[Vec<f64>]) -> f64 {
    let (mut ga, mut gb) = ([0.0; N_QUANTILES], [0.0; N_QUANTILES]);
    let mut acc = 0.0;
    for dir in dirs {
        on_grid(&project_sorted(a, dir), &mut ga);
        on_grid(&project_sorted(b, dir), &mut gb);
        for k in 0..N_QUANTILES {
            let d = ga[k] - gb[k];
            acc += d * d;
        }
    }
    (acc / (dirs.len() * N_QUANTILES) as f64).sqrt()
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".into());
    let g = Path::new(&root).join("verify").join("golden");
    let a = read_matrix(&g.join("sw2-a.txt"));
    let b = read_matrix(&g.join("sw2-b.txt"));
    let dirs = read_matrix(&g.join("sw2-dirs.txt"));
    let want = read_matrix(&g.join("sw2-value.txt"))[0][0];
    assert_eq!(dirs.len(), N_PROJ, "golden directions are not {N_PROJ}");

    let got = sliced_w2(&a, &b, &dirs);
    let rel = (got - want).abs() / want.abs();
    let ok = rel <= REL_TOL;
    println!(
        "sliced W2 on {} against {} samples, the {} directions PyTorch drew",
        a.len(),
        b.len(),
        N_PROJ
    );
    println!("  PyTorch  {want:.15}\n  Rust     {got:.15}");
    println!(
        "  relative {rel:.2e}   tol {REL_TOL:.0e}   {}",
        if ok { "ok" } else { "FAIL" }
    );

    // How much of the published fourth decimal is the choice of directions.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let d = a[0].len();
    let mut est = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        let fresh: Vec<Vec<f64>> = (0..N_PROJ).map(|_| rng.unit_direction(d)).collect();
        est.push(sliced_w2(&a, &b, &fresh));
    }
    let mean = est.iter().sum::<f64>() / est.len() as f64;
    let sd = (est.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (est.len() - 1) as f64).sqrt();
    let (lo, hi) = est.iter().fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));

    let big: Vec<Vec<f64>> = (0..BIG_PROJ).map(|_| rng.unit_direction(d)).collect();
    let converged = sliced_w2(&a, &b, &big);

    println!(
        "\n{REPLICATES} independent draws of {N_PROJ} directions, {} projections in total",
        REPLICATES * N_PROJ + BIG_PROJ
    );
    println!("  mean     {mean:.6}\n  sd       {sd:.6}\n  range    {lo:.6} to {hi:.6}");
    println!("  {BIG_PROJ} directions in one estimate: {converged:.6}");
    println!(
        "  the {N_PROJ} direction estimate carries about {:.1}% projection noise",
        100.0 * sd / mean
    );

    // The published value has to be an ordinary member of that spread, not an
    // outlier that a lucky seed produced.
    let z = (want - mean).abs() / sd;
    let inside = z <= 4.0;
    println!(
        "  the published value is {z:.2} sd from the mean of the draws   {}",
        if inside { "ok" } else { "FAIL" }
    );

    if !ok || !inside {
        println!("\nRust does not reproduce the sliced W2 kernel");
        exit(1);
    }
    println!("\nRust reproduces the sliced W2 kernel and bounds its projection noise");
}
