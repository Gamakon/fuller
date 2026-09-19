//! Where does `denoise_candidates_assuming` spend its time?
//!
//! Runs every expression in a text file (one per line: `math<TAB>v1,v2,...`)
//! twice: with NO rows (rewrite + extract only) and with `N_ROWS` rows (adds
//! evaluation on data and the prune loop). The difference is the share of the
//! work that a device evaluator could take.
//!
//!   cargo run --release --example profile_candidates -- exprs.tsv [n_rows]

use std::time::Instant;

use fuller::extract::denoise_candidates_assuming;

const K_VARIANTS: usize = 8;

/// Deterministic rows: a fixed LCG, values in (0.5, 4.5) so protected ops and
/// roots stay in their ordinary range.
fn rows_for(vars: &[String], n_rows: usize) -> Vec<Vec<(String, f64)>> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n_rows)
        .map(|_| {
            vars.iter()
                .map(|v| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
                    (v.clone(), 0.5 + 4.0 * unit)
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: profile_candidates exprs.tsv [n_rows]");
    let n_rows: usize = args.next().map_or(256, |s| s.parse().expect("n_rows"));
    let text = std::fs::read_to_string(&path).expect("read exprs file");

    let mut t_no_rows = 0.0_f64;
    let mut t_rows = 0.0_f64;
    let mut n_ok = 0usize;
    let mut n_err = 0usize;
    let mut n_cands = 0usize;
    let mut slowest: Vec<(f64, f64, String)> = Vec::new();

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (math, vars) = line.split_once('\t').expect("math<TAB>vars");
        let vars: Vec<String> =
            vars.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
        let rows = rows_for(&vars, n_rows);

        let t0 = Instant::now();
        let bare = denoise_candidates_assuming(math, &[], K_VARIANTS, &[], &[]);
        let a = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let with = denoise_candidates_assuming(math, &rows, K_VARIANTS, &[], &[]);
        let b = t1.elapsed().as_secs_f64();

        match (bare, with) {
            (Ok(_), Ok(c)) => {
                n_ok += 1;
                n_cands += c.len();
                t_no_rows += a;
                t_rows += b;
                slowest.push((b, a, math.to_string()));
            }
            _ => n_err += 1,
        }
    }

    slowest.sort_by(|x, y| y.0.total_cmp(&x.0));
    let data = t_rows - t_no_rows;
    println!("expressions ok={n_ok} err={n_err} rows={n_rows} k={K_VARIANTS}");
    println!("candidates returned      {n_cands}");
    println!("rewrite+extract (no rows) {t_no_rows:9.3} s");
    println!("with rows                 {t_rows:9.3} s");
    println!(
        "data share                {data:9.3} s  ({:.1}% of with-rows)",
        100.0 * data / t_rows
    );
    println!("per expression            {:9.3} ms", 1e3 * t_rows / n_ok as f64);
    println!("slowest 10 (with rows s / no rows s):");
    for (b, a, m) in slowest.iter().take(10) {
        let short: String = m.chars().take(90).collect();
        println!("  {b:7.3} / {a:7.3}  {short}");
    }
}
