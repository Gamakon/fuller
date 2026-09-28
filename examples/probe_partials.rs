//! End-to-end test of `GpuEvaluator::eval_with_partials`.
//!
//! Goal: prove that emitting per-subtree partials lets us LOCATE the Lorentz
//! factor `1/sqrt(1-(v/c)^2)` inside a real evolved gene, purely by VALUE-space
//! correlation against an independently-computed host reference — on real
//! I.48.2 data.
//!
//! Build + run:
//!   cargo build --release --features gpu --example probe_partials
//!   cargo run   --release --features gpu --example probe_partials
//!
//! Gated on `feature = "gpu"` inside the file (rather than a Cargo.toml
//! `[[example]]` block) so the plain build / clippy stay untouched.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("probe_partials requires the gpu feature: cargo run --release --features gpu --example probe_partials");
}

#[cfg(feature = "gpu")]
fn main() {
    if let Err(e) = run() {
        eprintln!("probe_partials FAILED: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), String> {
    use fuller::gpu_eval::{ExprBatch, GpuEvaluator, MAX_NODES, math_to_nodes};

    const DATA: &str = "/private/tmp/claude-501/-Users-andrewmorgan-Dev-gamakon-fuller/b3c1f1fe-1035-445f-ada0-9f05e732d5e6/scratchpad/data/feynman_I_48_2.tsv";
    const N_ROWS: usize = 4000;
    const N_VARS: usize = 3;

    // Opcode -> name, for a readable node table. Index is the Op discriminant.
    const OP_NAMES: [&str; 28] = [
        "Var", "Num", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Log", "Exp", "Sin",
        "Cos", "Tan", "Tanh", "Pow", "Pow2", "Pow3", "Inv", "ProtectedDiv", "ProtectedSqrt",
        "ProtectedLog", "ProtectedExp", "ProtectedInv", "Asin", "Acos", "ProtectedAsin",
        "ProtectedAcos",
    ];

    // --- 1. Load data: header `m v c target`; m,v,c = x_0,x_1,x_2. ---------
    let text = std::fs::read_to_string(DATA).map_err(|e| format!("read {DATA}: {e}"))?;
    let mut rows_flat: Vec<f32> = Vec::with_capacity(N_ROWS * N_VARS);
    let mut y: Vec<f32> = Vec::with_capacity(N_ROWS);
    let mut v_col: Vec<f64> = Vec::with_capacity(N_ROWS); // x_1
    let mut c_col: Vec<f64> = Vec::with_capacity(N_ROWS); // x_2

    for line in text.lines().skip(1).take(N_ROWS) {
        let f: Vec<f64> = line
            .split('\t')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("parse row {line:?}: {e}"))?;
        if f.len() < 4 {
            return Err(format!("row has {} cols, expected 4: {line:?}", f.len()));
        }
        rows_flat.push(f[0] as f32); // x_0 = m
        rows_flat.push(f[1] as f32); // x_1 = v
        rows_flat.push(f[2] as f32); // x_2 = c
        y.push(f[3] as f32);
        v_col.push(f[1]);
        c_col.push(f[2]);
    }
    let n_rows = y.len();
    if n_rows == 0 {
        return Err("no data rows loaded".into());
    }
    println!("loaded {n_rows} rows, {N_VARS} vars (rows_flat.len()={})", rows_flat.len());

    // --- 2. Build the gene: m*c^2 * Lorentz(v,c). ------------------------
    // Lorentz factor 1/sqrt(1-(v/c)^2) is the ProtectedInv-rooted SUBTREE.
    let vars = vec!["x_0".to_string(), "x_1".to_string(), "x_2".to_string()];
    let sexpr = r#"(Mul (Mul (Var "x_0") (Pow2 (Var "x_2"))) (ProtectedInv (ProtectedSqrt (Sub (Num 1.0) (Pow2 (ProtectedDiv (Var "x_1") (Var "x_2")))))))"#;
    let nodes = math_to_nodes(sexpr, &vars)?;
    let node_count = nodes.len();
    println!("gene has {node_count} nodes (MAX_NODES={MAX_NODES})");
    if node_count > MAX_NODES {
        return Err(format!("gene has {node_count} nodes > MAX_NODES {MAX_NODES}"));
    }

    println!("\nNODE TABLE (k : op):");
    for (k, nd) in nodes.iter().enumerate() {
        let name = OP_NAMES.get(nd.op as usize).copied().unwrap_or("?");
        println!("  k={k:2}  {name}");
    }

    let mut batch = ExprBatch::new();
    let expr = batch.push(&nodes);
    assert_eq!(expr, 0);

    // --- 3. Evaluate with partials on the GPU. ---------------------------
    let ev = GpuEvaluator::new(&rows_flat, N_VARS)?;
    let gpu_n_rows = ev.n_rows() as usize;
    if gpu_n_rows != n_rows {
        return Err(format!("GPU n_rows {gpu_n_rows} != host {n_rows}"));
    }
    let (roots, partials) = ev.eval_with_partials(&batch)?;

    // Slice out each subtree's value vector S_k.
    // partials[(expr * MAX_NODES + k) * n_rows + row]
    let subtree = |k: usize| -> &[f32] {
        let base = (expr * MAX_NODES + k) * n_rows;
        &partials[base..base + n_rows]
    };

    // --- 4. Host reference Lorentz(v,c). ---------------------------------
    let lorentz: Vec<f64> = v_col
        .iter()
        .zip(&c_col)
        .map(|(&v, &c)| {
            let r = v / c;
            1.0 / (1.0 - r * r).sqrt()
        })
        .collect();

    // --- Sanity checks (cheap, catch an indexing/formula bug). -----------
    // (i) partials[k=0] must equal roots bit-for-bit (both are the gene root).
    let s0 = subtree(0);
    let bit_match = s0.iter().zip(&roots).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("\nsanity: partials[k=0] == roots bit-for-bit: {bit_match}");
    if !bit_match {
        return Err("partials[k=0] != roots: partials indexing is wrong".into());
    }
    // (ii) roots vs target y: the gene IS the law, so |r| ~= 1, small rel-err.
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let roots64: Vec<f64> = roots.iter().map(|&v| v as f64).collect();
    let (r_root, n_root) = abs_pearson(&roots64, &y64);
    let root_rel = max_rel_err(&roots64, &y64);
    println!(
        "sanity: roots vs target y  |r|={r_root:.7}  max_rel_err={root_rel:.3e}  (finite rows {n_root})"
    );

    // --- 5. |r| between each subtree S_k and the Lorentz reference. ------
    let mut scored: Vec<(usize, &str, f64, usize)> = Vec::with_capacity(node_count);
    for (k, nd) in nodes.iter().enumerate() {
        let sk: Vec<f64> = subtree(k).iter().map(|&v| v as f64).collect();
        let (r, n_fin) = abs_pearson(&sk, &lorentz);
        let name = OP_NAMES.get(nd.op as usize).copied().unwrap_or("?");
        scored.push((k, name, r, n_fin));
    }
    // Descending by |r|; NaN/zero-variance columns sort last (r set to 0).
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTOP subtrees by |Pearson r| vs Lorentz reference:");
    println!("  {:>4}  {:<16} {:>12}  {:>8}", "k", "op", "|r|", "finite");
    for &(k, name, r, n_fin) in scored.iter().take(8) {
        println!("  {k:>4}  {name:<16} {r:>12.7}  {n_fin:>8}");
    }

    // --- 6. SUCCESS criterion. -------------------------------------------
    let (top_k, top_name, top_r, _) = scored[0];
    let (_, _, second_r, _) = scored.get(1).copied().unwrap_or((0, "", 0.0, 0));
    let n_over_999 = scored.iter().filter(|s| s.2 > 0.999).count();

    // Affine-free correctness check on the winner: does the GPU partial equal
    // the host Lorentz value, not just correlate with it?
    let winner: Vec<f64> = subtree(top_k).iter().map(|&v| v as f64).collect();
    let win_rel = max_rel_err(&winner, &lorentz);

    println!("\n--- RESULT ---");
    println!("top subtree: k={top_k} ({top_name})  |r|={top_r:.7}");
    println!("runner-up |r|={second_r:.7}   gap={:.7}", top_r - second_r);
    println!("subtrees with |r| > 0.999: {n_over_999} (Pearson is affine-invariant, so monotone ancestors correlate too — expected, not a bug)");
    println!("winner max_rel_err vs host Lorentz (f32 exactness, affine-free): {win_rel:.3e}");

    let is_pinv = top_name == "ProtectedInv";
    if top_r > 0.999 && is_pinv {
        println!("\nPASS: the top-correlated subtree is the ProtectedInv-rooted Lorentz factor (|r|={top_r:.7} > 0.999), located purely by value.");
        Ok(())
    } else {
        Err(format!(
            "FAIL: top subtree k={top_k} op={top_name} |r|={top_r:.7} (need |r|>0.999 AND op==ProtectedInv)"
        ))
    }
}

/// Absolute Pearson correlation over rows where BOTH vectors are finite.
/// Returns (|r|, n_finite). A zero-variance column yields r=0 (not NaN).
#[cfg(feature = "gpu")]
fn abs_pearson(a: &[f64], b: &[f64]) -> (f64, usize) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            xs.push(x);
            ys.push(y);
        }
    }
    let n = xs.len();
    if n < 2 {
        return (0.0, n);
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(&ys) {
        let dx = x - mx;
        let dy = y - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return (0.0, n); // zero variance -> undefined correlation -> 0
    }
    ((sxy / denom).abs(), n)
}

/// Max relative error max|a-b|/|b| over rows where both are finite and b != 0.
#[cfg(feature = "gpu")]
fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() && y != 0.0 {
            let e = (x - y).abs() / y.abs();
            if e > m {
                m = e;
            }
        }
    }
    m
}
