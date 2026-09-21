//! A CPU-only 2D UMAP of a small dense dataset — the embedding the
//! synthetic-validation-row generator (`smogd`) lays its grid over.
//!
//! The stages are Andrew Morgan's own tested CPU reference functions, COPIED
//! from the qdrant workspace (`lib/udv23-umap/src/ultradim/umap/`) and chained
//! the way `pipeline.rs::fit_from_graph` chains them on its CPU path:
//!
//! ```text
//! exact kNN → smooth_knn (ρ, σ) → edge_weights → symmetrize → low_weight_cull
//!           → eps_schedule → build_directed_csr → deterministic_init → optimize_cpu
//! ```
//!
//! What came from where:
//!
//! | here                     | source file     | functions                                   |
//! |--------------------------|-----------------|---------------------------------------------|
//! | `mod rng`                | `rng.rs`        | `mix64`, `mix64_3`, `mix64_mod` (verbatim)  |
//! | `DOMAIN_UMAP`            | `mod.rs`        | the constant (verbatim)                     |
//! | `mod fuzzy`              | `fuzzy.rs`      | `CooEdge`, the three reference constants, `smooth_knn_target`, `global_mean_dist`, `validate_knn_dists`, `validate_edge_inputs`, `smooth_one_row`, `edge_weight_one`, `smooth_knn_cpu`, `edge_weights_cpu` |
//! | `mod symmetrize`         | `symmetrize.rs` | `validate_directed`, `canonical_sort_indices`, `scan_groups`, `emission_order`, `symmetrize_cpu`, `low_weight_cull`, `eps_schedule` |
//! | `AB_DEFAULT`             | `ab_fit.rs`     | the pinned `AB_CACHE` row for (0.1, 1.0)    |
//! | `mod optimize`           | `optimize.rs`   | `Y0_INIT_STREAM`, `deterministic_init`, `OptimizeParams`, `validate_inputs`, `DirectedCsr`, `build_directed_csr`, `attract_coeff`, `repulse_coeff`, `epoch_pass_cpu`, `run_epoch_loop_cpu`, `optimize_cpu` |
//! | `resolve_n_epochs`       | `pipeline.rs`   | `resolve_n_epochs` (auto branch): 500 for n ≤ 1M, else 300 |
//!
//! Only `mod rng` is character-for-character. Everywhere else the arithmetic
//! and its operation order are the source's; what changed is plumbing:
//!
//! - Index loops clippy flags (`needless_range_loop`) are iterators
//!   (`chunks_exact`, `enumerate`, `push` for indexed fills), doc comments are
//!   trimmed of the source's GPU/WGSL/fixture/work-package references, and the
//!   assert messages no longer say "angular" or cite the qdrant design doc.
//! - **Distances are Euclidean.** The source documents its graph payload as
//!   "angular" because its production data is spherical, but nothing in the
//!   copied code assumes it: the fuzzy and symmetrize stages only need finite,
//!   non-negative, ascending f32 distances (no cap at π, no normalisation), and
//!   the optimiser's spherical mode is a separate branch. The word "angular"
//!   is gone from the messages; the Euclidean branch is the only one kept.
//! - `fuzzy`: the `_sub` variants (row stride `k_graph` vs used width `k_used`,
//!   for over-fetched graphs) are folded to one `k`; ours is never over-fetched.
//! - `ab_fit`: `embed_2d` pins (min_dist, spread) = (0.1, 1.0), which the
//!   source serves bit-exactly from its cache without running the fit, so only
//!   that recorded pair is here, not the Levenberg–Marquardt fitter.
//! - `optimize`: spherical mode, the `coincident_kick` flag (off by default in
//!   the source: a coincident distinct negative contributes exactly 0), per-row
//!   mobility, the β-spring, checkpoints/resume and the sampled cross-entropy
//!   report are dropped; `optimize_cpu` returns the positions. `epoch_pass_cpu`
//!   took 17 flat arguments under a lint suppression — the per-fit constants
//!   are now one `EpochCtx`. The loop body is operation for operation the same
//!   (a row's edges are walked as slices rather than by index, for clippy).
//! - The init is the seeded random one (`deterministic_init`), not the
//!   clustering init the source pipeline uses.
//!
//! The kNN is written here (the source takes the graph as input): exact brute
//! force, self-excluded, ascending, ties to the lower row index.
//!
//! Contract violations BETWEEN the stages are bugs in this file and panic, as
//! in the source. Bad INPUT to [`embed_2d`] is an `Err`.

/// umap-learn's defaults, which ManifoldGridSampler uses.
const N_NEIGHBORS: usize = 15;

/// Design §9 auto-epoch rule (`pipeline.rs::resolve_n_epochs`, with its
/// `n_epochs = 0` = auto argument folded in): 500 for `N ≤ 1M`, 300 above.
fn resolve_n_epochs(n_rows: usize) -> u32 {
    if n_rows <= 1_000_000 {
        500
    } else {
        300
    }
}

/// `FitParams` defaults (design §9): learning rate, negatives per due edge, γ.
const LEARNING_RATE: f32 = 1.0;
const NEGATIVE_SAMPLE_RATE: u32 = 5;
const REPULSION_STRENGTH: f32 = 1.0;

/// `ab_fit.rs::AB_CACHE`, the (min_dist = 0.1, spread = 1.0) row: EXACTLY what
/// python umap-learn 0.5.12 returned for `find_ab_params(1.0, 0.1)`. The
/// source's `find_ab` returns these literals for the default pair.
const AB_DEFAULT: (f64, f64) = (1.57694346046584, 0.8950608779639974);

/// First 8 bytes big-endian of SHA-256("udv23_umap_v1") — `umap/mod.rs`.
const DOMAIN_UMAP: u64 = 0xE93E_983F_1BE6_9706;

/// 2D UMAP of `x` (row-major, `width` columns, Euclidean), deterministic from `seed`.
/// n_neighbors = 15, min_dist = 0.1, spread = 1.0 — umap-learn's defaults, which
/// ManifoldGridSampler uses. Returns row-major n × 2.
pub fn embed_2d(x: &[f64], width: usize, seed: u64) -> Result<Vec<[f64; 2]>, String> {
    if width == 0 {
        return Err("embed_2d: width must be >= 1".to_string());
    }
    if !x.len().is_multiple_of(width) {
        return Err(format!(
            "embed_2d: x has {} values, not a whole number of {width}-column rows",
            x.len()
        ));
    }
    let n_rows = x.len() / width;
    if n_rows < 3 {
        return Err(format!("embed_2d: needs at least 3 rows, got {n_rows}"));
    }
    if n_rows > u32::MAX as usize {
        return Err(format!("embed_2d: {n_rows} rows exceed the u32 ordinal space"));
    }
    if let Some(at) = x.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "embed_2d: non-finite input {} at row {}, column {}",
            x[at],
            at / width,
            at % width
        ));
    }
    // If n ≤ n_neighbors there are only n − 1 other rows to be neighbours.
    let k = N_NEIGHBORS.min(n_rows - 1);
    let n_epochs = resolve_n_epochs(n_rows);
    let (knn_ids, knn_dists) = exact_knn(x, n_rows, width, k)?;

    // ---- Stage F: fuzzy → symmetrize → culls → eps → CSR (fit_from_graph) ----
    let (rho, sigma) = fuzzy::smooth_knn_cpu(&knn_dists, n_rows, k);
    let directed = fuzzy::edge_weights_cpu(&knn_ids, &knn_dists, &rho, &sigma, n_rows, k);
    // symmetrize_cpu culls exact-zero directed weights at entry; the low-weight
    // cull then drops w < w_max/n_epochs BEFORE the eps schedule.
    let mut sym = symmetrize::symmetrize_cpu(&directed, n_rows);
    symmetrize::low_weight_cull(&mut sym, n_epochs);
    let eps = symmetrize::eps_schedule(&sym).0;
    let src: Vec<u32> = sym.iter().map(|e| e.i).collect();
    let dst: Vec<u32> = sym.iter().map(|e| e.j).collect();
    let csr = optimize::build_directed_csr(n_rows as u32, &src, &dst, &eps);

    // ---- Stage I: seeded random initial positions in [-10, 10) --------------
    let y0 = optimize::deterministic_init(n_rows as u32, 2, seed);

    // ---- Stage O: optimize ---------------------------------------------------
    let (a, b) = AB_DEFAULT;
    let opt = optimize::OptimizeParams {
        a: a as f32,
        b: b as f32,
        gamma: REPULSION_STRENGTH,
        lr: LEARNING_RATE,
        n_epochs,
        neg_rate: NEGATIVE_SAMPLE_RATE,
        umap_seed: seed,
        dim: 2,
    };
    let y = optimize::optimize_cpu(&y0, &csr, &opt);
    if let Some(at) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("embed_2d: the layout diverged (row {} is {})", at / 2, y[at]));
    }
    Ok(y.chunks_exact(2).map(|p| [f64::from(p[0]), f64::from(p[1])]).collect())
}

/// Exact brute-force Euclidean kNN: row-major `n × k` ids and distances,
/// SELF-EXCLUDED, ascending, ties to the lower row index. Distances are
/// accumulated in f64 and stored f32 (the cast is monotone, so ascending
/// survives it). O(n²·width); each row selects its k smallest and sorts only
/// those. `Err` when a distance does not fit an f32 — the input's range is too
/// wide to embed, and the fuzzy stage would rightly refuse an infinity.
fn exact_knn(x: &[f64], n_rows: usize, width: usize, k: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
    let mut ids = Vec::with_capacity(n_rows * k);
    let mut dists = Vec::with_capacity(n_rows * k);
    let mut cand: Vec<(f64, u32)> = Vec::with_capacity(n_rows - 1);
    for i in 0..n_rows {
        let xi = &x[i * width..(i + 1) * width];
        cand.clear();
        for j in (0..n_rows).filter(|&j| j != i) {
            let xj = &x[j * width..(j + 1) * width];
            let mut d2 = 0.0f64;
            for (p, q) in xi.iter().zip(xj) {
                let t = p - q;
                d2 += t * t;
            }
            cand.push((d2, j as u32));
        }
        // (distance, index) is a total order with no equal keys, so the
        // unstable select and sort are deterministic.
        let by_dist_then_row = |p: &(f64, u32), q: &(f64, u32)| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1));
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, by_dist_then_row);
        }
        let nearest = &mut cand[..k];
        nearest.sort_unstable_by(by_dist_then_row);
        for &(d2, j) in nearest.iter() {
            let d = d2.sqrt() as f32;
            if !d.is_finite() {
                return Err(format!(
                    "embed_2d: the distance between rows {i} and {j} overflows f32 — \
                     the input's range is too wide to embed"
                ));
            }
            ids.push(j);
            dists.push(d);
        }
    }
    Ok((ids, dists))
}

// ---------------------------------------------------------------------------
// rng.rs — the counter-based generator
// ---------------------------------------------------------------------------
mod rng {
    /// SplitMix64 finaliser (Stafford mix13) — bit-for-bit mirror of
    /// `rp_formula::mix64`. Do not alter the constants or the operation order.
    #[inline]
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        x
    }

    /// Triple-mix of `(domain, a, b, c)`:
    /// `mix64(mix64(mix64(domain ^ a) ^ b) ^ c)`.
    ///
    /// `domain` is a domain-separation constant, optionally pre-XORed with a
    /// per-fit seed by the caller (`DOMAIN_UMAP ^ umap_seed`). `a`, `b`, `c` are
    /// the counters (e.g. epoch, edge index, sample index). Pure and total: same
    /// inputs, same output, on every platform.
    #[inline]
    pub fn mix64_3(domain: u64, a: u64, b: u64, c: u64) -> u64 {
        mix64(mix64(mix64(domain ^ a) ^ b) ^ c)
    }

    /// [`mix64_3`] reduced to `[0, n)` via the Lemire multiply-shift:
    /// `((h as u128 * n as u128) >> 64) as u32`. `n` must be non-zero —
    /// `[0, 0)` is empty, so a zero modulus is a caller bug and fails loudly.
    #[inline]
    pub fn mix64_mod(domain: u64, a: u64, b: u64, c: u64, n: u32) -> u32 {
        assert!(n > 0, "mix64_mod: modulus n must be non-zero ([0, 0) is empty)");
        let h = mix64_3(domain, a, b, c);
        ((u128::from(h) * u128::from(n)) >> 64) as u32
    }
}

// ---------------------------------------------------------------------------
// fuzzy.rs — smooth-kNN calibration (ρ, σ) and the directed membership weights
// ---------------------------------------------------------------------------
mod fuzzy {
    /// Bisection break tolerance on `|psum − target|` — umap-learn's
    /// `SMOOTH_K_TOLERANCE = 1e-5`, compared in f32 here.
    pub const SMOOTH_K_TOLERANCE: f32 = 1e-5;

    /// σ floor scale — umap-learn's `MIN_K_DIST_SCALE = 1e-3`, applied to the
    /// row mean (ρ > 0) or the global distance mean (ρ = 0).
    pub const MIN_K_DIST_SCALE: f32 = 1e-3;

    /// Fixed bisection iteration count — the reference's `n_iter=64`.
    pub const SMOOTH_KNN_N_ITER: u32 = 64;

    /// One directed fuzzy edge `(i, j, a_ij)`.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct CooEdge {
        /// Source row ordinal.
        pub i: u32,
        /// Neighbour row ordinal (from `knn_ids`; never equal to `i` — the
        /// graph is self-excluded).
        pub j: u32,
        /// Directed membership strength `a_ij ∈ (0, 1]`.
        pub w: f32,
    }

    /// Bisection target `log₂(k)` (bandwidth = 1.0). Computed once in f64 and
    /// cast to f32.
    pub fn smooth_knn_target(k: usize) -> f32 {
        assert!(k >= 1, "smooth_knn_target: k must be >= 1");
        (k as f64).log2() as f32
    }

    /// Global mean distance over the `n_rows × k` matrix — the reference's
    /// `np.mean(distances)`, consumed only by the ρ = 0 floor branch (all-zero
    /// rows, e.g. heavily duplicated data). Sequential f64 accumulation in
    /// row-major ascending order, cast to f32 once.
    pub fn global_mean_dist(knn_dists: &[f32], n_rows: usize, k: usize) -> f32 {
        if n_rows == 0 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for &d in &knn_dists[..n_rows * k] {
            acc += f64::from(d);
        }
        (acc / (n_rows as f64 * k as f64)) as f32
    }

    /// Validate the kNN distance matrix contract: `n_rows × k` row-major,
    /// finite, non-negative, ascending per row (ties allowed — duplicates are
    /// legal). Violations are builder bugs and panic loudly, never a silent
    /// fallback.
    fn validate_knn_dists(knn_dists: &[f32], n_rows: usize, k: usize) {
        assert!(k >= 1, "umap fuzzy: k must be >= 1");
        assert!(
            n_rows <= u32::MAX as usize,
            "umap fuzzy: n_rows ({n_rows}) exceeds the u32 ordinal space"
        );
        let expected = n_rows.checked_mul(k).expect("umap fuzzy: n_rows * k overflows usize");
        assert_eq!(
            knn_dists.len(),
            expected,
            "umap fuzzy: knn_dists length {} != n_rows * k = {expected}",
            knn_dists.len()
        );
        for (i, row) in knn_dists.chunks_exact(k).enumerate() {
            let mut prev = 0.0f32;
            for (j, &d) in row.iter().enumerate() {
                assert!(
                    d.is_finite() && d >= 0.0,
                    "umap fuzzy: knn_dists[row {i}, col {j}] = {d} is not a finite non-negative \
                     distance"
                );
                assert!(
                    d >= prev,
                    "umap fuzzy: knn_dists row {i} not ascending at col {j}: {prev} then {d}"
                );
                prev = d;
            }
        }
    }

    /// Validate the edge-weight inputs on top of [`validate_knn_dists`]:
    /// `knn_ids` shape matches, every id is a valid ordinal, no self edges, and
    /// ρ/σ carry one entry per row.
    fn validate_edge_inputs(
        knn_ids: &[u32],
        knn_dists: &[f32],
        rho: &[f32],
        sigma: &[f32],
        n_rows: usize,
        k: usize,
    ) {
        validate_knn_dists(knn_dists, n_rows, k);
        assert_eq!(
            knn_ids.len(),
            knn_dists.len(),
            "umap fuzzy: knn_ids length {} != knn_dists length {}",
            knn_ids.len(),
            knn_dists.len()
        );
        assert_eq!(rho.len(), n_rows, "umap fuzzy: rho length {} != n_rows {n_rows}", rho.len());
        assert_eq!(sigma.len(), n_rows, "umap fuzzy: sigma length {} != n_rows {n_rows}", sigma.len());
        for (i, row) in knn_ids.chunks_exact(k).enumerate() {
            for (j, &id) in row.iter().enumerate() {
                assert!(
                    (id as usize) < n_rows,
                    "umap fuzzy: knn_ids[row {i}, col {j}] = {id} is outside the ordinal space \
                     [0, {n_rows})"
                );
                assert!(
                    id as usize != i,
                    "umap fuzzy: self edge at row {i}, col {j} — the graph is SELF-EXCLUDED; \
                     the graph builder is broken"
                );
            }
        }
    }

    /// Per-row smooth-kNN calibration → `(ρ, σ)`. `row` is one kNN row (`k`
    /// ascending distances).
    fn smooth_one_row(row: &[f32], target: f32, mean_dist_global: f32) -> (f32, f32) {
        let k_used = row.len();
        // ρ: first strictly-positive distance (ascending ⇒ smallest nonzero) —
        // the reference's local_connectivity = 1.0 semantics; 0.0 for an
        // all-zero row.
        let mut rho = 0.0f32;
        for &dj in row {
            if dj > 0.0 {
                rho = dj;
                break;
            }
        }
        // 64-iteration bisection, f32 state exactly like the reference's pinned
        // numba locals. hi's "unset" sentinel is f32::MAX (the reference's
        // NPY_FLOATMAX); while unset, the else-branch doubles mid.
        let mut lo = 0.0f32;
        let mut hi = f32::MAX;
        let mut mid = 1.0f32;
        for _ in 0..SMOOTH_KNN_N_ITER {
            let mut psum = 0.0f32;
            for &dj in &row[1..k_used] {
                let d = dj - rho;
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }
            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) * 0.5;
            } else {
                lo = mid;
                if hi >= f32::MAX {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) * 0.5;
                }
            }
        }
        // MIN_K_DIST_SCALE floor: row mean when ρ > 0, global mean when ρ = 0
        // (the reference's two branches). Row mean is a fixed-order sequential
        // f32 sum.
        let mut s = 0.0f32;
        for &dj in row {
            s += dj;
        }
        let mean_row = s / k_used as f32;
        let floor = if rho > 0.0 {
            MIN_K_DIST_SCALE * mean_row
        } else {
            MIN_K_DIST_SCALE * mean_dist_global
        };
        let sigma = if mid < floor { floor } else { mid };
        (rho, sigma)
    }

    /// One directed weight — the reference's `compute_membership_strengths`:
    /// `d − ρ ≤ 0` OR `σ = 0` ⇒ exactly 1.0 (never `exp(0/0) = NaN`), else
    /// `exp(−(d − ρ)/σ)`.
    fn edge_weight_one(d: f32, rho: f32, sigma: f32) -> f32 {
        let diff = d - rho;
        if diff <= 0.0 || sigma == 0.0 {
            1.0
        } else {
            (-(diff / sigma)).exp()
        }
    }

    /// Smooth-kNN calibration over the whole graph → `(rho, sigma)`, one entry
    /// per row. NOTE the return order `(rho, sigma)` — umap-learn returns
    /// `(sigmas, rhos)`, reversed.
    pub fn smooth_knn_cpu(knn_dists: &[f32], n_rows: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
        validate_knn_dists(knn_dists, n_rows, k);
        let target = smooth_knn_target(k);
        let mean_dist_global = global_mean_dist(knn_dists, n_rows, k);
        let mut rho_out = Vec::with_capacity(n_rows);
        let mut sigma_out = Vec::with_capacity(n_rows);
        for row in knn_dists.chunks_exact(k) {
            let (rho, sigma) = smooth_one_row(row, target, mean_dist_global);
            rho_out.push(rho);
            sigma_out.push(sigma);
        }
        (rho_out, sigma_out)
    }

    /// Directed fuzzy edge weights `a_ij = exp(−max(0, d_ij − ρ_i)/σ_i)` over
    /// the whole graph, emitted as COO `(i, j, a)` in DETERMINISTIC row-major
    /// order: entry `e = i·k + j` describes `(i, knn_ids[e], a(knn_dists[e]))`.
    /// Nothing is culled here: exactly `n_rows · k` directed edges.
    pub fn edge_weights_cpu(
        knn_ids: &[u32],
        knn_dists: &[f32],
        rho: &[f32],
        sigma: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Vec<CooEdge> {
        validate_edge_inputs(knn_ids, knn_dists, rho, sigma, n_rows, k);
        let mut edges = Vec::with_capacity(n_rows * k);
        for i in 0..n_rows {
            let base = i * k;
            for j in 0..k {
                edges.push(CooEdge {
                    i: i as u32,
                    j: knn_ids[base + j],
                    w: edge_weight_one(knn_dists[base + j], rho[i], sigma[i]),
                });
            }
        }
        edges
    }
}

// ---------------------------------------------------------------------------
// symmetrize.rs — fuzzy union, the low-weight cull, the epochs-per-sample schedule
// ---------------------------------------------------------------------------
mod symmetrize {
    use super::fuzzy::CooEdge;

    /// Validate the directed COO contract: ordinals in range, no self edges,
    /// membership strengths in `(0, 1]`. Violations are upstream bugs and panic
    /// loudly. Duplicate directed pairs are caught during the group scan.
    fn validate_directed(directed: &[CooEdge], n_rows: usize) {
        assert!(n_rows > 0, "umap symmetrize: n_rows must be > 0");
        assert!(
            n_rows <= u32::MAX as usize,
            "umap symmetrize: n_rows ({n_rows}) exceeds the u32 ordinal space"
        );
        assert!(
            !directed.is_empty(),
            "umap symmetrize: empty directed COO — the fuzzy stage emits n_rows·k edges, \
             so an empty input is a caller bug"
        );
        assert!(
            directed.len() <= u32::MAX as usize,
            "umap symmetrize: {} directed edges exceed the u32 edge-index space",
            directed.len()
        );
        for (e, edge) in directed.iter().enumerate() {
            assert!(
                (edge.i as usize) < n_rows && (edge.j as usize) < n_rows,
                "umap symmetrize: record {e} ({}, {}) out of range for n_rows={n_rows} — \
                 corrupt directed COO",
                edge.i,
                edge.j
            );
            assert!(
                edge.i != edge.j,
                "umap symmetrize: self edge at record {e} (row {}) — the graph is \
                 SELF-EXCLUDED; the fuzzy stage never emits one",
                edge.i
            );
            assert!(
                edge.w.is_finite() && edge.w > 0.0 && edge.w <= 1.0,
                "umap symmetrize: directed weight w[{e}] = {} outside (0, 1] — not a \
                 membership strength (fuzzy stage contract)",
                edge.w
            );
        }
    }

    /// Deterministic TOTAL canonical order: `(min(i,j), max(i,j), src)`. For a
    /// pair `(p, q)` with `p < q` this puts the forward edge (src = p) before the
    /// reverse (src = q).
    fn canonical_sort_indices(directed: &[CooEdge]) -> Vec<u32> {
        let mut order: Vec<u32> = (0..directed.len() as u32).collect();
        order.sort_unstable_by_key(|&k| {
            let e = &directed[k as usize];
            (e.i.min(e.j), e.i.max(e.j), e.i)
        });
        order
    }

    /// Walk a canonically ordered permutation and produce the group table:
    /// `group_head` = head position per canonical-pair group PLUS a final
    /// sentinel (`= E`, so `group_head[g+1] − group_head[g]` is the group size),
    /// and the canonical `(min, max)` pair per group. Panics loudly on duplicate
    /// directed edges.
    fn scan_groups(directed: &[CooEdge], order: &[u32]) -> (Vec<u32>, Vec<(u32, u32)>) {
        let e = order.len();
        let mut group_head: Vec<u32> = Vec::new();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut g0 = 0usize;
        while g0 < e {
            let e0 = &directed[order[g0] as usize];
            let (pmin, pmax) = (e0.i.min(e0.j), e0.i.max(e0.j));
            let mut g1 = g0 + 1;
            while g1 < e {
                let e1 = &directed[order[g1] as usize];
                if (e1.i.min(e1.j), e1.i.max(e1.j)) != (pmin, pmax) {
                    break;
                }
                g1 += 1;
            }
            let size = g1 - g0;
            assert!(
                size <= 2,
                "umap symmetrize: canonical pair ({pmin}, {pmax}) appears {size} times — \
                 duplicate directed edges in the input (a kNN row listed the same \
                 neighbour twice; graph-builder bug)"
            );
            if size == 2 {
                let e1 = &directed[order[g0 + 1] as usize];
                assert!(
                    e1.i == e0.j && e1.j == e0.i,
                    "umap symmetrize: duplicate directed edge ({}, {}) — the directed COO \
                     must be pair-unique per direction (graph-builder bug)",
                    e0.i,
                    e0.j
                );
            }
            group_head.push(g0 as u32);
            pairs.push((pmin, pmax));
            g0 = g1;
        }
        group_head.push(e as u32); // sentinel: group_head[g+1] − group_head[g] = size
        assert!(
            pairs.len().checked_mul(2).is_some_and(|d| d <= u32::MAX as usize),
            "umap symmetrize: 2·E_sym exceeds the u32 edge-index space"
        );
        (group_head, pairs)
    }

    /// Both-direction emission order: for each canonical pair `(p, q)` emit
    /// `(p, q)` and `(q, p)`, globally sorted by `(src, dst)` — also
    /// `build_directed_csr`'s internal sort order. All `(src, dst)` keys are
    /// distinct, so `sort_unstable` over the total key is deterministic.
    ///
    /// Returns `(out_src, out_dst, slot_fwd, slot_rev)`: the emitted key columns
    /// plus, per group, the output slot of its forward (`src = min`) and reverse
    /// (`src = max`) record.
    fn emission_order(pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let g_count = pairs.len();
        let mut recs: Vec<(u32, u32, u32)> = Vec::with_capacity(2 * g_count);
        for (g, &(p, q)) in pairs.iter().enumerate() {
            recs.push((p, q, g as u32));
            recs.push((q, p, g as u32));
        }
        recs.sort_unstable_by_key(|&(s, d, _)| (s, d));
        let mut out_src = Vec::with_capacity(recs.len());
        let mut out_dst = Vec::with_capacity(recs.len());
        let mut slot_fwd = vec![0u32; g_count];
        let mut slot_rev = vec![0u32; g_count];
        for (pos, &(s, d, g)) in recs.iter().enumerate() {
            out_src.push(s);
            out_dst.push(d);
            if s == pairs[g as usize].0 {
                slot_fwd[g as usize] = pos as u32;
            } else {
                slot_rev[g as usize] = pos as u32;
            }
        }
        (out_src, out_dst, slot_fwd, slot_rev)
    }

    /// Symmetrize a directed fuzzy COO into the both-direction symmetric COO:
    /// group by canonical pair under the deterministic total order
    /// `(min, max, src)`, combine `w = a + b − a·b` in f64 (`b = 0` for a missing
    /// reverse), store f32, and emit BOTH directions sorted by `(src, dst)`.
    pub fn symmetrize_cpu(directed: &[CooEdge], n_rows: usize) -> Vec<CooEdge> {
        // Reference eliminate_zeros() — the umap-learn `fuzzy_simplicial_set`
        // cull performed immediately before symmetrization. A far neighbour of a
        // tightly-calibrated row (σ floored small) can underflow
        // `exp(−(d−ρ)/σ)` to exactly +0.0 in f32, which is not a membership
        // strength. At least one edge per row always survives (each row's first
        // strictly-positive-distance neighbour has d − ρ = 0 ⇒ weight exactly
        // 1.0), so the cull cannot empty a row.
        let culled: Vec<CooEdge> = directed.iter().copied().filter(|e| e.w != 0.0).collect();
        validate_directed(&culled, n_rows);
        let directed = culled.as_slice();
        let order = canonical_sort_indices(directed);
        let (group_head, pairs) = scan_groups(directed, &order);
        let g_count = pairs.len();

        // f64 t-conorm per group (f64 accumulate, f32 store). Evaluation shape
        // (a + b) − (a·b) — the elementwise form of `W + W.T − W.multiply(W.T)`.
        let mut w_group = Vec::with_capacity(g_count);
        for g in 0..g_count {
            let h = group_head[g] as usize;
            let a = f64::from(directed[order[h] as usize].w);
            let b = if group_head[g + 1] - group_head[g] == 2 {
                f64::from(directed[order[h + 1] as usize].w)
            } else {
                0.0
            };
            w_group.push((a + b - a * b) as f32);
        }

        // Both-direction emission through the shared slot tables.
        let (out_src, out_dst, slot_fwd, slot_rev) = emission_order(&pairs);
        let mut out = vec![CooEdge { i: 0, j: 0, w: 0.0 }; 2 * g_count];
        for (g, &w) in w_group.iter().enumerate() {
            for slot in [slot_fwd[g] as usize, slot_rev[g] as usize] {
                out[slot] = CooEdge { i: out_src[slot], j: out_dst[slot], w };
            }
        }
        out
    }

    /// THE reference low-weight cull: drop every record with
    /// `w < w_max / n_epochs` — umap-learn's `simplicial_set_embedding` graph
    /// cull, applied BEFORE [`eps_schedule`]. Load-bearing: culled edges would
    /// get `eps > n_epochs` (schedule no-ops), and underflow-scale weights would
    /// overflow `w_max / w` to `+inf`, which `build_directed_csr` rightly
    /// rejects. The max-weight record always survives; a row whose every edge is
    /// culled simply never moves under attraction — reference semantics.
    pub fn low_weight_cull(sym: &mut Vec<CooEdge>, n_epochs: u32) {
        assert!(
            !sym.is_empty(),
            "umap symmetrize: low_weight_cull on an empty symmetric COO — caller bug"
        );
        assert!(
            n_epochs >= 1,
            "umap symmetrize: low_weight_cull needs n_epochs >= 1 (got {n_epochs})"
        );
        let mut w_max = f32::NEG_INFINITY;
        for (e, edge) in sym.iter().enumerate() {
            assert!(
                edge.w.is_finite() && edge.w > 0.0,
                "umap symmetrize: low_weight_cull weight w[{e}] = {} must be finite and > 0 \
                 (symmetrize contract)",
                edge.w
            );
            if edge.w > w_max {
                w_max = edge.w;
            }
        }
        let threshold = w_max / n_epochs as f32;
        // Reference culls strictly-below (`data < max/n_epochs`); keeping
        // `w >= threshold` is exactly that complement.
        sym.retain(|edge| edge.w >= threshold);
        debug_assert!(
            !sym.is_empty(),
            "the max-weight record survives its own threshold — cannot empty the set"
        );
    }

    /// Epochs-per-sample schedule over a symmetrized COO (the reference
    /// `make_epochs_per_sample` identity for w > 0): returns `(eps, w_max)` with
    /// `eps[e] = w_max / w[e]` aligned 1:1 with the emitted record order.
    pub fn eps_schedule(sym: &[CooEdge]) -> (Vec<f32>, f32) {
        assert!(
            !sym.is_empty(),
            "umap symmetrize: eps_schedule on an empty symmetric COO — nothing to schedule"
        );
        let mut w_max = f32::NEG_INFINITY;
        for (e, edge) in sym.iter().enumerate() {
            assert!(
                edge.w.is_finite() && edge.w > 0.0,
                "umap symmetrize: symmetric weight w[{e}] = {} must be finite and > 0 — \
                 eps = w_max/w would not be a valid schedule",
                edge.w
            );
            if edge.w > w_max {
                w_max = edge.w;
            }
        }
        let eps: Vec<f32> = sym.iter().map(|edge| w_max / edge.w).collect();
        (eps, w_max)
    }
}

// ---------------------------------------------------------------------------
// optimize.rs — seeded init, the directed sym-CSR, the Jacobi edge-sampled optimiser
// ---------------------------------------------------------------------------
mod optimize {
    use super::rng::{mix64_3, mix64_mod};
    use super::DOMAIN_UMAP;

    /// Stream tag separating the y0-init draws from every other UMAP mix64
    /// stream (the negative-sampling counters are (epoch, e, s) with
    /// epoch < 2^32, so a first-slot constant ≥ 2^32 can never collide).
    pub const Y0_INIT_STREAM: u64 = u64::from_le_bytes(*b"UMAP__Y0");

    /// Deterministic random initial positions in `[-10, 10)` (the umap-learn
    /// random-init range), one f32 per `(ordinal, dim_index)`:
    ///
    /// ```text
    /// h = mix64_3(DOMAIN_UMAP ^ umap_seed, Y0_INIT_STREAM, ordinal, d)
    /// y = f32(h >> 40) / 2^24 * 20 - 10
    /// ```
    ///
    /// Every operation is exact in f32, so the init is bit-identical across
    /// platforms.
    pub fn deterministic_init(n_rows: u32, dim: u32, umap_seed: u64) -> Vec<f32> {
        assert!(n_rows > 0, "deterministic_init: n_rows must be > 0");
        assert!(
            (1..=256).contains(&dim),
            "deterministic_init: dim must be in 1..=256, got {dim}",
        );
        let dom = DOMAIN_UMAP ^ umap_seed;
        let mut y = Vec::with_capacity(n_rows as usize * dim as usize);
        for i in 0..u64::from(n_rows) {
            for d in 0..u64::from(dim) {
                let h = mix64_3(dom, Y0_INIT_STREAM, i, d);
                let u = (h >> 40) as u32 as f32; // 24-bit integer — exact in f32
                let f = u / 16_777_216.0; // exact power-of-two divide -> [0, 1)
                y.push(f * 20.0 - 10.0);
            }
        }
        y
    }

    /// Fit parameters of the layout optimizer. `a`/`b` are the low-D kernel
    /// shape (cast to f32 — the epoch computes in f32).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OptimizeParams {
        /// Low-D kernel `a`.
        pub a: f32,
        /// Low-D kernel `b`.
        pub b: f32,
        /// Repulsion strength γ (default 1.0).
        pub gamma: f32,
        /// Initial learning rate (linear decay, default 1.0).
        pub lr: f32,
        /// Total epochs.
        pub n_epochs: u32,
        /// Negatives per due edge (default 5).
        pub neg_rate: u32,
        /// Per-fit seed, folded as `DOMAIN_UMAP ^ umap_seed`.
        pub umap_seed: u64,
        /// Output dimensionality.
        pub dim: u32,
    }

    impl OptimizeParams {
        /// Loud validation of the numeric domain — violations are caller bugs.
        pub fn validate(&self) {
            assert!(
                self.a.is_finite() && self.a > 0.0,
                "OptimizeParams: a must be finite and > 0, got {}",
                self.a,
            );
            assert!(
                self.b.is_finite() && self.b > 0.0,
                "OptimizeParams: b must be finite and > 0, got {}",
                self.b,
            );
            assert!(
                self.gamma.is_finite() && self.gamma >= 0.0,
                "OptimizeParams: gamma must be finite and >= 0, got {}",
                self.gamma,
            );
            assert!(
                self.lr.is_finite() && self.lr > 0.0,
                "OptimizeParams: lr must be finite and > 0, got {}",
                self.lr,
            );
            assert!(self.n_epochs >= 1, "OptimizeParams: n_epochs must be >= 1");
            assert!(
                (1..=256).contains(&self.dim),
                "OptimizeParams: dim must be in 1..=256, got {}",
                self.dim,
            );
        }
    }

    /// Input validation of the optimizer. Contract violations fail loudly —
    /// they are caller bugs, not runtime conditions.
    fn validate_inputs(y0: &[f32], csr: &DirectedCsr, params: &OptimizeParams) {
        params.validate();
        assert!(csr.n_rows > 0, "optimizer: CSR must cover at least one row");
        assert_eq!(
            y0.len(),
            csr.n_rows as usize * params.dim as usize,
            "optimizer: y0 length must be n_rows * dim",
        );
        assert_eq!(
            csr.row_start.len(),
            csr.n_rows as usize,
            "optimizer: row_start length must equal n_rows",
        );
        assert_eq!(
            csr.row_end.len(),
            csr.n_rows as usize,
            "optimizer: row_end length must equal n_rows",
        );
        assert_eq!(csr.col.len(), csr.eps.len(), "optimizer: col and eps must be aligned 1:1");
        assert!(
            !csr.col.is_empty(),
            "optimizer: the directed sym-CSR carries no edges — nothing to optimize",
        );
    }

    /// Directed symmetric CSR consumed by the optimizer: both directions of
    /// every symmetric edge present (what makes the epoch a pure gather),
    /// grouped by source row in ascending `(src, dst)` order.
    #[derive(Clone, Debug, PartialEq)]
    pub struct DirectedCsr {
        /// Number of rows N (also the negative-draw modulus).
        pub n_rows: u32,
        /// First out-edge index per row (len N).
        pub row_start: Vec<u32>,
        /// One-past-last out-edge index per row (len N).
        pub row_end: Vec<u32>,
        /// Destination ordinal per directed edge (len E_dir).
        pub col: Vec<u32>,
        /// `epochs_per_sample[e] = w_max / w[e]`, aligned 1:1 with `col`.
        pub eps: Vec<f32>,
    }

    /// Build the directed sym-CSR from a directed COO `(src, dst)` with its
    /// 1:1-aligned eps schedule. Deterministic stable order: sort by
    /// `(src, dst)`, carrying eps through the same permutation. Input violations
    /// fail loudly: length mismatches, out-of-range ordinals, self-loops,
    /// duplicate directed pairs, non-finite or non-positive eps, or more than
    /// u32::MAX edges.
    pub fn build_directed_csr(n_rows: u32, src: &[u32], dst: &[u32], eps: &[f32]) -> DirectedCsr {
        assert!(n_rows > 0, "build_directed_csr: n_rows must be > 0");
        assert_eq!(src.len(), dst.len(), "build_directed_csr: src/dst length mismatch");
        assert_eq!(
            src.len(),
            eps.len(),
            "build_directed_csr: eps must be aligned 1:1 with the COO records",
        );
        assert!(
            src.len() <= u32::MAX as usize,
            "build_directed_csr: {} edges exceed the u32 edge-index space",
            src.len(),
        );
        for k in 0..src.len() {
            assert!(
                src[k] < n_rows && dst[k] < n_rows,
                "build_directed_csr: record {k} ({}, {}) out of range for n_rows={n_rows}",
                src[k],
                dst[k],
            );
            assert!(
                src[k] != dst[k],
                "build_directed_csr: self-loop at record {k} (row {}) — symmetrized graphs have none",
                src[k],
            );
            assert!(
                eps[k].is_finite() && eps[k] > 0.0,
                "build_directed_csr: eps[{k}] = {} must be finite and > 0",
                eps[k],
            );
        }

        let mut order: Vec<u32> = (0..src.len() as u32).collect();
        order.sort_by_key(|&k| (src[k as usize], dst[k as usize])); // stable sort

        let mut col = Vec::with_capacity(src.len());
        let mut eps_sorted = Vec::with_capacity(src.len());
        let mut counts = vec![0u32; n_rows as usize];
        let mut prev: Option<(u32, u32)> = None;
        for &k in &order {
            let pair = (src[k as usize], dst[k as usize]);
            if prev == Some(pair) {
                panic!(
                    "build_directed_csr: duplicate directed edge ({}, {}) — symmetrized COO must be pair-unique",
                    pair.0, pair.1,
                );
            }
            prev = Some(pair);
            col.push(pair.1);
            eps_sorted.push(eps[k as usize]);
            counts[pair.0 as usize] += 1;
        }

        let mut row_start = Vec::with_capacity(n_rows as usize);
        let mut row_end = Vec::with_capacity(n_rows as usize);
        let mut cum = 0u32;
        for &count in &counts {
            row_start.push(cum);
            cum += count;
            row_end.push(cum);
        }
        DirectedCsr { n_rows, row_start, row_end, col, eps: eps_sorted }
    }

    /// Attractive coefficient `g_a / (y_i - y_j)`:
    /// `−2ab·(d²)^(b−1) / (1 + a·(d²)^b)` — left-to-right products, `b_m1`
    /// hoisted. Caller guards `d2 > 0`.
    fn attract_coeff(a: f32, b: f32, b_m1: f32, d2: f32) -> f32 {
        let pa = d2.powf(b_m1);
        let pb = d2.powf(b);
        let num = -2.0 * a * b * pa;
        let den = 1.0 + a * pb;
        num / den
    }

    /// Repulsive coefficient `g_r / (y_i - y_n)`:
    /// `2γb / ((0.001 + d²)(1 + a·(d²)^b))`. Caller guards `dn2 > 0` (the
    /// `dn2 == 0` contribution is exactly 0).
    fn repulse_coeff(a: f32, b: f32, gamma: f32, dn2: f32) -> f32 {
        let pbn = dn2.powf(b);
        let gr_num = 2.0 * gamma * b;
        let den1 = 0.001 + dn2;
        let den2 = 1.0 + a * pbn;
        gr_num / (den1 * den2)
    }

    /// What every epoch of one fit shares — the source passed these as flat
    /// scalars.
    struct EpochCtx {
        dim: usize,
        a: f32,
        b: f32,
        b_m1: f32,
        gamma: f32,
        neg_rate: u32,
        dom: u64,
    }

    /// One Jacobi epoch over all rows. `y_in` is read-only, `y_out` fully
    /// rewritten — no aliasing, bit-reproducible.
    fn epoch_pass_cpu(
        y_in: &[f32],
        y_out: &mut [f32],
        next_due: &mut [f32],
        csr: &DirectedCsr,
        ctx: &EpochCtx,
        epoch: u32,
        alpha: f32,
    ) {
        let EpochCtx { dim, a, b, b_m1, gamma, neg_rate, dom } = *ctx;
        let epoch_f = epoch as f32;
        let mut acc = vec![0.0f32; dim];
        for i in 0..csr.n_rows as usize {
            for slot in acc.iter_mut() {
                *slot = 0.0;
            }
            let rs = csr.row_start[i] as usize;
            let re = csr.row_end[i] as usize;
            // The row's out-edges, walked as slices; `e` is the edge's global
            // index, which the negative draws are keyed by.
            let edges = csr.col[rs..re].iter().zip(&csr.eps[rs..re]);
            for (at, (due, (&col, &eps))) in next_due[rs..re].iter_mut().zip(edges).enumerate() {
                if *due > epoch_f {
                    continue;
                }
                let e = rs + at;
                let j = col as usize;
                let mut d2 = 0.0f32;
                for d in 0..dim {
                    let t = y_in[i * dim + d] - y_in[j * dim + d];
                    d2 += t * t;
                }
                if d2 > 0.0 {
                    let ga = attract_coeff(a, b, b_m1, d2);
                    for (d, slot) in acc.iter_mut().enumerate() {
                        let contrib = ga * (y_in[i * dim + d] - y_in[j * dim + d]);
                        *slot += contrib.clamp(-4.0, 4.0);
                    }
                }
                *due += eps; // source row owns this edge's schedule
                for s in 0..neg_rate {
                    // Counter order: (epoch, edge, sample).
                    let nn = mix64_mod(dom, u64::from(epoch), e as u64, u64::from(s), csr.n_rows);
                    if nn as usize == i {
                        continue; // self-draw skipped, not redrawn
                    }
                    let mut dn2 = 0.0f32;
                    for d in 0..dim {
                        let t = y_in[i * dim + d] - y_in[nn as usize * dim + d];
                        dn2 += t * t;
                    }
                    if dn2 > 0.0 {
                        let gr = repulse_coeff(a, b, gamma, dn2);
                        for (d, slot) in acc.iter_mut().enumerate() {
                            let contrib = gr * (y_in[i * dim + d] - y_in[nn as usize * dim + d]);
                            *slot += contrib.clamp(-4.0, 4.0);
                        }
                    }
                }
            }
            for (d, &g) in acc.iter().enumerate() {
                y_out[i * dim + d] = y_in[i * dim + d] + alpha * g;
            }
        }
    }

    /// The CPU epoch loop: epochs `0..n_epochs` from `(y, next_due)`, learning
    /// rate decaying linearly to zero.
    fn run_epoch_loop_cpu(
        mut y_cur: Vec<f32>,
        mut next_due: Vec<f32>,
        csr: &DirectedCsr,
        params: &OptimizeParams,
    ) -> Vec<f32> {
        let ctx = EpochCtx {
            dim: params.dim as usize,
            a: params.a,
            b: params.b,
            b_m1: params.b - 1.0,
            gamma: params.gamma,
            neg_rate: params.neg_rate,
            dom: DOMAIN_UMAP ^ params.umap_seed,
        };
        let mut y_next = vec![0.0f32; y_cur.len()];
        for epoch in 0..params.n_epochs {
            let alpha = params.lr * (1.0 - (epoch as f32) / (params.n_epochs as f32));
            epoch_pass_cpu(&y_cur, &mut y_next, &mut next_due, csr, &ctx, epoch, alpha);
            std::mem::swap(&mut y_cur, &mut y_next); // Jacobi swap after the full pass
        }
        y_cur
    }

    /// CPU reference optimizer — THE semantic ground truth of stage O.
    /// Deterministic and bit-reproducible run-to-run. Returns the final
    /// positions, N × dim row-major in ordinal order.
    pub fn optimize_cpu(y0: &[f32], csr: &DirectedCsr, params: &OptimizeParams) -> Vec<f32> {
        validate_inputs(y0, csr, params);
        // next_due init = eps[e].
        run_epoch_loop_cpu(y0.to_vec(), csr.eps.clone(), csr, params)
    }
}

#[cfg(test)]
mod tests {
    use super::rng::mix64_3;
    use super::*;
    use std::time::Instant;

    /// Uniform in [0, 1), keyed by (stream, row, column).
    fn unit(stream: u64, i: u64, c: u64) -> f64 {
        (mix64_3(0xF011E5, stream, i, c) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gaussian-ish, mean 0, sd 1: twelve uniforms summed (Irwin–Hall).
    fn gaussish(stream: u64, i: u64, c: u64) -> f64 {
        (0..12).map(|t| unit(stream, i, c * 12 + t)).sum::<f64>() - 6.0
    }

    /// Three blobs of `per` points in 4D, centres 12 apart, sd 1; row r is in blob r % 3.
    fn blobs(per: usize) -> (Vec<f64>, Vec<usize>) {
        let centres = [[0.0, 0.0, 0.0, 0.0], [12.0, 0.0, 12.0, 0.0], [0.0, 12.0, 0.0, 12.0]];
        let mut x = Vec::new();
        let mut label = Vec::new();
        for r in 0..3 * per {
            let blob = r % 3;
            for (c, centre) in centres[blob].iter().enumerate() {
                x.push(centre + gaussish(1, r as u64, c as u64));
            }
            label.push(blob);
        }
        (x, label)
    }

    fn flat(y: &[[f64; 2]]) -> Vec<f64> {
        y.iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    #[test]
    fn same_seed_is_bit_identical_and_another_seed_differs() {
        let (x, _) = blobs(40);
        let bits = |y: Vec<[f64; 2]>| -> Vec<u64> { flat(&y).iter().map(|v| v.to_bits()).collect() };
        let a = bits(embed_2d(&x, 4, 7).unwrap());
        let b = bits(embed_2d(&x, 4, 7).unwrap());
        let c = bits(embed_2d(&x, 4, 8).unwrap());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn separated_blobs_stay_separated() {
        let (x, label) = blobs(100);
        let y = embed_2d(&x, 4, 1).unwrap();
        let (nearest, _) = exact_knn(&flat(&y), y.len(), 2, 1).unwrap();
        let own = nearest.iter().enumerate().filter(|&(i, &j)| label[i] == label[j as usize]).count();
        let share = own as f64 / y.len() as f64;
        eprintln!("blobs: nearest 2D neighbour in own blob for {:.1}% of points", 100.0 * share);
        assert!(share >= 0.95, "only {share:.3} of points kept their blob");
    }

    #[test]
    fn curved_sheet_keeps_its_neighbourhoods() {
        let n = 600usize;
        let mut x = Vec::with_capacity(n * 3);
        for i in 0..n as u64 {
            let (u, v) = (unit(2, i, 0), unit(2, i, 1));
            x.extend([u, v, 0.3 * (std::f64::consts::PI * u).sin()]);
        }
        let y = embed_2d(&x, 3, 3).unwrap();
        let k = 10;
        let (high, _) = exact_knn(&x, n, 3, k).unwrap();
        let (low, _) = exact_knn(&flat(&y), n, 2, k).unwrap();
        let mut shared = 0usize;
        for (h, l) in high.chunks_exact(k).zip(low.chunks_exact(k)) {
            shared += h.iter().filter(|id| l.contains(id)).count();
        }
        let overlap = shared as f64 / (n * k) as f64;
        eprintln!("sheet: mean 10-NN overlap {overlap:.3}");
        assert!(overlap >= 0.5, "mean 10-NN overlap {overlap:.3} < 0.5");
    }

    #[test]
    fn knn_on_a_line_by_hand() {
        // Points on a line at 0, 1, 3, 7, and a duplicate of the 3.
        let x = [0.0, 1.0, 3.0, 7.0, 3.0];
        let (ids, dists) = exact_knn(&x, 5, 1, 2).unwrap();
        // Rows 2 and 4 coincide: each is the other's nearest, at 0. Rows 0, 1
        // and 3 are equally far from both — the tie goes to the lower index, 2.
        assert_eq!(ids, vec![1, 2, 0, 2, 4, 1, 2, 4, 2, 1]);
        assert_eq!(dists, vec![1.0, 3.0, 1.0, 2.0, 0.0, 2.0, 4.0, 4.0, 0.0, 2.0]);
    }

    #[test]
    fn bad_input_is_an_error() {
        let ok = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(embed_2d(&ok, 0, 1).is_err());
        assert!(embed_2d(&ok[..4], 2, 1).is_err()); // 2 rows
        assert!(embed_2d(&ok[..5], 2, 1).is_err()); // ragged
        assert!(embed_2d(&[0.0, 1.0, f64::NAN, 3.0, 4.0, 5.0], 2, 1).is_err());
        assert!(embed_2d(&[0.0, 1.0, f64::INFINITY, 3.0, 4.0, 5.0], 2, 1).is_err());
        assert!(embed_2d(&[-1e300, 0.0, 1e300], 1, 1).is_err()); // distance overflows f32
    }

    #[test]
    fn few_rows_and_duplicate_rows_embed() {
        // n ≤ n_neighbors uses k = n − 1; identical rows have all-zero distances.
        let y = embed_2d(&[0.0, 1.0, 2.0], 1, 1).unwrap(); // the smallest: n = 3, k = 2
        assert_eq!(y.len(), 3);
        let y = embed_2d(&[0.0, 1.0, 2.0, 3.0], 1, 1).unwrap();
        assert_eq!(y.len(), 4);
        let y = embed_2d(&[5.0; 40], 2, 1).unwrap();
        assert_eq!(y.len(), 20);
        assert!(flat(&y).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn five_thousand_by_six_completes() {
        let (n, width) = (5_000usize, 6usize);
        let x: Vec<f64> = (0..(n * width) as u64).map(|t| unit(3, t / width as u64, t % width as u64)).collect();
        let t0 = Instant::now();
        let y = embed_2d(&x, width, 11).unwrap();
        eprintln!("embed_2d 5000 x 6: {:.2} s", t0.elapsed().as_secs_f64());
        assert_eq!(y.len(), n);
        assert!(flat(&y).iter().all(|v| v.is_finite()));
    }
}
