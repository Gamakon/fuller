// PATTERN VERIFY — the on-GPU residual test for typed snap.
//
// After the eval kernel leaves `n_expr * n_rows` predictions resident (candidate
// SUBTREE columns first, then PATTERN columns), this kernel scores every
// (subtree, pattern) pair WITHOUT moving per-row values to the host: one
// invocation per pair reduces over the rows to the RATIO subtree/pattern and
// emits its coefficient of variation. Only `n_cand * n_pat` scalars leave the GPU.
//
// This is the whole-output residual test ("y / template is simple") applied per
// subtree: if subtree/pattern is CONSTANT across the rows, the subtree IS the
// pattern up to a single scale factor (the outer model supplies the scale). The
// score is std(ratio)/|mean(ratio)| in [0, inf): 0 = perfectly constant ratio =
// the pattern; larger = not this shape. No affine fit, no intercept — a pattern
// is a MULTIPLICATIVE factor, matched by a flat ratio, not by a regression.

struct Meta {
    n_cand:  u32,
    n_pat:   u32,
    n_rows:  u32,
    min_rows: u32,   // fewer usable rows than this -> the pair is unscorable (out = 2.0)
};

// preds[expr * n_rows + row]; expressions 0..n_cand are subtrees, then patterns.
@group(0) @binding(0) var<storage, read>       preds: array<f32>;
// out[cand * n_pat + pat] = coefficient of variation of the ratio, or the
// SENTINEL 2.0 when unscorable (too few usable rows, or a ~0 mean ratio).
@group(0) @binding(1) var<storage, read_write> out:   array<f32>;
@group(0) @binding(2) var<uniform>             cfg:   Meta;

const UNSCORABLE: f32 = 2.0;
fn is_finite(x: f32) -> bool { return abs(x) <= 3.4028235e38; }

@compute @workgroup_size(64, 1, 1)
fn verify_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    let total = cfg.n_cand * cfg.n_pat;
    if (pair >= total) { return; }

    let cand = pair / cfg.n_pat;
    let pat  = pair % cfg.n_pat;
    let c_base = cand * cfg.n_rows;
    let p_base = (cfg.n_cand + pat) * cfg.n_rows;

    // THE RESIDUAL TEST, per subtree: is the RATIO subtree/pattern constant over
    // the rows? A constant ratio means the subtree IS the pattern up to a single
    // scale factor (the outer model supplies that scale) — the same "y / template
    // is simple" test the whole-output residual path uses, applied to a subtree.
    // NO affine fit, NO intercept. The score is the coefficient of variation of
    // the ratio, std(r)/|mean(r)|, in [0, inf): 0 = perfectly constant ratio =
    // the pattern; larger = not this shape.
    //
    // f64 is unavailable in WGSL; accumulate the ratio's mean and second moment
    // in f32. n_rows is bounded (~4000) and values are model-scale, so f32 ranks
    // correctly; the host re-scores the single winner in f64 before trusting it.
    var n:  f32 = 0.0;
    var sr: f32 = 0.0;   // sum of ratios
    var srr: f32 = 0.0;  // sum of ratio^2
    var row: u32 = 0u;
    loop {
        if (row >= cfg.n_rows) { break; }
        let cv = preds[c_base + row];
        let pv = preds[p_base + row];
        // Drop a row where either side is non-finite OR the pattern is ~0 (the
        // ratio is meaningless there, not evidence against the shape).
        if (is_finite(cv) && is_finite(pv) && abs(pv) > 1e-20) {
            let r = cv / pv;
            if (is_finite(r)) {
                n   = n + 1.0;
                sr  = sr + r;
                srr = srr + r * r;
            }
        }
        row = row + 1u;
    }

    if (n < f32(cfg.min_rows)) { out[pair] = UNSCORABLE; return; }
    let mean = sr / n;
    let variance = max(srr / n - mean * mean, 0.0);
    // A ratio centred on ~0 has no meaningful scale (subtree ~ 0 everywhere):
    // reject rather than divide by a near-zero mean.
    if (abs(mean) < 1e-20) { out[pair] = UNSCORABLE; return; }
    out[pair] = sqrt(variance) / abs(mean);   // coefficient of variation
}
