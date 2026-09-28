// PATTERN VERIFY — the on-GPU reduction for typed snap.
//
// After the eval kernel leaves `n_expr * n_rows` predictions resident (candidate
// SUBTREE columns first, then PATTERN columns), this kernel scores every
// (subtree, pattern) pair WITHOUT moving per-row values to the host: one
// invocation per pair reduces over the rows to the five affine-fit sums and
// emits a single `1 - R^2`. Only `n_cand * n_pat` scalars ever leave the GPU.
//
// A pair's `1 - R^2` is the residual of the least-squares affine fit
// `subtree ~ a*pattern + b`, which is a function of the sums alone:
//   var_c = Scc - Sc*Sc/n;  var_p = Spp - Sp*Sp/n;  cov = Scp - Sc*Sp/n
//   R^2 = cov*cov / (var_c*var_p);  out = 1 - R^2
// Rows where either column is non-finite are dropped from all sums (a pole must
// not poison the fit), and `n` is that jointly-finite count.

struct Meta {
    n_cand:  u32,
    n_pat:   u32,
    n_rows:  u32,
    min_rows: u32,   // fewer jointly-finite rows than this -> the pair is unscorable (out = 2.0)
};

// preds[expr * n_rows + row]; experssions 0..n_cand are subtrees, then patterns.
@group(0) @binding(0) var<storage, read>       preds: array<f32>;
// out[cand * n_pat + pat] = 1 - R^2, or the SENTINEL 2.0 when unscorable.
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

    // f64 is unavailable in WGSL; accumulate in f32. n_rows is bounded (<= the
    // fit split, ~4000) and the values are model-scale, so f32 sums rank
    // correctly. The host re-scores the single winner in f64 before it is
    // trusted, exactly as the engine re-scores device finalists on the CPU.
    var n:   f32 = 0.0;
    var sc:  f32 = 0.0;
    var sp:  f32 = 0.0;
    var scc: f32 = 0.0;
    var spp: f32 = 0.0;
    var scp: f32 = 0.0;
    var row: u32 = 0u;
    loop {
        if (row >= cfg.n_rows) { break; }
        let cv = preds[c_base + row];
        let pv = preds[p_base + row];
        if (is_finite(cv) && is_finite(pv)) {
            n   = n + 1.0;
            sc  = sc + cv;
            sp  = sp + pv;
            scc = scc + cv * cv;
            spp = spp + pv * pv;
            scp = scp + cv * pv;
        }
        row = row + 1u;
    }

    if (n < f32(cfg.min_rows)) { out[pair] = UNSCORABLE; return; }
    let var_c = scc - sc * sc / n;
    let var_p = spp - sp * sp / n;
    if (var_c <= 0.0 || var_p <= 0.0) { out[pair] = UNSCORABLE; return; }   // a constant column
    let cov = scp - sc * sp / n;
    let r2 = clamp(cov * cov / (var_c * var_p), 0.0, 1.0);
    out[pair] = 1.0 - r2;
}
