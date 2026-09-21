// Snap, stage 3: the guard. "Snap proposes, the R² guard disposes": each
// expression's original and its VARIANTS grafted variants have been evaluated
// on the guarded rows by the evaluator, earlier in the same command buffer;
// this kernel reduces each to R² and keeps a variant only if R² drops by no
// more than the tolerance. The CPU twin is `snap_guard.rs::guard_predictions`.
//
// R² is of the expression AS IT IS — no scale or offset is re-fitted — over
// rows [row_lo, row_hi): 1 - sum(((y - p) * inv_scale)^2), inv_scale =
// 1 / sqrt(sum((y - mean y)^2)) from the host in f64. Scaling the residual
// BEFORE it is squared keeps a target near 1e-34 (h, hbar) or 1e30 inside f32.
// Sums are Neumaier-compensated in row order, as in score.wgsl.
//
// The squares and the sum are TWO passes: `guard_residuals` turns every
// prediction into its scaled squared residual in place, `guard_main` only adds
// them. A product that feeds a sum in one kernel is open to fusing into a
// multiply-add, which the host twin does not do; with the product stored first
// there is nothing to fuse. A last-bit difference in R² is not harmless here: a
// tie between two variants on "the higher R²" went the other way (measured).
//
// Nothing here is a final decision: the host re-derives every KEPT variant in
// f64 before it is reported or written into a gene.

struct GuardCfg {
    n_expr: u32,
    n_rows: u32,
    row_lo: u32,
    row_hi: u32,
    // Invocations per dispatch row, for the 2-D dispatch.
    stride: u32,
    r2_drop_tol: f32,
    inv_scale: f32,
    // Forms in the block `guard_residuals` is bound to.
    n_forms: u32,
    // 0, which the compiler cannot know: see `add`.
    zero: u32,
};

const VARIANTS: u32 = 4u;
const VINFO_STRIDE: u32 = 4u;
const VERDICT_STRIDE: u32 = 8u;
const NONE: u32 = 0xffffffffu;
const NAN_BITS: u32 = 0x7fc00000u;
const GRAFTED: u32 = 1u;

const KEPT: u32 = 0u;
const NO_VARIANT: u32 = 1u;
const R2_DROPPED: u32 = 2u;
const NOT_FINITE: u32 = 3u;
const REFUSED: u32 = 4u;
const NO_BASELINE: u32 = 5u;

@group(0) @binding(0) var<storage, read>       len_in:    array<u32>;
// Per variant: (length, status, sites grafted, atoms in the expression).
@group(0) @binding(1) var<storage, read>       vinfo:     array<u32>;
// Per variant: its length if GRAFTED, else 0 — the evaluator's `lengths`.
@group(0) @binding(2) var<storage, read_write> var_len:   array<u32>;
@group(0) @binding(3) var<uniform>             gcfg:      GuardCfg;
// Expression-major, n_rows each — originals; variants: the scaled squared
// residuals `guard_residuals` left where the evaluator's predictions were.
@group(0) @binding(4) var<storage, read>       sq_org:    array<f32>;
@group(0) @binding(5) var<storage, read>       sq_var:    array<f32>;
@group(0) @binding(6) var<storage, read>       y:         array<f32>;
// Per expression: (slot or NONE, status, R² original, R² per variant, 0).
@group(0) @binding(7) var<storage, read_write> verdicts:  array<u32>;
// One block of predictions, originals or variants, squared in place.
@group(0) @binding(8) var<storage, read_write> block:     array<f32>;

// Before the evaluation: a slot that is not GRAFTED gets length 0, which the
// evaluator answers with NaN without walking it.
@compute @workgroup_size(64, 1, 1)
fn guard_lengths(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y * gcfg.stride + gid.x;
    if (t >= gcfg.n_expr * VARIANTS) { return; }
    var n: u32 = 0u;
    if (vinfo[t * VINFO_STRIDE + 1u] == GRAFTED) { n = vinfo[t * VINFO_STRIDE]; }
    var_len[t] = n;
}

fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

// A float as the compiler must take it: through an integer XOR with a zero
// that arrives in the uniform block, so nothing can be reassociated across it.
fn keep(v: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(v) ^ gcfg.zero);
}

// Neumaier's compensated addition: (sum, compensation) += v.
//
// The compensation is ALGEBRAICALLY zero — (s - (s + v)) + v — and a compiler
// that treats floats as reals reassociates it away: written plainly, this
// device returned the UNcompensated sum, bit for bit (measured against the host
// twin with its compensation switched off). Every intermediate is therefore
// `keep`-ed, and the rounding error is computed as written.
fn add(s: vec2<f32>, v: f32) -> vec2<f32> {
    let t = keep(s.x + v);
    var c = s.y;
    if (abs(s.x) >= abs(v)) {
        c = keep(c + keep(keep(s.x - t) + v));
    } else {
        c = keep(c + keep(keep(v - t) + s.x));
    }
    return vec2<f32>(t, c);
}

// After the evaluation: prediction -> ((y - p) * inv_scale)^2, in place. A
// prediction that is not finite stays so (NaN and inf survive the arithmetic),
// and so does a residual whose square leaves f32.
@compute @workgroup_size(64, 1, 1)
fn guard_residuals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * gcfg.stride + gid.x;
    if (i >= gcfg.n_forms * gcfg.n_rows) { return; }
    let d = (y[i % gcfg.n_rows] - block[i]) * gcfg.inv_scale;
    block[i] = d * d;
}

// (R², 1.0 if the squared residual is finite on every guarded row) of the form
// whose block starts at `at` — among the variants, or the originals.
fn r_squared(variant: bool, at: u32) -> vec2<f32> {
    var ss = vec2<f32>(0.0, 0.0);
    var ok: f32 = 1.0;
    for (var row = gcfg.row_lo; row < gcfg.row_hi; row = row + 1u) {
        var sq = sq_org[at + row];
        if (variant) { sq = sq_var[at + row]; }
        if (finite(sq)) {
            ss = add(ss, sq);
        } else {
            ok = 0.0;
        }
    }
    return vec2<f32>(1.0 - keep(ss.x + ss.y), ok);
}

@compute @workgroup_size(64, 1, 1)
fn guard_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.y * gcfg.stride + gid.x;
    if (e >= gcfg.n_expr) { return; }
    let out = e * VERDICT_STRIDE;
    verdicts[out] = NONE;
    for (var k = 2u; k < VERDICT_STRIDE - 1u; k = k + 1u) {
        verdicts[out + k] = NAN_BITS;
    }
    verdicts[out + 7u] = 0u;
    if (len_in[e] == 0u) {
        verdicts[out + 1u] = REFUSED;
        return;
    }
    let original = r_squared(false, e * gcfg.n_rows);
    if (original.y == 0.0 || !finite(original.x)) {
        verdicts[out + 1u] = NO_BASELINE;
        return;
    }
    verdicts[out + 2u] = bitcast<u32>(original.x);
    let floor_r2 = original.x - gcfg.r2_drop_tol;

    // Fewest nodes; then the higher R²; then the lower slot (strict
    // comparisons in slot order).
    var best: u32 = NONE;
    var best_nodes: u32 = 0u;
    var best_r2: f32 = 0.0;
    var grafted: u32 = 0u;
    var dropped: u32 = 0u;
    for (var v = 0u; v < VARIANTS; v = v + 1u) {
        let t = e * VARIANTS + v;
        if (vinfo[t * VINFO_STRIDE + 1u] != GRAFTED) { continue; }
        grafted = grafted + 1u;
        let r = r_squared(true, t * gcfg.n_rows);
        if (r.y == 0.0) { continue; }
        verdicts[out + 3u + v] = bitcast<u32>(r.x);
        // A NaN or infinite R² is told by its bits, not by a comparison a
        // fast-math compiler may turn round.
        if (!finite(r.x) || r.x < floor_r2) {
            dropped = dropped + 1u;
            continue;
        }
        let nodes = vinfo[t * VINFO_STRIDE];
        if (best == NONE || nodes < best_nodes || (nodes == best_nodes && r.x > best_r2)) {
            best = v;
            best_nodes = nodes;
            best_r2 = r.x;
        }
    }
    verdicts[out] = best;
    if (best != NONE) {
        verdicts[out + 1u] = KEPT;
    } else if (grafted == 0u) {
        verdicts[out + 1u] = NO_VARIANT;
    } else if (dropped != 0u) {
        verdicts[out + 1u] = R2_DROPPED;
    } else {
        verdicts[out + 1u] = NOT_FINITE;
    }
}
