// Scoring on the device: every (chromosome, linker, wrapper) candidate linked,
// wrapped, scaled by least squares on the train rows, and its errors reduced
// over the train / validation / edge rows — from the evaluator's predictions
// where they already are. chrom_score.rs is the f64 definition; this is the
// same procedure in f32 with compensated sums, used to RANK. Whatever is
// reported or stops a fit is re-scored in f64 on the host.
//
// One thread per (chromosome, linker); its three wrappers share the row loop.
// A thread writes only its own 3 x 9 output slots. Reductions are fixed-order
// sums (row order), so a result does not depend on scheduling.

struct Meta {
    n_chromosomes: u32,
    genes_per: u32,
    n_rows: u32,
    n_train: u32,
    n_val: u32,
    n_extrap: u32,
    y_mean_train: f32,
    // 0, which the compiler cannot know: see `keep`.
    zero: u32,
}

@group(0) @binding(0) var<uniform> sp: Meta;
@group(0) @binding(1) var<storage, read> preds: array<f32>;       // gene-major, n_rows each
@group(0) @binding(2) var<storage, read> gene_ok: array<u32>;
@group(0) @binding(3) var<storage, read> chromosomes: array<u32>; // genes_per gene indices each
@group(0) @binding(4) var<storage, read> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;

const N_LINKERS: u32 = 3u;   // avg, mul, add — chrom_score's order in the engine
const N_WRAPPERS: u32 = 3u;  // identity, log_abs, sqrt_abs
const WIDTH: u32 = 10u;      // a, b, mse_t, mse_v, max_err_v, mse_e, mae_t, mae_v, mae_e, redundancy
const LOO_STRIDE: u32 = 4u;  // leave-one-gene-out reads every 4th train row
const NAN_BITS: u32 = 0x7FC00000u;
const CONSTANT_REL_TOL: f32 = 2e-6;

fn finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

fn linked(c: u32, linker: u32, row: u32) -> f32 {
    var acc = select(0.0, 1.0, linker == 1u);
    for (var g = 0u; g < sp.genes_per; g = g + 1u) {
        let v = preds[chromosomes[c * sp.genes_per + g] * sp.n_rows + row];
        if (linker == 1u) {
            acc = keep(acc * v);
        } else {
            acc = keep(acc + v);
        }
    }
    if (linker == 0u) {
        acc = div(acc, f32(sp.genes_per));
    }
    return acc;
}

// The linked value with gene `skip` replaced by `held` (its mean): what the
// model computes when that gene is switched off.
fn linked_without(c: u32, linker: u32, row: u32, skip: u32, held: f32) -> f32 {
    var acc = select(0.0, 1.0, linker == 1u);
    for (var g = 0u; g < sp.genes_per; g = g + 1u) {
        var v = preds[chromosomes[c * sp.genes_per + g] * sp.n_rows + row];
        if (g == skip) {
            v = held;
        }
        if (linker == 1u) {
            acc = keep(acc * v);
        } else {
            acc = keep(acc + v);
        }
    }
    if (linker == 0u) {
        acc = div(acc, f32(sp.genes_per));
    }
    return acc;
}

// sqrt(x), CORRECTLY ROUNDED, the way `div` is: the device's root s is a first
// guess, x - s*s is formed exactly and s corrected by residual / 2s.
fn root(x: f32) -> f32 {
    let s = keep(sqrt(x));
    if (s == 0.0 || !finite(s)) {
        return s;
    }
    let p = keep(s * s);
    let sh = high(s);
    let sl = keep(s - sh);
    let e = keep(keep(sl * sl) - keep(keep(keep(p - keep(sh * sh)) - keep(sl * sh)) - keep(sh * sl)));
    let r = keep(keep(x - p) - e);
    if (!finite(r)) {
        return s;
    }
    return keep(s + keep(r / keep(2.0 * s)));
}

fn wrapped(v: f32, w: u32) -> f32 {
    if (w == 1u) {
        return log(abs(v) + 1e-12);
    }
    if (w == 2u) {
        return root(abs(v));
    }
    return v;
}

// A float as the compiler must take it: through an integer XOR with a zero
// that arrives in the uniform block, so nothing can be reassociated across it
// and a product cannot be fused into the sum it feeds (snap_guard.wgsl's rule).
fn keep(v: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(v) ^ sp.zero);
}

// Neumaier's compensated addition: (sum, compensation) += v.
//
// The compensation is ALGEBRAICALLY zero — (s - (s + v)) + v — and a compiler
// that treats floats as reals reassociates it away. Written plainly, this
// device's scale for y = 2x over 40,000 rows was 1.9999967 and 1 - R² 1e-9,
// where the sums as written give 2 and 0 (measured against the CPU twin,
// `score.rs::score_f32`). Every intermediate is therefore `keep`-ed.
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

// The high part of `v` for Dekker's exact product (Veltkamp's split).
fn high(v: f32) -> f32 {
    let c = keep(4097.0 * v);
    return keep(c - keep(c - v));
}

// x / y, CORRECTLY ROUNDED. This device's division is not: its quotients sat a
// last place from the host's (measured — 1 ulp in a, in an MSE), and a scale
// that is a last place off moves every residual. So the device's quotient q is
// only a first guess: the residual x - q*y is formed EXACTLY (Dekker's product,
// every step `keep`-ed) and q corrected by residual / y, where a last place of
// that small quotient no longer reaches the result. The CPU twin runs the same
// steps.
fn div(x: f32, y: f32) -> f32 {
    let q = keep(x / y);
    let p = keep(q * y);
    let qh = high(q);
    let ql = keep(q - qh);
    let yh = high(y);
    let yl = keep(y - yh);
    let e = keep(keep(ql * yl) - keep(keep(keep(p - keep(qh * yh)) - keep(ql * yh)) - keep(qh * yl)));
    let r = keep(keep(x - p) - e);
    if (!finite(r) || !finite(q)) {
        return q;
    }
    return keep(q + keep(r / y));
}

// The total of a compensated sum.
fn total(s: vec2<f32>) -> f32 {
    return keep(s.x + s.y);
}

@compute @workgroup_size(64)
fn score_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= sp.n_chromosomes * N_LINKERS) {
        return;
    }
    let c = t / N_LINKERS;
    let linker = t % N_LINKERS;
    let out = t * N_WRAPPERS * WIDTH;
    let nan = bitcast<f32>(NAN_BITS);
    for (var i = 0u; i < N_WRAPPERS * WIDTH; i = i + 1u) {
        scores[out + i] = nan;
    }
    for (var g = 0u; g < sp.genes_per; g = g + 1u) {
        if (gene_ok[chromosomes[c * sp.genes_per + g]] == 0u) {
            return;
        }
    }
    let nt = sp.n_train;
    let v1 = nt + sp.n_val;

    // Pass 1, train rows: the mean of each wrapper's values — summed as offsets
    // from the first row's value, so a nearly constant column loses nothing —
    // and their range, which decides "constant" exactly, with no sum involved.
    var sum: array<vec2<f32>, 3>;
    var ok: array<bool, 3>;
    var x0: array<f32, 3>;
    var lo: array<f32, 3>;
    var hi: array<f32, 3>;
    let v0 = linked(c, linker, 0u);
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        sum[w] = vec2<f32>(0.0, 0.0);
        ok[w] = true;
        x0[w] = wrapped(v0, w);
        lo[w] = x0[w];
        hi[w] = x0[w];
    }
    // A non-finite value ANYWHERE rejects the candidate, as chrom_score does.
    for (var row = 0u; row < sp.n_rows; row = row + 1u) {
        let v = linked(c, linker, row);
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            let x = wrapped(v, w);
            if (!finite(v) || !finite(x)) {
                ok[w] = false;
            } else if (row < nt) {
                sum[w] = add(sum[w], keep(x - x0[w]));
                lo[w] = min(lo[w], x);
                hi[w] = max(hi[w], x);
            }
        }
    }
    var mx: array<f32, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        mx[w] = keep(x0[w] + div(total(sum[w]), f32(nt)));
    }

    // Pass 2, train rows: centred sums for the least-squares scale.
    var sxx: array<vec2<f32>, 3>;
    var sxy: array<vec2<f32>, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        sxx[w] = vec2<f32>(0.0, 0.0);
        sxy[w] = vec2<f32>(0.0, 0.0);
    }
    for (var row = 0u; row < nt; row = row + 1u) {
        let v = linked(c, linker, row);
        let dy = keep(y[row] - sp.y_mean_train);
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            if (ok[w]) {
                // A product that feeds a sum is `keep`-ed first: fused into a
                // multiply-add it is rounded once, and the CPU twin rounds twice.
                let dx = keep(wrapped(v, w) - mx[w]);
                sxx[w] = add(sxx[w], keep(dx * dx));
                sxy[w] = add(sxy[w], keep(dx * dy));
            }
        }
    }
    var a: array<f32, 3>;
    var b: array<f32, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        let xx = total(sxx[w]);
        // Constant to the resolution the predictions have (chrom_score's rule).
        // every value within tol of the mean  <=>  the range within 2 tol.
        if (!ok[w] || (hi[w] - lo[w]) <= 2.0 * (1e-8 + CONSTANT_REL_TOL * abs(mx[w])) || xx <= 0.0) {
            ok[w] = false;
            continue;
        }
        a[w] = div(total(sxy[w]), xx);
        b[w] = keep(sp.y_mean_train - keep(a[w] * mx[w]));
        if (!finite(a[w]) || !finite(b[w])) {
            ok[w] = false;
        }
    }

    // Pass 3, every row: squared and absolute residuals per split.
    var sq: array<vec2<f32>, 9>;   // [wrapper * 3 + split]
    var ab: array<vec2<f32>, 9>;
    var worst: array<f32, 3>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        sq[i] = vec2<f32>(0.0, 0.0);
        ab[i] = vec2<f32>(0.0, 0.0);
    }
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        worst[w] = 0.0;
    }
    for (var row = 0u; row < sp.n_rows; row = row + 1u) {
        let v = linked(c, linker, row);
        var split = 2u;
        if (row < nt) {
            split = 0u;
        } else if (row < v1) {
            split = 1u;
        }
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            if (ok[w]) {
                let d = keep(y[row] - keep(keep(a[w] * wrapped(v, w)) + b[w]));
                sq[w * 3u + split] = add(sq[w * 3u + split], keep(d * d));
                ab[w * 3u + split] = add(ab[w * 3u + split], abs(d));
                if (split == 1u) {
                    worst[w] = max(worst[w], abs(d));
                }
            }
        }
    }
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        if (!ok[w]) {
            continue;
        }
        let o = out + w * WIDTH;
        let n = array<f32, 3>(f32(nt), f32(sp.n_val), f32(max(sp.n_extrap, 1u)));
        scores[o] = a[w];
        scores[o + 1u] = b[w];
        scores[o + 2u] = div(total(sq[w * 3u]), n[0]);
        scores[o + 3u] = div(total(sq[w * 3u + 1u]), n[1]);
        scores[o + 4u] = worst[w];
        scores[o + 5u] = div(total(sq[w * 3u + 2u]), n[2]);
        scores[o + 6u] = div(total(ab[w * 3u]), n[0]);
        scores[o + 7u] = div(total(ab[w * 3u + 1u]), n[1]);
        scores[o + 8u] = div(total(ab[w * 3u + 2u]), n[2]);
        scores[o + 9u] = 0.0;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            if (!finite(scores[o + i])) {
                for (var k = 0u; k < WIDTH; k = k + 1u) {
                    scores[o + k] = nan;
                }
                ok[w] = false;
                break;
            }
        }
    }

    // LEAVE ONE GENE OUT. Switch each VARYING gene off (hold it at its mean),
    // refit the scale, and see how much of the fit goes: for a law every gene
    // that does anything is load-bearing; refined noise carries genes whose
    // removal costs nothing. redundancy = 1 - median over the varying genes of
    // (1 - R2_without / R2_full), in [0, 1]; 0 = every varying gene matters.
    // With an intercept R2 = sxy^2 / (sxx * syy), so no residual pass is needed.
    // Reads every LOO_STRIDE-th train row.
    var syy = vec2<f32>(0.0, 0.0);
    for (var row = 0u; row < nt; row = row + LOO_STRIDE) {
        let dy = keep(y[row] - sp.y_mean_train);
        syy = add(syy, keep(dy * dy));
    }
    let yy = total(syy);
    var loss: array<f32, 24>;       // [wrapper * 8 + varying gene]
    var n_loss: array<u32, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        n_loss[w] = 0u;
    }
    for (var g = 0u; g < min(sp.genes_per, 8u); g = g + 1u) {
        let gene = chromosomes[c * sp.genes_per + g];
        let g0 = preds[gene * sp.n_rows];
        var gsum = vec2<f32>(0.0, 0.0);
        var gmin = g0;
        var ghi = g0;
        var n_read = 0.0;
        for (var row = 0u; row < nt; row = row + LOO_STRIDE) {
            let v = preds[gene * sp.n_rows + row];
            gsum = add(gsum, keep(v - g0));
            gmin = min(gmin, v);
            ghi = max(ghi, v);
            n_read = n_read + 1.0;
        }
        let gmean = keep(g0 + div(total(gsum), n_read));
        if ((ghi - gmin) <= 2.0 * (1e-8 + CONSTANT_REL_TOL * abs(gmean))) {
            continue;               // a constant gene: nothing to switch off
        }
        var m1: array<vec2<f32>, 3>;
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            m1[w] = vec2<f32>(0.0, 0.0);
        }
        for (var row = 0u; row < nt; row = row + LOO_STRIDE) {
            let v = linked_without(c, linker, row, g, gmean);
            for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
                m1[w] = add(m1[w], wrapped(v, w));
            }
        }
        var xx: array<vec2<f32>, 3>;
        var xy: array<vec2<f32>, 3>;
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            xx[w] = vec2<f32>(0.0, 0.0);
            xy[w] = vec2<f32>(0.0, 0.0);
        }
        for (var row = 0u; row < nt; row = row + LOO_STRIDE) {
            let v = linked_without(c, linker, row, g, gmean);
            let dy = keep(y[row] - sp.y_mean_train);
            for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
                let dx = keep(wrapped(v, w) - div(total(m1[w]), n_read));
                xx[w] = add(xx[w], keep(dx * dx));
                xy[w] = add(xy[w], keep(dx * dy));
            }
        }
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            if (!ok[w]) {
                continue;
            }
            let o = out + w * WIDTH;
            let full = keep(1.0 - div(keep(scores[o + 2u] * f32(nt)), max(keep(yy * f32(LOO_STRIDE)), 1e-30)));
            let sxx_w = total(xx[w]);
            let sxy_w = total(xy[w]);
            var without = 0.0;
            if (sxx_w > 0.0 && yy > 0.0 && finite(sxx_w) && finite(sxy_w)) {
                without = clamp(div(keep(sxy_w * sxy_w), keep(sxx_w * yy)), 0.0, 1.0);
            }
            var l = 1.0;
            if (full > 1e-6) {
                l = clamp(keep(1.0 - div(without, full)), 0.0, 1.0);
            }
            loss[w * 8u + n_loss[w]] = l;
            n_loss[w] = n_loss[w] + 1u;
        }
    }
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        if (!ok[w] || n_loss[w] == 0u) {
            continue;
        }
        // median of at most 8 values: insertion sort, then the middle (mean of
        // the two middles for an even count)
        let n = n_loss[w];
        for (var i = 1u; i < n; i = i + 1u) {
            let v = loss[w * 8u + i];
            var j = i;
            loop {
                if (j == 0u || loss[w * 8u + j - 1u] <= v) {
                    break;
                }
                loss[w * 8u + j] = loss[w * 8u + j - 1u];
                j = j - 1u;
            }
            loss[w * 8u + j] = v;
        }
        let median = keep(0.5 * keep(loss[w * 8u + (n - 1u) / 2u] + loss[w * 8u + n / 2u]));
        scores[out + w * WIDTH + 9u] = keep(1.0 - median);
    }
}
