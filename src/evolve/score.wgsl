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
    pad0: u32,
}

@group(0) @binding(0) var<uniform> sp: Meta;
@group(0) @binding(1) var<storage, read> preds: array<f32>;       // gene-major, n_rows each
@group(0) @binding(2) var<storage, read> gene_ok: array<u32>;
@group(0) @binding(3) var<storage, read> chromosomes: array<u32>; // genes_per gene indices each
@group(0) @binding(4) var<storage, read> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;

const N_LINKERS: u32 = 3u;   // avg, mul, add — chrom_score's order in the engine
const N_WRAPPERS: u32 = 3u;  // identity, log_abs, sqrt_abs
const WIDTH: u32 = 9u;       // a, b, mse_t, mse_v, max_err_v, mse_e, mae_t, mae_v, mae_e
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
            acc = acc * v;
        } else {
            acc = acc + v;
        }
    }
    if (linker == 0u) {
        acc = acc / f32(sp.genes_per);
    }
    return acc;
}

fn wrapped(v: f32, w: u32) -> f32 {
    if (w == 1u) {
        return log(abs(v) + 1e-12);
    }
    if (w == 2u) {
        return sqrt(abs(v));
    }
    return v;
}

// Neumaier's compensated addition: (sum, compensation) += v.
fn add(s: vec2<f32>, v: f32) -> vec2<f32> {
    let t = s.x + v;
    var c = s.y;
    if (abs(s.x) >= abs(v)) {
        c = c + ((s.x - t) + v);
    } else {
        c = c + ((v - t) + s.x);
    }
    return vec2<f32>(t, c);
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
                sum[w] = add(sum[w], x - x0[w]);
                lo[w] = min(lo[w], x);
                hi[w] = max(hi[w], x);
            }
        }
    }
    var mx: array<f32, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        mx[w] = x0[w] + (sum[w].x + sum[w].y) / f32(nt);
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
        let dy = y[row] - sp.y_mean_train;
        for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
            if (ok[w]) {
                let dx = wrapped(v, w) - mx[w];
                sxx[w] = add(sxx[w], dx * dx);
                sxy[w] = add(sxy[w], dx * dy);
            }
        }
    }
    var a: array<f32, 3>;
    var b: array<f32, 3>;
    for (var w = 0u; w < N_WRAPPERS; w = w + 1u) {
        let xx = sxx[w].x + sxx[w].y;
        // Constant to the resolution the predictions have (chrom_score's rule).
        // every value within tol of the mean  <=>  the range within 2 tol.
        if (!ok[w] || (hi[w] - lo[w]) <= 2.0 * (1e-8 + CONSTANT_REL_TOL * abs(mx[w])) || xx <= 0.0) {
            ok[w] = false;
            continue;
        }
        a[w] = (sxy[w].x + sxy[w].y) / xx;
        b[w] = sp.y_mean_train - a[w] * mx[w];
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
                let d = y[row] - (a[w] * wrapped(v, w) + b[w]);
                sq[w * 3u + split] = add(sq[w * 3u + split], d * d);
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
        scores[o + 2u] = (sq[w * 3u].x + sq[w * 3u].y) / n[0];
        scores[o + 3u] = (sq[w * 3u + 1u].x + sq[w * 3u + 1u].y) / n[1];
        scores[o + 4u] = worst[w];
        scores[o + 5u] = (sq[w * 3u + 2u].x + sq[w * 3u + 2u].y) / n[2];
        scores[o + 6u] = (ab[w * 3u].x + ab[w * 3u].y) / n[0];
        scores[o + 7u] = (ab[w * 3u + 1u].x + ab[w * 3u + 1u].y) / n[1];
        scores[o + 8u] = (ab[w * 3u + 2u].x + ab[w * 3u + 2u].y) / n[2];
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            if (!finite(scores[o + i])) {
                for (var k = 0u; k < WIDTH; k = k + 1u) {
                    scores[o + k] = nan;
                }
                break;
            }
        }
    }
}
