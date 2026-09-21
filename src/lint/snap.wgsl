// Snap, stage 1: literal -> lattice entry id (or NONE) and a sign bit.
//
// One thread per literal. The table is the constant lattice's |value|s, sorted
// strictly ascending in f32. The CPU twin is `snap_table.rs::SnapTable::search`
// in `LitMode::F32`: the same integer search, the same band in the same order
// and the same f32 operations, so the two agree bit for bit. Nothing here is a
// final decision: the host re-derives a hit in f64 (`confirm_f64_in`) before
// it is reported or written back.
//
// WHICH ENTRY of the band: the smallest HFF TrueNorth angle over (x_near,
// x_size, x_fit). The angle is acos(1 - min(energy / 3, 1)), monotone in the
// energy x_near^2 + x_size^2 + x_fit^2, so this kernel takes the argmin of the
// energy and never calls acos. x_size^2 and x_fit^2 arrive squared in
// `affinity` — data, the host's `SnapAffinity` — and are only ADDED here.

const NONE: u32 = 0xffffffffu;
const INFO_STRIDE: u32 = 4u;
const FLAG_NEGATED: u32 = 1u;
const MANTISSA: u32 = 0x007fffffu;
// The exponent the literal is moved to before any arithmetic: [4, 8).
const FRAME_EXPONENT: u32 = 129u;
// Entries scanned each side of the insertion point.
const BAND: u32 = 16u;
// Second word of a hit whose band reaches past BAND: no entry, and counted.
const BAND_OVERFLOW: u32 = 2u;
const FAMILIES: u32 = 6u;
// Words of x_fit^2 at the head of `affinity`; x_size^2 per node count follows.
const FIT_WORDS: u32 = 18u;

struct Cfg {
    n_lit: u32,
    n_table: u32,
    // Threads per dispatch row: a batch past 65535 groups wraps into y.
    stride: u32,
    // Binary-search steps: fixed, so every thread runs the same loop.
    iters: u32,
    rel_tol: f32,
    // 1 / rel_tol, and 0 when rel_tol is 0.
    inv_tol: f32,
    pad1: u32,
    pad2: u32,
}

// Literals AND the table arrive as f32 BITS. Non-finite and subnormal literals
// are refused on the bit pattern; the search compares integers (the bit
// patterns of non-negative floats order as the floats do).
@group(0) @binding(0) var<storage, read>       lits:     array<u32>;
@group(0) @binding(1) var<storage, read>       table:    array<u32>;
// Per entry: (template offset in nodes, template node count, flags, family).
@group(0) @binding(2) var<storage, read>       info:     array<u32>;
// Per literal: (entry id or NONE, sign bit — or BAND_OVERFLOW beside NONE).
@group(0) @binding(3) var<storage, read_write> hits:     array<u32>;
@group(0) @binding(4) var<uniform>             cfg:      Cfg;
// Per literal: its context code (cyclic, algebraic, exponential).
@group(0) @binding(5) var<storage, read>       contexts: array<u32>;
// x_fit^2 [context * FAMILIES + family], then x_size^2 [node count].
@group(0) @binding(6) var<storage, read>       affinity: array<f32>;

// One candidate as (distance, scale, 1.0 if it may qualify): its relative
// error is distance / scale. Both numbers are first moved, by one exact power
// of two, to where the literal lies in [4, 8): there no difference, product,
// quotient or tolerance is subnormal or overflows — `abs(x - t) / t` on the raw
// values matched f32::MAX and f32::MIN_POSITIVE to entries the host refused.
// `snap_table.rs::frame_f32` is this function.
fn frame(x: u32, t: u32) -> vec3<f32> {
    if (t == 0u) {
        // Against a zero constant the error is the literal's own magnitude.
        return vec3<f32>(bitcast<f32>(x), 1.0, 1.0);
    }
    let x_exp = x >> 23u;
    let t_exp = t >> 23u;
    // More than a binade apart: the relative error is over 1/2.
    if (x_exp == 0u || t_exp + 1u < x_exp || t_exp > x_exp + 1u) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let xf = bitcast<f32>((x & MANTISSA) | (FRAME_EXPONENT << 23u));
    let tf = bitcast<f32>((t & MANTISSA) | ((t_exp + FRAME_EXPONENT - x_exp) << 23u));
    return vec3<f32>(abs(xf - tf), tf, 1.0);
}

fn in_band(c: vec3<f32>) -> bool {
    return c.z != 0.0 && c.x <= cfg.rel_tol * c.y;
}

@compute @workgroup_size(64)
fn snap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * cfg.stride + gid.x;
    if (i >= cfg.n_lit) { return; }
    hits[i * 2u] = NONE;
    hits[i * 2u + 1u] = 0u;

    let bits = lits[i];
    let x = bits & 0x7fffffffu;
    let expo = x >> 23u;
    // inf / NaN, or subnormal (zero itself is a literal like any other).
    if (expo == 255u || (expo == 0u && x != 0u)) { return; }

    // Lower bound: the first entry >= x.
    var lo: u32 = 0u;
    var hi: u32 = cfg.n_table;
    for (var k: u32 = 0u; k < cfg.iters; k = k + 1u) {
        if (lo < hi) {
            let mid = (lo + hi) / 2u;
            if (table[mid] < x) { lo = mid + 1u; } else { hi = mid; }
        }
    }

    // The band: BAND entries each side of the insertion point. An entry within
    // tolerance just past either end means the scan would miss candidates.
    var first: u32 = 0u;
    if (lo > BAND) { first = lo - BAND; }
    let last = min(lo + BAND, cfg.n_table);
    var past = false;
    if (first > 0u) { past = in_band(frame(x, table[first - 1u])); }
    if (last < cfg.n_table) { past = past || in_band(frame(x, table[last])); }
    if (past) {
        hits[i * 2u + 1u] = BAND_OVERFLOW;
        return;
    }

    // Ascending, strict `<`: a tie in energy keeps the lower entry. Energies
    // are compared as (x - bx)(x + bx) < bc - c: no product feeds a sum.
    let fit = contexts[i] * FAMILIES;
    var best: u32 = NONE;
    var best_x: f32 = 0.0;
    var best_c: f32 = 0.0;
    for (var k: u32 = 0u; k < 2u * BAND; k = k + 1u) {
        let j = first + k;
        if (j < last) {
            let c = frame(x, table[j]);
            if (in_band(c)) {
                let near = (c.x / c.y) * cfg.inv_tol;
                let cost = affinity[FIT_WORDS + info[j * INFO_STRIDE + 1u]] + affinity[fit + info[j * INFO_STRIDE + 3u]];
                if (best == NONE || (near - best_x) * (near + best_x) < best_c - cost) {
                    best = j;
                    best_x = near;
                    best_c = cost;
                }
            }
        }
    }
    if (best == NONE) { return; }

    hits[i * 2u] = best;
    if (table[best] != 0u) {
        hits[i * 2u + 1u] = (bits >> 31u) ^ (info[best * INFO_STRIDE + 2u] & FLAG_NEGATED);
    }
}
