// Snap, stage 1: literal -> lattice entry id (or NONE) and a sign bit.
//
// One thread per literal. The table is the constant lattice's |value|s, sorted
// strictly ascending in f32. The CPU twin is `snap_table.rs::SnapTable::nearest`
// in `LitMode::F32`: the same integer search and the same multiplies in the
// same order, so the two agree bit for bit. Nothing here is a final decision:
// the host re-derives a hit in f64 (`confirm_f64`) before it is reported or
// written back.

const NONE: u32 = 0xffffffffu;
const INFO_STRIDE: u32 = 4u;
const FLAG_NEGATED: u32 = 1u;
const MANTISSA: u32 = 0x007fffffu;
// The exponent the literal is moved to before any arithmetic: [4, 8).
const FRAME_EXPONENT: u32 = 129u;

struct Cfg {
    n_lit: u32,
    n_table: u32,
    // Threads per dispatch row: a batch past 65535 groups wraps into y.
    stride: u32,
    // Binary-search steps: fixed, so every thread runs the same loop.
    iters: u32,
    rel_tol: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Literals AND the table arrive as f32 BITS. Non-finite and subnormal literals
// are refused on the bit pattern; the search compares integers (the bit
// patterns of non-negative floats order as the floats do).
@group(0) @binding(0) var<storage, read>       lits:   array<u32>;
@group(0) @binding(1) var<storage, read>       table:  array<u32>;
// Per entry: (template offset in nodes, template node count, flags, 0).
@group(0) @binding(2) var<storage, read>       info:   array<u32>;
// Per literal: (entry id or NONE, sign bit).
@group(0) @binding(3) var<storage, read_write> hits:   array<u32>;
@group(0) @binding(4) var<uniform>             cfg:    Cfg;

// One candidate as (distance, scale, 1.0 if it may qualify): its relative
// error is distance / scale, and nobody divides — `abs(x - t) / t` matched
// f32::MAX and f32::MIN_POSITIVE to entries the host refused. Both numbers are
// first moved, by one exact power of two, to where the literal lies in [4, 8):
// there no difference, product or tolerance is subnormal or overflows.
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

    // The entry below, then the entry at or above: strict `<`, so an exact
    // tie in relative error keeps the lower entry.
    var best: u32 = NONE;
    var best_d: f32 = 0.0;
    var best_scale: f32 = 1.0;
    if (lo > 0u) {
        let c = frame(x, table[lo - 1u]);
        if (c.z != 0.0 && c.x <= cfg.rel_tol * c.y) { best = lo - 1u; best_d = c.x; best_scale = c.y; }
    }
    if (lo < cfg.n_table) {
        let c = frame(x, table[lo]);
        // d / scale < best_d / best_scale, cross-multiplied.
        if (c.z != 0.0 && c.x <= cfg.rel_tol * c.y && (best == NONE || c.x * best_scale < best_d * c.y)) {
            best = lo;
        }
    }
    if (best == NONE) { return; }

    hits[i * 2u] = best;
    if (table[best] != 0u) {
        hits[i * 2u + 1u] = (bits >> 31u) ^ (info[best * INFO_STRIDE + 2u] & FLAG_NEGATED);
    }
}
