// THE POOLED JUDGE (pooled_recall, fuller harness `DeviceExecutor::judge_pooled`):
// the exact re-rank on the device. One invocation per (stage, query): the seeds'
// candidate lists pooled (pads dropped, deduplicated), entries at or past the
// stage's arrived rows counted broken and not ranked, the rest scored by the
// exact sparse dot of their CSR record with the query's record and ranked by
// score then row id; found = the top `top`'s rows in the query's accept set.
// Writes (found, pooled, broken) per (stage, query); the host sums them into
// windows.
//
// THE DOT IS EXACT: each product of two f32 values is an integer mantissa
// product (48 bits) times a power of two, added into a 128-bit fixed-point
// accumulator (least bit 2^-124) with integer arithmetic only, positive and
// negative parts apart; so the ranking is the exact one (the host's f64 sum
// differs from it only within its own rounding, ~1e-15), and no floating
// point operation the device's fast math could reorder is used.

struct Params {
    stages: u32,
    seeds: u32,
    queries: u32,
    c_max: u32,
    top: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cands: array<u32>;
@group(0) @binding(2) var<storage, read> rec_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> rec_indices: array<u32>;
// The records' f32 values, as their bits.
@group(0) @binding(4) var<storage, read> rec_values: array<u32>;
// Per query, its record's row.
@group(0) @binding(5) var<storage, read> qrow: array<u32>;
// Per stage, its arrived rows.
@group(0) @binding(6) var<storage, read> arrived: array<u32>;
// Per stage and query, the accept set's range in accept_rows: stage s, query q
// at accept_offsets[s * (queries + 1) + q .. + 1], absolute into accept_rows.
@group(0) @binding(7) var<storage, read> accept_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> accept_rows: array<u32>;
@group(0) @binding(9) var<storage, read_write> judged: array<u32>;

const PAD: u32 = 0xFFFFFFFFu;
const MAX_POOL: u32 = 256u;
const MAX_TOP: u32 = 16u;

var<private> pool: array<u32, 256>;
var<private> top_id: array<u32, 16>;
var<private> top_dot: array<vec4<u32>, 16>;

// 128-bit unsigned arithmetic, four u32 limbs, limb x the lowest.
fn add128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r: vec4<u32>;
    var carry = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let s = a[i] + b[i];
        let c1 = select(0u, 1u, s < a[i]);
        let t = s + carry;
        let c2 = select(0u, 1u, t < s);
        r[i] = t;
        carry = c1 + c2;
    }
    return r;
}

fn sub128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var r: vec4<u32>;
    var borrow = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let d = a[i] - b[i];
        let b1 = select(0u, 1u, a[i] < b[i]);
        let t = d - borrow;
        let b2 = select(0u, 1u, d < borrow);
        r[i] = t;
        borrow = b1 + b2;
    }
    return r;
}

// v shifted left by s bits (s < 128), the bits past the top dropped.
fn shl128(v: vec4<u32>, s: u32) -> vec4<u32> {
    let words = s / 32u;
    let bits = s % 32u;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = 3; i >= 0; i = i - 1) {
        let src = i - i32(words);
        if (src < 0) {
            continue;
        }
        var w = v[src] << bits;
        if (bits != 0u && src >= 1) {
            w = w | (v[src - 1] >> (32u - bits));
        }
        r[i] = w;
    }
    return r;
}

// v shifted right by s bits (s < 128), truncating.
fn shr128(v: vec4<u32>, s: u32) -> vec4<u32> {
    let words = s / 32u;
    let bits = s % 32u;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = 0; i < 4; i = i + 1) {
        let src = i + i32(words);
        if (src > 3) {
            continue;
        }
        var w = v[src] >> bits;
        if (bits != 0u && src <= 2) {
            w = w | (v[src + 1] << (32u - bits));
        }
        r[i] = w;
    }
    return r;
}

// a > b as signed 128-bit (two's complement) numbers.
fn greater128(a: vec4<u32>, b: vec4<u32>) -> bool {
    let sa = a.w >> 31u;
    let sb = b.w >> 31u;
    if (sa != sb) {
        return sa == 0u;
    }
    for (var i = 3; i >= 0; i = i - 1) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return false;
}

// One f32's (sign, 24-bit mantissa, exponent): the value is mant * 2^exp.
// A value that is not finite counts as zero.
fn split_f32(bits: u32) -> vec3<u32> {
    let e = (bits >> 23u) & 0xFFu;
    let f = bits & 0x7FFFFFu;
    if (e == 0xFFu) {
        return vec3<u32>(0u, 0u, 0u);
    }
    if (e == 0u) {
        // exponent -149, offset by 300 to stay unsigned
        return vec3<u32>(bits >> 31u, f, 151u);
    }
    return vec3<u32>(bits >> 31u, f | 0x800000u, e + 150u);
}

// The exact sparse dot of records a and b, 128-bit fixed point, least bit 2^-124.
fn exact_dot(a: u32, b: u32) -> vec4<u32> {
    var pos = vec4<u32>(0u, 0u, 0u, 0u);
    var neg = vec4<u32>(0u, 0u, 0u, 0u);
    for (var i = rec_offsets[a]; i < rec_offsets[a + 1u]; i = i + 1u) {
        for (var j = rec_offsets[b]; j < rec_offsets[b + 1u]; j = j + 1u) {
            if (rec_indices[i] != rec_indices[j]) {
                continue;
            }
            let x = split_f32(rec_values[i]);
            let y = split_f32(rec_values[j]);
            if (x.y == 0u || y.y == 0u) {
                continue;
            }
            // The 48-bit mantissa product, by 16-bit halves.
            let xl = x.y & 0xFFFFu;
            let xh = x.y >> 16u;
            let yl = y.y & 0xFFFFu;
            let yh = y.y >> 16u;
            let ll = xl * yl;
            let mid = xh * yl + xl * yh;
            let hh = xh * yh;
            let lo = ll + (mid << 16u);
            let carry = select(0u, 1u, lo < ll);
            let hi = hh + (mid >> 16u) + carry;
            let p = vec4<u32>(lo, hi, 0u, 0u);
            // value = p * 2^(ex + ey), each exponent offset by 300; the accumulator's least bit is 2^-124.
            let shift = i32(x.z + y.z) - 600 + 124;
            var v = vec4<u32>(0u, 0u, 0u, 0u);
            if (shift >= 0) {
                if (shift < 128) {
                    v = shl128(p, u32(shift));
                }
            } else if (shift > -128) {
                v = shr128(p, u32(-shift));
            }
            if ((x.x ^ y.x) == 0u) {
                pos = add128(pos, v);
            } else {
                neg = add128(neg, v);
            }
        }
    }
    return sub128(pos, neg);
}

// (dot, id) ranks before (dot2, id2): the greater dot, then the smaller id.
fn before(d: vec4<u32>, id: u32, d2: vec4<u32>, id2: u32) -> bool {
    if (greater128(d, d2)) {
        return true;
    }
    if (greater128(d2, d)) {
        return false;
    }
    return id < id2;
}

@compute @workgroup_size(64)
fn judge_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.stages * params.queries) {
        return;
    }
    let stage0 = idx / params.queries;
    let q = idx % params.queries;
    // The pool: every seed's list, pads dropped.
    var n = 0u;
    for (var s = 0u; s < params.seeds; s = s + 1u) {
        let base = ((stage0 * params.seeds + s) * params.queries + q) * params.c_max;
        for (var j = 0u; j < params.c_max; j = j + 1u) {
            let c = cands[base + j];
            if (c != PAD && n < MAX_POOL) {
                pool[n] = c;
                n = n + 1u;
            }
        }
    }
    // Sorted (insertion), then deduplicated.
    for (var i = 1u; i < n; i = i + 1u) {
        let v = pool[i];
        var j = i;
        loop {
            if (j == 0u || pool[j - 1u] <= v) { break; }
            pool[j] = pool[j - 1u];
            j = j - 1u;
        }
        pool[j] = v;
    }
    var m = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        if (m == 0u || pool[i] != pool[m - 1u]) {
            pool[m] = pool[i];
            m = m + 1u;
        }
    }
    let arrived_rows = arrived[stage0];
    let qr = qrow[q];
    let top = min(params.top, MAX_TOP);
    var kept = 0u;
    var broken = 0u;
    for (var i = 0u; i < m; i = i + 1u) {
        let c = pool[i];
        if (c >= arrived_rows) {
            broken = broken + 1u;
            continue;
        }
        let d = exact_dot(qr, c);
        // Insert into the top list, ranked.
        if (kept < top || before(d, c, top_dot[kept - 1u], top_id[kept - 1u])) {
            var at = min(kept, top - 1u);
            if (kept < top) {
                kept = kept + 1u;
            }
            loop {
                if (at == 0u || !before(d, c, top_dot[at - 1u], top_id[at - 1u])) { break; }
                top_dot[at] = top_dot[at - 1u];
                top_id[at] = top_id[at - 1u];
                at = at - 1u;
            }
            top_dot[at] = d;
            top_id[at] = c;
        }
    }
    // Found: the top's rows in the accept set (sorted), by binary search.
    let ab = stage0 * (params.queries + 1u) + q;
    let lo0 = accept_offsets[ab];
    let hi0 = accept_offsets[ab + 1u];
    var found = 0u;
    for (var t = 0u; t < kept; t = t + 1u) {
        var lo = lo0;
        var hi = hi0;
        loop {
            if (lo >= hi) { break; }
            let mid = lo + (hi - lo) / 2u;
            let r = accept_rows[mid];
            if (r == top_id[t]) {
                found = found + 1u;
                break;
            }
            if (r < top_id[t]) { lo = mid + 1u; } else { hi = mid; }
        }
    }
    judged[idx * 3u] = min(found, top);
    judged[idx * 3u + 1u] = m;
    judged[idx * 3u + 2u] = broken;
}
