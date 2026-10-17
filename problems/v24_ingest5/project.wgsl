// THE v24 KEY PROJECTION ON THE DEVICE: harness machinery, hand-written, not
// evolved (fuller problems/v24_ingest5/problem.toml, [[machine]] project).
// One workgroup a record: its sparse unit vector (CSR) projected through one
// seed's pinned Rademacher formula to a unit-norm 2,048-D key, written at
// row wid.x of `keys`. The recipe is qdrant's, verbatim:
//   lib/rp-formula/src/lib.rs (RademacherFormula: new :76, entry :101,
//     accumulate_sparse :118, scalar body :141), RADEMACHER_DOMAIN_V1 :63;
//   ultradimdb/src/formula.rs project_row_one_key :176 (accumulate, then
//     l2_normalise_in_place :1221).
//   a = mix64(DOMAIN ^ seed); b[p] = mix64(a ^ p); h = mix64(b[p] ^ d);
//   entry(p, d) = (bit 63 of h ? -1 : +1) / sqrt(2048) in f32;
//   acc[p] += value * entry(p, d), indices ascending (the canonical order);
//   key = acc / |acc|.
// WGSL has no 64-bit integer: a u64 is vec2<u32>(lo, hi), the 64-bit
// multiply built from 16-bit halves (exact). Each product value * (1/sqrt
// 2048) is formed once by one thread into workgroup memory and loaded before
// it is added (the sign applied as add or subtract: IEEE x - y == x + -y), so
// no multiply can fuse with the add: acc is qdrant's f32 bit for bit. qdrant
// takes the norm in f64; WGSL has none, so the norm here is an f32 sum of
// squares (a tree over the workgroup) and the key may differ from qdrant's in
// the last bits (the GPU test, harness v24_ingest5_projects_its_keys_on_the_device,
// measures it). A record of more than 64 entries (the family's
// max_nnz_per_row is 14) writes non-finite keys.

struct PCfg {
    seed_lo: u32,
    seed_hi: u32,
    row_begin: u32,
    row_count: u32,
};

@group(0) @binding(0) var<storage, read> rec_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> rec_indices: array<u32>;
@group(0) @binding(2) var<storage, read> rec_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<uniform> pcfg: PCfg;

const D_PROJ: u32 = 2048u;
const THREADS: u32 = 256u;
const MAX_NNZ: u32 = 64u;
// RADEMACHER_DOMAIN_V1 = 0x6C1C8DC3E04C9F0A.
const DOMAIN_LO: u32 = 0xE04C9F0Au;
const DOMAIN_HI: u32 = 0x6C1C8DC3u;
// SplitMix64's multipliers 0xBF58476D1CE4E5B9 and 0x94D049BB133111EB.
const M1_LO: u32 = 0x1CE4E5B9u;
const M1_HI: u32 = 0xBF58476Du;
const M2_LO: u32 = 0x133111EBu;
const M2_HI: u32 = 0x94D049BBu;
// 1.0f32 / 2048f32.sqrt(), the f32 qdrant computes (bits 0x3cb504f3).
const INV_BITS: u32 = 0x3cb504f3u;

var<workgroup> prod: array<f32, 64>;
var<workgroup> sums: array<f32, 256>;

// The full 64-bit product of two u32 (16-bit halves, exact).
fn mul_wide(x: u32, y: u32) -> vec2<u32> {
    let x0 = x & 0xFFFFu;
    let x1 = x >> 16u;
    let y0 = y & 0xFFFFu;
    let y1 = y >> 16u;
    let p00 = x0 * y0;
    let p01 = x0 * y1;
    let p10 = x1 * y0;
    let p11 = x1 * y1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// The low 64 bits of a * (m_hi, m_lo).
fn mul64(a: vec2<u32>, m_lo: u32, m_hi: u32) -> vec2<u32> {
    let w = mul_wide(a.x, m_lo);
    return vec2<u32>(w.x, w.y + a.x * m_hi + a.y * m_lo);
}

// x >> n for 0 < n < 32.
fn shr64(x: vec2<u32>, n: u32) -> vec2<u32> {
    return vec2<u32>((x.x >> n) | (x.y << (32u - n)), x.y >> n);
}

// SplitMix64's finaliser (Stafford mix13), qdrant's mix64.
fn mix64(v: vec2<u32>) -> vec2<u32> {
    var x = v ^ shr64(v, 30u);
    x = mul64(x, M1_LO, M1_HI);
    x = x ^ shr64(x, 27u);
    x = mul64(x, M2_LO, M2_HI);
    return x ^ shr64(x, 31u);
}

@compute @workgroup_size(256)
fn project(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let row = pcfg.row_begin + wid.x;
    let begin = rec_offsets[row];
    let nnz = rec_offsets[row + 1u] - begin;
    let inv = bitcast<f32>(INV_BITS);
    // This thread's rows of the projection: p = t + 256 j.
    let a = mix64(vec2<u32>(pcfg.seed_lo ^ DOMAIN_LO, pcfg.seed_hi ^ DOMAIN_HI));
    var b: array<vec2<u32>, 8>;
    var acc: array<f32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        b[j] = mix64(a ^ vec2<u32>(t + THREADS * j, 0u));
        acc[j] = 0.0;
    }
    if (t < MAX_NNZ) {
        var v = 0.0;
        if (t < nnz) {
            v = rec_values[begin + t] * inv;
        }
        prod[t] = v;
    }
    workgroupBarrier();
    let n = min(nnz, MAX_NNZ);
    for (var i = 0u; i < n; i = i + 1u) {
        let d = rec_indices[begin + i];
        let pv = prod[i];
        for (var j = 0u; j < 8u; j = j + 1u) {
            let h = mix64(b[j] ^ vec2<u32>(d, 0u));
            if ((h.y >> 31u) == 0u) {
                acc[j] = acc[j] + pv;
            } else {
                acc[j] = acc[j] - pv;
            }
        }
    }
    var ss = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        ss = ss + acc[j] * acc[j];
    }
    sums[t] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            sums[t] = sums[t] + sums[t + s];
        }
        workgroupBarrier();
    }
    var norm = sqrt(sums[0]);
    if (nnz > MAX_NNZ) {
        // Not a key: a zero norm makes every component non-finite.
        norm = 0.0;
    }
    for (var j = 0u; j < 8u; j = j + 1u) {
        keys[wid.x * D_PROJ + t + THREADS * j] = acc[j] / norm;
    }
}
