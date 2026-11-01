
// Virtual 4-bit weight blocks, generated in nibble space (levels 0..15). VERSION 2: a direct-indexed CDF table
// (tables_fast.f32), one (sum, sum of squares) subgroup reduction a pass, 2 correction passes. Needs the SUBGROUP feature.
//
// One workgroup = one block of 64 cells, one invocation = one cell. Per block the kernel reads only
// (m_mean, m_std, scale, bias) and derives everything else from (seed, block offset, lane) through a
// counter-based 32-bit hash. Shared, read-only tables: P15, the NB+1 span-bin edges, and NB quantile
// tables of the standardised cell value (513 points each), fitted on real 4-bit blocks.
//
//  1. span = 15*|scale|/m_std (the grid's width in standard deviations, known from the inputs) picks
//     the bin, i.e. the shape table: a narrow grid means heavy tails.
//  2. one random lane is pinned to level 0 (every real block has its extreme on the bias rail), one
//     other lane to the top level (15 with probability P15, else 14).
//  3. every other lane draws z by inverse CDF from its bin's table, TRUNCATED to the z range that
//     lands on levels 0..top, so nothing is clamped onto a rail that the data would not put there.
//  4. FIX passes nudge z so the quantised block's mean and std move to the stored m_mean, m_std
//     (fixed-order tree reductions: deterministic).
//  5. out = q*scale + bias, q in 0..top.

const NB: u32 = 16u;
const TP: u32 = 513u;
override FIX: u32 = 2u;
const NC: u32 = 1024u;   // direct CDF table: per bin (zmin, zmax, NC values of F on a uniform z grid)

struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32 }

@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;   // m_mean, m_std, scale, bias
@group(0) @binding(1) var<storage, read> tabs: array<f32>;           // P15, edges[NB+1], T[NB][TP]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;      // n_blocks * 64
@group(0) @binding(3) var<uniform> cfg: Meta;

var<workgroup> red: array<vec2<f32>, 64>;
var<private> lane_of_subgroup: u32;

fn fmix32(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u; x *= 0x85ebca6bu;
    x ^= x >> 13u; x *= 0xc2b2ae35u;
    x ^= x >> 16u;
    return x;
}
fn hash(off: u32, lane: u32, stream: u32) -> u32 {
    return fmix32(fmix32(cfg.seed * 0x9e3779b9u ^ off) ^ (lane | (stream << 8u)));
}
fn u01(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }

fn tab_at(k: u32, i: u32) -> f32 { return tabs[1u + NB + 1u + k * TP + i]; }
// inverse CDF: u in [0,1] -> z
fn inv_cdf(k: u32, u: f32) -> f32 {
    let x = clamp(u, 0.0, 1.0) * f32(TP - 1u);
    let i = min(TP - 2u, u32(x));
    let t = x - f32(i);
    return tab_at(k, i) * (1.0 - t) + tab_at(k, i + 1u) * t;
}
// CDF: z -> u in [0,1], direct-indexed (no search): the table holds F on a uniform z grid per bin
fn cdf(k: u32, z: f32) -> f32 {
    let base = 1u + NB + 1u + NB * TP + k * (2u + NC);
    let z0 = tabs[base]; let z1 = tabs[base + 1u];
    let x = clamp((z - z0) / (z1 - z0), 0.0, 1.0) * f32(NC - 1u);
    let i = min(NC - 2u, u32(x));
    let t = x - f32(i);
    return tabs[base + 2u + i] * (1.0 - t) + tabs[base + 3u + i] * t;
}
// sum over the workgroup of (v, v*v): a subgroup sum, then the subgroups' partial sums added in subgroup order
// (deterministic on a given device; one barrier pair a pass instead of seven)
fn wg_sum2(v: vec2<f32>, sid: u32, ssize: u32) -> vec2<f32> {
    let part = subgroupAdd(v);
    if (sid == 0u) { red[lane_of_subgroup] = part; }
    workgroupBarrier();
    var r = vec2<f32>(0.0);
    for (var g = 0u; g < 64u / ssize; g++) { r += red[g]; }
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(64)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) ssize: u32) {
    lane_of_subgroup = lane / ssize;
    let blk = wid.y * cfg.groups_x + wid.x;
    let valid = blk < cfg.n_blocks;
    let p = params[min(blk, cfg.n_blocks - 1u)];
    let m = p.x; let s = max(p.y, 1e-12); let sc = p.z; let b = p.w;
    let off = cfg.first_offset + blk;

    // 1. the shape bin from the grid's width in standard deviations
    let span = 15.0 * abs(sc) / s;
    var k = 0u;
    for (var j = 1u; j < NB; j++) { if (span >= tabs[1u + j]) { k = j; } }
    // 2. the pinned lanes and the top level (block-level draws: the same in every lane)
    let top = select(14.0, 15.0, u01(hash(off, 0u, 1u)) < tabs[0]);
    let la = hash(off, 0u, 2u) % 64u;
    let lb = (la + 1u + hash(off, 0u, 3u) % 63u) % 64u;
    // 3. truncated inverse-CDF draw
    let za = (b - 0.5 * sc - m) / s;
    let zb = (b + (top + 0.5) * sc - m) / s;
    let flo = cdf(k, min(za, zb));
    let fhi = cdf(k, max(za, zb));
    var z = inv_cdf(k, flo + u01(hash(off, lane, 0u)) * (fhi - flo));
    // 4./5. quantise, pin, and move the block's mean and std to the stored ones
    var y = 0.0;
    for (var it = 0u; it <= FIX; it++) {
        var q = clamp(round((m + s * z - b) / sc), 0.0, top);
        if (lane == la) { q = 0.0; }
        if (lane == lb) { q = top; }
        y = q * sc + b;
        if (it < FIX) {
            let r = wg_sum2(vec2<f32>(y, y * y), sid, ssize) / 64.0;
            let ym = r.x;
            let ys = sqrt(max(r.y - ym * ym, 0.0));
            z = (z - (ym - m) / s) * select(1.0, s / ys, ys > 0.0);
        }
    }
    if (valid) { out[blk * 64u + lane] = y; }
}
