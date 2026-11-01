// Virtual 4-bit weight blocks, generated in nibble space (levels 0..15).
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
const FIX: u32 = 4u;

struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32 }

@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;   // m_mean, m_std, scale, bias
@group(0) @binding(1) var<storage, read> tabs: array<f32>;           // P15, edges[NB+1], T[NB][TP]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;      // n_blocks * 64
@group(0) @binding(3) var<uniform> cfg: Meta;

var<workgroup> red: array<f32, 64>;

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
// CDF: z -> u in [0,1] (binary search in the monotone table)
fn cdf(k: u32, z: f32) -> f32 {
    if (z <= tab_at(k, 0u)) { return 0.0; }
    if (z >= tab_at(k, TP - 1u)) { return 1.0; }
    var lo = 0u; var hi = TP - 1u;
    loop {
        if (hi - lo <= 1u) { break; }
        let mid = (lo + hi) / 2u;
        if (tab_at(k, mid) <= z) { lo = mid; } else { hi = mid; }
    }
    let a = tab_at(k, lo); let b = tab_at(k, hi);
    let t = select(0.0, (z - a) / (b - a), b > a);
    return (f32(lo) + t) / f32(TP - 1u);
}
// fixed-order sum over the workgroup (deterministic)
fn wg_sum(v: f32, lane: u32) -> f32 {
    red[lane] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lane < s) { red[lane] = red[lane] + red[lane + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(64)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
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
            let ym = wg_sum(y, lane) / 64.0;
            let d = y - ym;
            let ys = sqrt(wg_sum(d * d, lane) / 64.0);
            z = (z - (ym - m) / s) * select(1.0, s / ys, ys > 0.0);
        }
    }
    if (valid) { out[blk * 64u + lane] = y; }
}
