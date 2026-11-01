
// Virtual 4-bit weight blocks, generated in nibble space (levels 0..15). VERSION 4: a block is ONE SUBGROUP of 32
// invocations, each holding 2 cells (c and c + 32), so every block reduction is a subgroupAdd with no workgroup
// barrier. Requires a subgroup size of exactly 32 (Apple GPUs); any other size writes NaN, loudly. Values are the
// same draws as version 3 (each cell hashes its own index). VERSION 3: the span bin is a direct
// lookup (tables_v3.f32 appends smin, smax and 4096 bins) instead of a 15-step loop over the edges. VERSION 2: a direct-indexed CDF table
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
const NC: u32 = 1024u;
const NS: u32 = 4096u;   // span -> bin lookup cells   // direct CDF table: per bin (zmin, zmax, NC values of F on a uniform z grid)

struct Meta { n_blocks: u32, seed: u32, groups_x: u32, first_offset: u32 }

@group(0) @binding(0) var<storage, read> params: array<vec4<f32>>;   // m_mean, m_std, scale, bias
@group(0) @binding(1) var<storage, read> tabs: array<f32>;           // P15, edges[NB+1], T[NB][TP]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;      // n_blocks * 64
@group(0) @binding(3) var<uniform> cfg: Meta;

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

// one block: the two cells (lane, lane + 32) of this invocation; all block-level sums are subgroup sums
fn gen_pair(blk: u32, lane: u32) -> vec2<f32> {
    let p = params[min(blk, cfg.n_blocks - 1u)];
    let m = p.x; let s = max(p.y, 1e-12); let sc = p.z; let b = p.w;
    let off = cfg.first_offset + blk;
    let span = 15.0 * abs(sc) / s;
    let sbase = 1u + NB + 1u + NB * TP + NB * (2u + NC);
    let smin = tabs[sbase]; let smax = tabs[sbase + 1u];
    let k = u32(tabs[sbase + 2u + min(NS - 1u, u32(clamp((span - smin) / (smax - smin), 0.0, 1.0) * f32(NS)))]);
    let top = select(14.0, 15.0, u01(hash(off, 0u, 1u)) < tabs[0]);
    let la = hash(off, 0u, 2u) % 64u;
    let lb = (la + 1u + hash(off, 0u, 3u) % 63u) % 64u;
    let za = (b - 0.5 * sc - m) / s;
    let zb = (b + (top + 0.5) * sc - m) / s;
    let flo = cdf(k, min(za, zb));
    let fhi = cdf(k, max(za, zb));
    let c = vec2<u32>(lane, lane + 32u);
    var z = vec2<f32>(inv_cdf(k, flo + u01(hash(off, c.x, 0u)) * (fhi - flo)),
                      inv_cdf(k, flo + u01(hash(off, c.y, 0u)) * (fhi - flo)));
    var y = vec2<f32>(0.0);
    for (var it = 0u; it <= FIX; it++) {
        var q = clamp(round((m + s * z - b) / sc), vec2<f32>(0.0), vec2<f32>(top));
        q = select(q, vec2<f32>(0.0), c == vec2<u32>(la));
        q = select(q, vec2<f32>(top), c == vec2<u32>(lb));
        y = q * sc + b;
        if (it < FIX) {
            let r = subgroupAdd(vec2<f32>(y.x + y.y, y.x * y.x + y.y * y.y)) / 64.0;
            let ym = r.x;
            let ys = sqrt(max(r.y - ym * ym, 0.0));
            z = (z - (ym - m) / s) * select(1.0, s / ys, ys > 0.0);
        }
    }
    return y;
}

@compute @workgroup_size(32)
fn gen_block(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
             @builtin(subgroup_size) ssize: u32) {
    let blk = wid.y * cfg.groups_x + wid.x;
    var y = gen_pair(blk, lane);
    if (ssize != 32u) { y = vec2<f32>(bitcast<f32>(0x7fc00000u)); }
    if (blk < cfg.n_blocks) { out[blk * 64u + lane] = y.x; out[blk * 64u + lane + 32u] = y.y; }
}
