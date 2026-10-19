// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 256>;

fn ldv(item: u32, start: u32, loc: u32, lim: u32) -> vec4<f32> {
    var r = vec4<f32>(0.0);
    if (item < cfg.m) {
        let base = item * cfg.d + start;
        if (loc < lim) { r.x = phi[base]; }
        if (loc + 1u < lim) { r.y = phi[base + 1u]; }
        if (loc + 2u < lim) { r.z = phi[base + 2u]; }
        if (loc + 3u < lim) { r.w = phi[base + 3u]; }
    }
    return r;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let isb = lidx >= 128u;
    let q = lidx & 127u;
    let it = q >> 4u;
    let c4 = q & 15u;
    var lima: u32 = 0u;
    if (cfg.r_off < cfg.d) { lima = min(cfg.r_rows, cfg.d - cfg.r_off); }
    let loc = select(wid.y * 64u, wid.x * 64u, isb) + c4 * 4u;
    let start = select(cfg.r_off + loc, loc, isb);
    let lim = select(lima, cfg.d, isb);

    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);

    let v0 = ldv(it, start, loc, lim);
    if (isb) { wg_b[q & 255u] = v0; } else { wg_a[q & 255u] = v0; }
    workgroupBarrier();

    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        let cur = (t >> 3u) & 1u;
        let nxt = 1u - cur;
        let nv = ldv(t + 8u + it, start, loc, lim);
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let av = wg_a[(cur * 128u + i * 16u + lid.y) & 255u];
            let bv = wg_b[(cur * 128u + i * 16u + lid.x) & 255u];
            c0 = fma(vec4<f32>(av.x), bv, c0);
            c1 = fma(vec4<f32>(av.y), bv, c1);
            c2 = fma(vec4<f32>(av.z), bv, c2);
            c3 = fma(vec4<f32>(av.w), bv, c3);
        }
        if (isb) { wg_b[(nxt * 128u + q) & 255u] = nv; } else { wg_a[(nxt * 128u + q) & 255u] = nv; }
        workgroupBarrier();
    }

    let row0 = wid.y * 64u + lid.y * 4u;
    let col0 = wid.x * 64u + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            var cv = c0;
            if (i == 1u) { cv = c1; }
            if (i == 2u) { cv = c2; }
            if (i == 3u) { cv = c3; }
            let rb = lr * cfg.d;
            if (col0 < cfg.d) { h[rb + col0] = h[rb + col0] + cv.x; }
            if (col0 + 1u < cfg.d) { h[rb + col0 + 1u] = h[rb + col0 + 1u] + cv.y; }
            if (col0 + 2u < cfg.d) { h[rb + col0 + 2u] = h[rb + col0 + 2u] + cv.z; }
            if (col0 + 3u < cfg.d) { h[rb + col0 + 3u] = h[rb + col0 + 3u] + cv.w; }
        }
    }
}
