// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;

fn ld(t: u32, c: u32, ok: bool) -> f32 {
    if (ok) { return phi[t * cfg.d + c]; }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tt = t + (idx >> 4u);
            let cg = (idx & 15u) * 4u;
            let inb = tt < m;
            let c0 = colbase + cg;
            let l0 = rowbase + cg;
            let g0 = cfg.r_off + l0;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            va.x = ld(tt, c0, inb && (c0 < d));
            va.y = ld(tt, c0 + 1u, inb && (c0 + 1u < d));
            va.z = ld(tt, c0 + 2u, inb && (c0 + 2u < d));
            va.w = ld(tt, c0 + 3u, inb && (c0 + 3u < d));
            vb.x = ld(tt, g0, inb && (l0 < cfg.r_rows) && (g0 < d));
            vb.y = ld(tt, g0 + 1u, inb && (l0 + 1u < cfg.r_rows) && (g0 + 1u < d));
            vb.z = ld(tt, g0 + 2u, inb && (l0 + 2u < cfg.r_rows) && (g0 + 2u < d));
            vb.w = ld(tt, g0 + 3u, inb && (l0 + 3u < cfg.r_rows) && (g0 + 3u < d));
            wg_a[idx & 511u] = va;
            wg_b[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a = wg_a[(kk * 16u + lid.x) & 511u];
            let b = wg_b[(kk * 16u + lid.y) & 511u];
            a0 = a0 + vec4<f32>(b.x) * a;
            a1 = a1 + vec4<f32>(b.y) * a;
            a2 = a2 + vec4<f32>(b.z) * a;
            a3 = a3 + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    let col = colbase + lid.x * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowbase + lid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var acc = a0;
            if (j == 1u) { acc = a1; }
            if (j == 2u) { acc = a2; }
            if (j == 3u) { acc = a3; }
            let base = lr * d + col;
            if (col < d) { h[base] = h[base] + acc.x; }
            if (col + 1u < d) { h[base + 1u] = h[base + 1u] + acc.y; }
            if (col + 2u < d) { h[base + 2u] = h[base + 2u] + acc.z; }
            if (col + 3u < d) { h[base + 3u] = h[base + 3u] + acc.w; }
        }
    }
}
