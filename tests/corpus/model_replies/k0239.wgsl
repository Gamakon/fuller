// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var q: u32 = 0u; q < 8u; q = q + 1u) {
        acc[q & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let s = lidx + 128u * k;
            let item = s >> 4u;
            let v = s & 15u;
            let ti = t + item;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (ti < m) {
                let base = ti * d;
                let c = col0 + v * 4u;
                if (c < d) { a.x = phi[base + c]; }
                if (c + 1u < d) { a.y = phi[base + c + 1u]; }
                if (c + 2u < d) { a.z = phi[base + c + 2u]; }
                if (c + 3u < d) { a.w = phi[base + c + 3u]; }
                let lr = row0 + v * 4u;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { b.x = phi[base + gr]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < d) { b.y = phi[base + gr + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < d) { b.z = phi[base + gr + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < d) { b.w = phi[base + gr + 3u]; }
            }
            wg_va[s & 255u] = a;
            wg_vb[s & 255u] = b;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let a = wg_va[(tt * 16u + lid.x) & 255u];
            let b0 = wg_vb[(tt * 16u + lid.y * 2u) & 255u];
            let b1 = wg_vb[(tt * 16u + lid.y * 2u + 1u) & 255u];
            acc[0u] = acc[0u] + vec4<f32>(b0.x) * a;
            acc[1u] = acc[1u] + vec4<f32>(b0.y) * a;
            acc[2u] = acc[2u] + vec4<f32>(b0.z) * a;
            acc[3u] = acc[3u] + vec4<f32>(b0.w) * a;
            acc[4u] = acc[4u] + vec4<f32>(b1.x) * a;
            acc[5u] = acc[5u] + vec4<f32>(b1.y) * a;
            acc[6u] = acc[6u] + vec4<f32>(b1.z) * a;
            acc[7u] = acc[7u] + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }
    let cbase = col0 + lid.x * 4u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let r = acc[i & 7u];
            let o = lr * d + cbase;
            if (cbase < d) { h[o] = h[o] + r.x; }
            if (cbase + 1u < d) { h[o + 1u] = h[o + 1u] + r.y; }
            if (cbase + 2u < d) { h[o + 2u] = h[o + 2u] + r.z; }
            if (cbase + 3u < d) { h[o + 3u] = h[o + 3u] + r.w; }
        }
    }
}
