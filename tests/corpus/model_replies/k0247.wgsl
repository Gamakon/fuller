// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    var c4 = vec4<f32>(0.0);
    var c5 = vec4<f32>(0.0);
    var c6 = vec4<f32>(0.0);
    var c7 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let it = t + (e >> 4u);
            let q = (e & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (it < m) {
                let base = it * d;
                let col = colbase + q;
                if (col < d) { a.x = phi[base + col]; }
                if (col + 1u < d) { a.y = phi[base + col + 1u]; }
                if (col + 2u < d) { a.z = phi[base + col + 2u]; }
                if (col + 3u < d) { a.w = phi[base + col + 3u]; }
                let lr = rowbase + q;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { b.x = phi[base + gr]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < d) { b.y = phi[base + gr + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < d) { b.z = phi[base + gr + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < d) { b.w = phi[base + gr + 3u]; }
            }
            wg_va[e & 511u] = a;
            wg_vb[e & 511u] = b;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            let a = wg_va[(s * 16u + lid.x) & 511u];
            let b0 = wg_vb[(s * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(s * 16u + lid.y * 2u + 1u) & 511u];
            c0 = c0 + vec4<f32>(b0.x) * a;
            c1 = c1 + vec4<f32>(b0.y) * a;
            c2 = c2 + vec4<f32>(b0.z) * a;
            c3 = c3 + vec4<f32>(b0.w) * a;
            c4 = c4 + vec4<f32>(b1.x) * a;
            c5 = c5 + vec4<f32>(b1.y) * a;
            c6 = c6 + vec4<f32>(b1.z) * a;
            c7 = c7 + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }
    var acc: array<vec4<f32>, 8>;
    acc[0] = c0; acc[1] = c1; acc[2] = c2; acc[3] = c3;
    acc[4] = c4; acc[5] = c5; acc[6] = c6; acc[7] = c7;
    let col0 = colbase + lid.x * 4u;
    for (var jr: u32 = 0u; jr < 8u; jr = jr + 1u) {
        let lr = rowbase + lid.y * 8u + jr;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[jr & 7u];
            let o = lr * d + col0;
            if (col0 < d) { h[o] = h[o] + v.x; }
            if (col0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (col0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (col0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
