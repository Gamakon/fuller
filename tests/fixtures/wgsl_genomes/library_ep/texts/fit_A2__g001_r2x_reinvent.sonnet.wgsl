// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0: vec4<f32> = vec4<f32>(0.0);
    var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0);
    var acc3: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let i = lidx + 256u * j;
            let item = t + (i >> 4u);
            let c = (i & 15u) * 4u;
            var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
            var b0: f32 = 0.0; var b1: f32 = 0.0; var b2: f32 = 0.0; var b3: f32 = 0.0;
            if (item < cfg.m) {
                let base = item * d;
                let gc = wid.x * 64u + c;
                if (gc < d) { a0 = phi[base + gc]; }
                if (gc + 1u < d) { a1 = phi[base + gc + 1u]; }
                if (gc + 2u < d) { a2 = phi[base + gc + 2u]; }
                if (gc + 3u < d) { a3 = phi[base + gc + 3u]; }
                let lr = wid.y * 64u + c;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { b0 = phi[base + gr]; }
                if (lr + 1u < cfg.r_rows && gr + 1u < d) { b1 = phi[base + gr + 1u]; }
                if (lr + 2u < cfg.r_rows && gr + 2u < d) { b2 = phi[base + gr + 2u]; }
                if (lr + 3u < cfg.r_rows && gr + 3u < d) { b3 = phi[base + gr + 3u]; }
            }
            wg_va[i] = vec4<f32>(a0, a1, a2, a3);
            wg_vb[i] = vec4<f32>(b0, b1, b2, b3);
        }
        workgroupBarrier();
        var t0: vec4<f32> = vec4<f32>(0.0);
        var t1: vec4<f32> = vec4<f32>(0.0);
        var t2: vec4<f32> = vec4<f32>(0.0);
        var t3: vec4<f32> = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a = wg_va[kk * 16u + lid.x];
            let b = wg_vb[kk * 16u + lid.y];
            t0 = t0 + vec4<f32>(b.x) * a;
            t1 = t1 + vec4<f32>(b.y) * a;
            t2 = t2 + vec4<f32>(b.z) * a;
            t3 = t3 + vec4<f32>(b.w) * a;
        }
        acc0 = acc0 + t0;
        acc1 = acc1 + t1;
        acc2 = acc2 + t2;
        acc3 = acc3 + t3;
        workgroupBarrier();
    }
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 4u;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = row0 + r;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v: vec4<f32> = acc0;
            if (r == 1u) { v = acc1; }
            if (r == 2u) { v = acc2; }
            if (r == 3u) { v = acc3; }
            let o = lr * d + col0;
            if (col0 < d) { h[o] = h[o] + v.x; }
            if (col0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (col0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (col0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
