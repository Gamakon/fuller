// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let d = cfg.d;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let i = lidx + 256u * k2;
            let item = t + (i >> 4u);
            let c4 = (i & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                let gc = colBase + c4;
                if (gc < d) { a.x = phi[base + gc]; }
                if (gc + 1u < d) { a.y = phi[base + gc + 1u]; }
                if (gc + 2u < d) { a.z = phi[base + gc + 2u]; }
                if (gc + 3u < d) { a.w = phi[base + gc + 3u]; }
                let lr0 = rowBase + c4;
                let gr0 = cfg.r_off + lr0;
                if (lr0 < cfg.r_rows && gr0 < d) { b.x = phi[base + gr0]; }
                if (lr0 + 1u < cfg.r_rows && gr0 + 1u < d) { b.y = phi[base + gr0 + 1u]; }
                if (lr0 + 2u < cfg.r_rows && gr0 + 2u < d) { b.z = phi[base + gr0 + 2u]; }
                if (lr0 + 3u < cfg.r_rows && gr0 + 3u < d) { b.w = phi[base + gr0 + 3u]; }
            }
            wg_va[i] = a;
            wg_vb[i] = b;
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        var u0 = vec4<f32>(0.0);
        var u1 = vec4<f32>(0.0);
        var u2 = vec4<f32>(0.0);
        var u3 = vec4<f32>(0.0);
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let k0 = (2u * j) * 16u;
            let a0 = wg_va[k0 + lid.x];
            let b0 = wg_vb[k0 + lid.y];
            let a1 = wg_va[k0 + 16u + lid.x];
            let b1 = wg_vb[k0 + 16u + lid.y];
            s0 = s0 + vec4<f32>(b0.x) * a0;
            s1 = s1 + vec4<f32>(b0.y) * a0;
            s2 = s2 + vec4<f32>(b0.z) * a0;
            s3 = s3 + vec4<f32>(b0.w) * a0;
            u0 = u0 + vec4<f32>(b1.x) * a1;
            u1 = u1 + vec4<f32>(b1.y) * a1;
            u2 = u2 + vec4<f32>(b1.z) * a1;
            u3 = u3 + vec4<f32>(b1.w) * a1;
        }
        acc0 = acc0 + (s0 + u0);
        acc1 = acc1 + (s1 + u1);
        acc2 = acc2 + (s2 + u2);
        acc3 = acc3 + (s3 + u3);
        workgroupBarrier();
    }
    let c0 = colBase + lid.x * 4u;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = rowBase + lid.y * 4u + r;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = acc0;
            if (r == 1u) { v = acc1; }
            if (r == 2u) { v = acc2; }
            if (r == 3u) { v = acc3; }
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + v.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
