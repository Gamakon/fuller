// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    var u0 = vec4<f32>(0.0); var u1 = vec4<f32>(0.0); var u2 = vec4<f32>(0.0); var u3 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    let dm = cfg.d - 1u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 256u * k;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (t + (e >> 6u) < cfg.m) {
                let base = (t + (e >> 6u)) * cfg.d;
                f0 = phi[base + min(wid.x * 64u + (e & 63u), dm)];
                f1 = phi[base + min(cfg.r_off + wid.y * 64u + (e & 63u), dm)];
            }
            wg_a[e] = f0;
            wg_b[e] = f1;
        }
        workgroupBarrier();
        var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(wg_a[o + lid.x], wg_a[o + lid.x + 32u], wg_a[o + lid.x], wg_a[o + lid.x + 32u]);
            let rb = o + lid.y * 8u;
            let b0 = wg_b[rb]; let b1 = wg_b[rb + 1u]; let b2 = wg_b[rb + 2u]; let b3 = wg_b[rb + 3u];
            let b4 = wg_b[rb + 4u]; let b5 = wg_b[rb + 5u]; let b6 = wg_b[rb + 6u]; let b7 = wg_b[rb + 7u];
            v0 = v0 + a * vec4<f32>(b0, b0, b1, b1);
            v1 = v1 + a * vec4<f32>(b2, b2, b3, b3);
            v2 = v2 + a * vec4<f32>(b4, b4, b5, b5);
            v3 = v3 + a * vec4<f32>(b6, b6, b7, b7);
        }
        if (((t >> 5u) & 1u) == 0u) {
            s0 = s0 + v0; s1 = s1 + v1; s2 = s2 + v2; s3 = s3 + v3;
        } else {
            u0 = u0 + v0; u1 = u1 + v1; u2 = u2 + v2; u3 = u3 + v3;
        }
        workgroupBarrier();
    }
    pv[0] = s0 + u0; pv[1] = s1 + u1; pv[2] = s2 + u2; pv[3] = s3 + u3;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let lr = wid.y * 64u + lid.y * 8u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let v = pv[j >> 1u];
            let q = (j & 1u) * 2u;
            let c0 = wid.x * 64u + lid.x;
            let c1 = c0 + 32u;
            if (c0 < cfg.d) {
                h[lr * cfg.d + c0] = h[lr * cfg.d + c0] + v[q];
            }
            if (c1 < cfg.d) {
                h[lr * cfg.d + c1] = h[lr * cfg.d + c1] + v[q + 1u];
            }
        }
    }
}
