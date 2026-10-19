// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    let li = lid.y + 8u * lid.x;
    let d = cfg.d;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = li + k * 128u;
            let item = t + (e >> 6u);
            let cc = e & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (item < cfg.m) {
                let c = colBase + cc;
                if (c < d) {
                    fa = phi[item * d + c];
                }
                let lr = rowBase + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    fb = phi[item * d + cfg.r_off + lr];
                }
            }
            wg_a[e] = fa;
            wg_b[e] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let ab = kk * 64u + lid.x * 4u;
            let a = vec4<f32>(wg_a[ab], wg_a[ab + 1u], wg_a[ab + 2u], wg_a[ab + 3u]);
            let bb = kk * 64u + lid.y * 8u;
            v0 = v0 + wg_b[bb] * a;
            v1 = v1 + wg_b[bb + 1u] * a;
            v2 = v2 + wg_b[bb + 2u] * a;
            v3 = v3 + wg_b[bb + 3u] * a;
            v4 = v4 + wg_b[bb + 4u] * a;
            v5 = v5 + wg_b[bb + 5u] * a;
            v6 = v6 + wg_b[bb + 6u] * a;
            v7 = v7 + wg_b[bb + 7u] * a;
        }
        workgroupBarrier();
    }
    let cb = colBase + lid.x * 4u;
    let rb = rowBase + lid.y * 8u;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        var v = v0;
        if (j == 1u) { v = v1; }
        if (j == 2u) { v = v2; }
        if (j == 3u) { v = v3; }
        if (j == 4u) { v = v4; }
        if (j == 5u) { v = v5; }
        if (j == 6u) { v = v6; }
        if (j == 7u) { v = v7; }
        let lr = rb + j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let base = lr * d + cb;
            if (cb < d) { h[base] = h[base] + v.x; }
            if (cb + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (cb + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (cb + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
