// grid 16 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 512>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let colBase = wid.x * 128u;
    let rowBase = wid.y * 64u;
    var acc: array<vec4<f32>, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = vec4<f32>(0.0);
    }
    for (var t = 0u; t < m; t = t + 16u) {
        // load A tile (columns)
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let item = t + (e >> 5u);
            let col = colBase + (e & 31u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d + col;
                if (col < d) { v.x = phi[base]; }
                if (col + 1u < d) { v.y = phi[base + 1u]; }
                if (col + 2u < d) { v.z = phi[base + 2u]; }
                if (col + 3u < d) { v.w = phi[base + 3u]; }
            }
            wa[e & 511u] = v;
        }
        // load B tile (rows)
        for (var k = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 128u * k;
            let item = t + (e >> 4u);
            let lr = rowBase + (e & 15u) * 4u;
            var v = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d + cfg.r_off + lr;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) { v.x = phi[base]; }
                if (lr + 1u < cfg.r_rows && cfg.r_off + lr + 1u < d) { v.y = phi[base + 1u]; }
                if (lr + 2u < cfg.r_rows && cfg.r_off + lr + 2u < d) { v.z = phi[base + 2u]; }
                if (lr + 3u < cfg.r_rows && cfg.r_off + lr + 3u < d) { v.w = phi[base + 3u]; }
            }
            wb[e & 255u] = v;
        }
        workgroupBarrier();
        for (var s = 0u; s < 16u; s = s + 1u) {
            let a0 = wa[(s * 32u + tx) & 511u];
            let a1 = wa[(s * 32u + tx + 16u) & 511u];
            let b0 = wb[(s * 16u + ty) & 255u];
            let b1 = wb[(s * 16u + ty + 8u) & 255u];
            var bv: array<f32, 8>;
            bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w;
            bv[4] = b1.x; bv[5] = b1.y; bv[6] = b1.z; bv[7] = b1.w;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let bs = vec4<f32>(bv[i & 7u]);
                acc[(i * 2u) & 15u] = acc[(i * 2u) & 15u] + bs * a0;
                acc[(i * 2u + 1u) & 15u] = acc[(i * 2u + 1u) & 15u] + bs * a1;
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        var lr = rowBase + ty * 4u + i;
        if (i >= 4u) { lr = rowBase + 32u + ty * 4u + (i - 4u); }
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var g = 0u; g < 2u; g = g + 1u) {
                let c0 = colBase + tx * 4u + 64u * g;
                let a = acc[(i * 2u + g) & 15u];
                let o = lr * d + c0;
                if (c0 < d) { h[o] = h[o] + a.x; }
                if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + a.y; }
                if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + a.z; }
                if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + a.w; }
            }
        }
    }
}
