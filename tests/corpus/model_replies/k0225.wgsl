// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        acc[i & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < cfg.m; t = t + 32u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let item = t + (e >> 4u);
            let q = (e & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                let c = col0 + q;
                if (c < d) { a.x = phi[base + c]; }
                if (c + 1u < d) { a.y = phi[base + c + 1u]; }
                if (c + 2u < d) { a.z = phi[base + c + 2u]; }
                if (c + 3u < d) { a.w = phi[base + c + 3u]; }
                let lr = row0 + q;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) { b.x = phi[base + cfg.r_off + lr]; }
                if (lr + 1u < cfg.r_rows && cfg.r_off + lr + 1u < d) { b.y = phi[base + cfg.r_off + lr + 1u]; }
                if (lr + 2u < cfg.r_rows && cfg.r_off + lr + 2u < d) { b.z = phi[base + cfg.r_off + lr + 2u]; }
                if (lr + 3u < cfg.r_rows && cfg.r_off + lr + 3u < d) { b.w = phi[base + cfg.r_off + lr + 3u]; }
            }
            wg_va[e & 511u] = a;
            wg_vb[e & 511u] = b;
        }
        workgroupBarrier();
        for (var k = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b0 = wg_vb[(k * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(k * 16u + lid.y * 2u + 1u) & 511u];
            acc[0u] = acc[0u] + b0.x * a;
            acc[1u] = acc[1u] + b0.y * a;
            acc[2u] = acc[2u] + b0.z * a;
            acc[3u] = acc[3u] + b0.w * a;
            acc[4u] = acc[4u] + b1.x * a;
            acc[5u] = acc[5u] + b1.y * a;
            acc[6u] = acc[6u] + b1.z * a;
            acc[7u] = acc[7u] + b1.w * a;
        }
        workgroupBarrier();
    }
    let c = col0 + lid.x * 4u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[i & 7u];
            let idx = lr * d + c;
            if (c < d) { h[idx] = h[idx] + v.x; }
            if (c + 1u < d) { h[idx + 1u] = h[idx + 1u] + v.y; }
            if (c + 2u < d) { h[idx + 2u] = h[idx + 2u] + v.z; }
            if (c + 3u < d) { h[idx + 3u] = h[idx + 3u] + v.w; }
        }
    }
}
