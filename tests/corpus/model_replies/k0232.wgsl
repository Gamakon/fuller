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
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        acc[i & 7u] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let q = idx & 15u;
            let c0 = wid.x * 64u + q * 4u;
            let l0 = wid.y * 64u + q * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (c0 + j < cfg.d) {
                        a[j] = phi[base + c0 + j];
                    }
                    if (l0 + j < cfg.r_rows && cfg.r_off + l0 + j < cfg.d) {
                        b[j] = phi[base + cfg.r_off + l0 + j];
                    }
                }
            }
            wg_va[idx & 511u] = a;
            wg_vb[idx & 511u] = b;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            let a = wg_va[(s * 16u + lid.x) & 511u];
            let b0 = wg_vb[(s * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(s * 16u + lid.y * 2u + 1u) & 511u];
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
    let c0 = wid.x * 64u + lid.x * 4u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let v = acc[i & 7u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (c0 + j < cfg.d) {
                    let o = lr * cfg.d + c0 + j;
                    h[o] = h[o] + v[j];
                }
            }
        }
    }
}
