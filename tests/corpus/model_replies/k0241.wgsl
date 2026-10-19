// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<vec4<f32>, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        acc[j & 7u] = vec4<f32>(0.0);
    }
    for (var t = 0u; t < m; t = t + 32u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let q = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let c = col0 + q + e;
                    if (c < d) {
                        a[e & 3u] = phi[base + c];
                    }
                    let lr = row0 + q + e;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                        b[e & 3u] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_a[idx & 511u] = a;
            wg_b[idx & 511u] = b;
        }
        workgroupBarrier();
        for (var s = 0u; s < 32u; s = s + 1u) {
            let a = wg_a[(s * 16u + lid.x) & 511u];
            let b0 = wg_b[(s * 16u + lid.y * 2u) & 511u];
            let b1 = wg_b[(s * 16u + lid.y * 2u + 1u) & 511u];
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
    let cbase = col0 + lid.x * 4u;
    for (var j = 0u; j < 8u; j = j + 1u) {
        let lr = row0 + lid.y * 8u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[j & 7u];
            for (var e = 0u; e < 4u; e = e + 1u) {
                let c = cbase + e;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + v[e & 3u];
                }
            }
        }
    }
}
