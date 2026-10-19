// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        acc[j & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < cfg.m; t = t + 16u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let s = lidx + 128u * k;
            let item = t + (s >> 4u);
            let q = (s & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let c = col0 + q + e;
                    if (c < d) {
                        a[e] = phi[base + c];
                    }
                    let lr = row0 + q + e;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                        b[e] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_a[s & 255u] = a;
            wg_b[s & 255u] = b;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let a = wg_a[(kk * 16u + lid.x) & 255u];
            let b0 = wg_b[(kk * 16u + lid.y * 2u) & 255u];
            let b1 = wg_b[(kk * 16u + lid.y * 2u + 1u) & 255u];
            acc[0] = acc[0] + vec4<f32>(b0.x) * a;
            acc[1] = acc[1] + vec4<f32>(b0.y) * a;
            acc[2] = acc[2] + vec4<f32>(b0.z) * a;
            acc[3] = acc[3] + vec4<f32>(b0.w) * a;
            acc[4] = acc[4] + vec4<f32>(b1.x) * a;
            acc[5] = acc[5] + vec4<f32>(b1.y) * a;
            acc[6] = acc[6] + vec4<f32>(b1.z) * a;
            acc[7] = acc[7] + vec4<f32>(b1.w) * a;
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < 8u; j = j + 1u) {
        let lr = row0 + lid.y * 8u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[j & 7u];
            for (var e = 0u; e < 4u; e = e + 1u) {
                let c = col0 + lid.x * 4u + e;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[e];
                }
            }
        }
    }
}
