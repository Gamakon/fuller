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
    for (var r = 0u; r < 8u; r = r + 1u) { acc[r & 7u] = vec4<f32>(0.0); }
    let d = cfg.d;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < cfg.m; t = t + 32u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let j = idx >> 4u;
            let q = (idx & 15u) * 4u;
            let item = t + j;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let c = col0 + q + e;
                    if (c < d) { a[e & 3u] = phi[base + c]; }
                    let lr = row0 + q + e;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) { b[e & 3u] = phi[base + cfg.r_off + lr]; }
                }
            }
            wg_va[idx & 511u] = a;
            wg_vb[idx & 511u] = b;
        }
        workgroupBarrier();
        for (var j = 0u; j < 32u; j = j + 1u) {
            let a = wg_va[(j * 16u + lid.x) & 511u];
            let b0 = wg_vb[(j * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(j * 16u + lid.y * 2u + 1u) & 511u];
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
    for (var k = 0u; k < 8u; k = k + 1u) {
        let lr = row0 + lid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                let c = col0 + lid.x * 4u + e;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[k & 7u][e & 3u];
                }
            }
        }
    }
}
