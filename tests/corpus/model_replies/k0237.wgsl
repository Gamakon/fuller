// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        acc[j & 3u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    for (var t = 0u; t < cfg.m; t = t + 32u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 4u);
            let cq = (idx & 15u) * 4u;
            let col0 = wid.x * 64u + cq;
            let row0 = wid.y * 64u + cq;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var e = 0u; e < 4u; e = e + 1u) {
                    if (col0 + e < d) {
                        a[e & 3u] = phi[base + col0 + e];
                    }
                    if (row0 + e < cfg.r_rows && cfg.r_off + row0 + e < d) {
                        b[e & 3u] = phi[base + cfg.r_off + row0 + e];
                    }
                }
            }
            wg_va[idx & 511u] = a;
            wg_vb[idx & 511u] = b;
        }
        workgroupBarrier();
        for (var i = 0u; i < 32u; i = i + 1u) {
            let a = wg_va[(i * 16u + lid.x) & 511u];
            let b = wg_vb[(i * 16u + lid.y) & 511u];
            acc[0u] = acc[0u] + b.x * a;
            acc[1u] = acc[1u] + b.y * a;
            acc[2u] = acc[2u] + b.z * a;
            acc[3u] = acc[3u] + b.w * a;
        }
        workgroupBarrier();
    }
    let col0 = (wid.x * 16u + lid.x) * 4u;
    let row0 = (wid.y * 16u + lid.y) * 4u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var e = 0u; e < 4u; e = e + 1u) {
                let c = col0 + e;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[j & 3u][e & 3u];
                }
            }
        }
    }
}
