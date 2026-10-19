// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let g = lidx & 15u;
    let col0 = wid.x * 64u + g * 4u;
    let row0 = wid.y * 64u + g * 4u;
    for (var t = 0u; t < m; t = t + 32u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let s = lidx + 256u * k;
            let item = t + (s >> 4u);
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    if (col0 + q < d) {
                        a[q & 3u] = phi[base + col0 + q];
                    }
                    if (row0 + q < cfg.r_rows && cfg.r_off + row0 + q < d) {
                        b[q & 3u] = phi[base + cfg.r_off + row0 + q];
                    }
                }
            }
            wg_va[s & 511u] = a;
            wg_vb[s & 511u] = b;
        }
        workgroupBarrier();
        for (var j = 0u; j < 32u; j = j + 1u) {
            let a = wg_va[(j * 16u + lid.x) & 511u];
            let b = wg_vb[(j * 16u + lid.y) & 511u];
            acc0 = acc0 + vec4<f32>(b.x) * a;
            acc1 = acc1 + vec4<f32>(b.y) * a;
            acc2 = acc2 + vec4<f32>(b.z) * a;
            acc3 = acc3 + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    let cbase = (wid.x * 16u + lid.x) * 4u;
    let rbase = (wid.y * 16u + lid.y) * 4u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let lr = rbase + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = acc0;
            if (k == 1u) { v = acc1; }
            if (k == 2u) { v = acc2; }
            if (k == 3u) { v = acc3; }
            for (var q = 0u; q < 4u; q = q + 1u) {
                let c = cbase + q;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[q & 3u];
                }
            }
        }
    }
}
