// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var tot: array<vec4<f32>, 4>;
    tot[0] = vec4<f32>(0.0);
    tot[1] = vec4<f32>(0.0);
    tot[2] = vec4<f32>(0.0);
    tot[3] = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = idx >> 6u;
            let col = idx & 63u;
            let item = t + row;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                if (colbase + col < cfg.d) {
                    va = phi[item * cfg.d + colbase + col];
                }
                if (rowbase + col < cfg.r_rows && cfg.r_off + rowbase + col < cfg.d) {
                    vb = phi[item * cfg.d + cfg.r_off + rowbase + col];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        var acc: array<vec4<f32>, 4>;
        acc[0] = vec4<f32>(0.0);
        acc[1] = vec4<f32>(0.0);
        acc[2] = vec4<f32>(0.0);
        acc[3] = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a0 = wg_a[k * 64u + lid.x];
            let a1 = wg_a[k * 64u + lid.x + 32u];
            let av = vec4<f32>(a0, a1, a0, a1);
            let bo = k * 64u + lid.y * 8u;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let b0 = wg_b[bo + 2u * q];
                let b1 = wg_b[bo + 2u * q + 1u];
                acc[q] = acc[q] + vec4<f32>(b0, b0, b1, b1) * av;
            }
        }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            tot[q] = tot[q] + acc[q];
        }
        workgroupBarrier();
    }
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let lr = rowbase + lid.y * 8u + 2u * q + s;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                let v = tot[q];
                let c0 = colbase + lid.x;
                let c1 = c0 + 32u;
                var x0: f32 = v.x;
                var x1: f32 = v.y;
                if (s == 1u) {
                    x0 = v.z;
                    x1 = v.w;
                }
                if (c0 < cfg.d) {
                    h[lr * cfg.d + c0] = h[lr * cfg.d + c0] + x0;
                }
                if (c1 < cfg.d) {
                    h[lr * cfg.d + c1] = h[lr * cfg.d + c1] + x1;
                }
            }
        }
    }
}
