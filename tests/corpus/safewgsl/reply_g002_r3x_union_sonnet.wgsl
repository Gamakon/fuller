// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let tt = idx >> 6u;
            let c = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (t + tt < m) {
                let lr = rbase + c;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    f0 = phi[(t + tt) * d + cfg.r_off + lr];
                }
                if (cbase + c < d) {
                    f1 = phi[(t + tt) * d + cbase + c];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 2u) {
            var pc: array<f32, 4>;
            var pd: array<f32, 4>;
            var pe: array<f32, 4>;
            var pf: array<f32, 4>;
            for (var j = 0u; j < 4u; j = j + 1u) {
                pc[j] = wg_a[k * 64u + lid.y + 16u * j];
                pd[j] = wg_b[k * 64u + lid.x + 16u * j];
                pe[j] = wg_a[k * 64u + 64u + lid.y + 16u * j];
                pf[j] = wg_b[k * 64u + 64u + lid.x + 16u * j];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var l = 0u; l < 4u; l = l + 1u) {
                    pa[j * 4u + l] = pa[j * 4u + l] + pc[j] * pd[l];
                    pb[j * 4u + l] = pb[j * 4u + l] + pe[j] * pf[l];
                }
            }
        }
        workgroupBarrier();
    }
    for (var k = 0u; k < 4u; k = k + 1u) {
        let row = rbase + lid.y + 16u * k;
        if (row < cfg.r_rows && cfg.r_off + row < d) {
            for (var k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = cbase + lid.x + 16u * k2;
                if (col < d) {
                    let i = row * d + col;
                    h[i] = h[i] + (pa[k * 4u + k2] + pb[k * 4u + k2]);
                }
            }
        }
    }
}
