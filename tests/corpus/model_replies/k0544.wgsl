// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var kk: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let id = lidx + 256u * k;
            let tr = t + (id >> 6u);
            let cc = id & 63u;
            f0 = 0.0;
            f1 = 0.0;
            if (tr < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    f0 = phi[tr * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    f1 = phi[tr * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[id] = f0;
            wg_b[id] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (kk = 0u; kk < 32u; kk = kk + 1u) {
            for (k = 0u; k < 8u; k = k + 1u) {
                let bv = wg_b[kk * 64u + lid.y + 8u * k];
                for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                    pb[k * 2u + k2] = pb[k * 2u + k2] + bv * wg_a[kk * 64u + lid.x + 32u * k2];
                }
            }
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let row = wid.y * 64u + lid.y + 8u * k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x + 32u * k2;
                if (col < cfg.d) {
                    h[row * cfg.d + col] = h[row * cfg.d + col] + pa[k * 2u + k2];
                }
            }
        }
    }
}
