// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            f0 = 0.0f;
            f1 = 0.0f;
            let q = lidx + 256u * k;
            let row = t + (q >> 6u);
            let cc = q & 63u;
            if (row < cfg.m) {
                if ((wid.x * 64u + cc) < cfg.d) {
                    f0 = phi[row * cfg.d + wid.x * 64u + cc];
                }
                if (((wid.y * 64u + cc) < cfg.r_rows) && ((cfg.r_off + wid.y * 64u + cc) < cfg.d)) {
                    f1 = phi[row * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[q] = f0;
            wg_b[q] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0f;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                for (k3 = 0u; k3 < 2u; k3 = k3 + 1u) {
                    pb[k2 * 2u + k3] = pb[k2 * 2u + k3] + wg_b[k * 64u + lid.y * 8u + k2] * wg_a[k * 64u + lid.x + 32u * k3];
                }
            }
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y * 8u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 32u * k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 2u + k2];
                }
            }
        }
    }
}
