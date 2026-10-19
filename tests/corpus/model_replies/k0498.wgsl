// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let col = idx & 63u;
            let item = t + (idx >> 6u);
            var f0: f32 = 0.0f;
            var f1: f32 = 0.0f;
            if (item < cfg.m) {
                if ((wid.x * 64u + col) < cfg.d) {
                    f0 = phi[item * cfg.d + wid.x * 64u + col];
                }
                if (((wid.y * 64u + col) < cfg.r_rows) && ((cfg.r_off + wid.y * 64u + col) < cfg.d)) {
                    f1 = phi[item * cfg.d + cfg.r_off + wid.y * 64u + col];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0f;
        }
        for (k = 0u; k < 32u; k = k + 2u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let b0 = wg_b[k * 64u + lid.y + 16u * k2];
                let b1 = wg_b[k * 64u + 64u + lid.y + 16u * k2];
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    pb[k2 * 4u + k3] = pb[k2 * 4u + k3] + (b0 * wg_a[k * 64u + lid.x + 16u * k3] + b1 * wg_a[k * 64u + 64u + lid.x + 16u * k3]);
                }
            }
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 16u * k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
                }
            }
        }
    }
}
