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
    var q: array<f32, 64>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = t + (idx >> 6u);
            let cc = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (tr < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    f0 = phi[tr * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    f1 = phi[tr * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                q[k] = 0.0;
            }
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                let base = (hh * 16u + k) * 64u;
                let b0 = wg_b[base + lid.y];
                let b1 = wg_b[base + lid.y + 16u];
                let b2 = wg_b[base + lid.y + 32u];
                let b3 = wg_b[base + lid.y + 48u];
                let a0 = wg_a[base + lid.x];
                let a1 = wg_a[base + lid.x + 16u];
                let a2 = wg_a[base + lid.x + 32u];
                let a3 = wg_a[base + lid.x + 48u];
                q[0] = q[0] + b0 * a0; q[1] = q[1] + b0 * a1; q[2] = q[2] + b0 * a2; q[3] = q[3] + b0 * a3;
                q[4] = q[4] + b1 * a0; q[5] = q[5] + b1 * a1; q[6] = q[6] + b1 * a2; q[7] = q[7] + b1 * a3;
                q[8] = q[8] + b2 * a0; q[9] = q[9] + b2 * a1; q[10] = q[10] + b2 * a2; q[11] = q[11] + b2 * a3;
                q[12] = q[12] + b3 * a0; q[13] = q[13] + b3 * a1; q[14] = q[14] + b3 * a2; q[15] = q[15] + b3 * a3;
            }
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                pa[k] = pa[k] + q[k];
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 16u * k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
                }
            }
        }
    }
}
