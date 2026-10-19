// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pd: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var t: u32 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
        pi[k] = min(cfg.r_off + wid.y * 64u + k * 8u + lid.y, cfg.d - 1u);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        pj[k] = min(wid.x * 64u + k * 16u + lid.x, cfg.d - 1u);
    }
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 32u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            if (t + k < cfg.m) {
                let base = (t + k) * cfg.d;
                for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                    pc[k2] = phi[base + pi[k2]];
                }
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    pd[k2] = phi[base + pj[k2]];
                }
                for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                        pb[k2 * 4u + k3] = pb[k2 * 4u + k3] + pc[k2] * pd[k3];
                    }
                }
            }
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        let lr = wid.y * 64u + k * 8u + lid.y;
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let c = wid.x * 64u + k2 * 16u + lid.x;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[k * 4u + k2];
            }
        }
    }
}
