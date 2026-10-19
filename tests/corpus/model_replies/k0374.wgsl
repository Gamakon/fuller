// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var pd: array<f32, 64>;
    var pi: array<u32, 16>;
    var pj: array<u32, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
        pi[k] = min((cfg.r_off + (gid.y * 8u)) + k, cfg.d - 1u);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        pj[k] = min((gid.x * 4u) + k, cfg.d - 1u);
    }
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 32u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            if ((t + k) < cfg.m) {
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
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let lr = (gid.y * 8u) + k;
            let c = (gid.x * 4u) + k2;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[k * 4u + k2];
            }
        }
    }
}
