// grid 16 8 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>;
    var pi: array<u32, 16>; var pj: array<u32, 16>;
    let rb = (wid.y * 32u) + lid.y;
    let cb = (wid.x * 128u) + lid.x;
    for (k = 0u; k < 4u; k = k + 1u) {
        pi[((k) & 15u)] = min(((cfg.r_off + rb) + (k * 8u)), (cfg.d - 1u));
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        pj[((k) & 15u)] = min((cb + (k * 16u)), (cfg.d - 1u));
    }
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 32u; k = k + 1u) {
            pb[((k) & 63u)] = 0.0f;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            if (((t + k) < cfg.m)) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    pc[((k2) & 63u)] = phi[(((t + k) * cfg.d) + pi[((k2) & 15u)])];
                }
                for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                    pd[((k2) & 63u)] = phi[(((t + k) * cfg.d) + pj[((k2) & 15u)])];
                }
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    for (k3 = 0u; k3 < 8u; k3 = k3 + 1u) {
                        pb[((((k2 * 8u) + k3)) & 63u)] = (pb[((((k2 * 8u) + k3)) & 63u)] + (pc[((k2) & 63u)] * pd[((k3) & 63u)]));
                    }
                }
            }
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            pa[((k) & 63u)] = (pa[((k) & 63u)] + pb[((k) & 63u)]);
        }
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
            let lr = rb + (min(k, 3u) * 8u);
            let c = cb + (min(k2, 7u) * 16u);
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                h[(lr * cfg.d + c)] = h[(lr * cfg.d + c)] + pa[((((k * 8u) + k2)) & 63u)];
            }
        }
    }
}
