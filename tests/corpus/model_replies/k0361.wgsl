// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var kk: u32 = 0u;
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            f0 = 0.0f;
            f1 = 0.0f;
            if ((t + ((lidx + (k * 256u)) / 64u)) < cfg.m) {
                if (((wid.y * 64u) + ((lidx + (k * 256u)) % 64u)) < cfg.r_rows && ((cfg.r_off + (wid.y * 64u)) + ((lidx + (k * 256u)) % 64u)) < cfg.d) {
                    f0 = phi[((((t + ((lidx + (k * 256u)) / 64u)) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + ((lidx + (k * 256u)) % 64u)];
                }
                if (((wid.x * 64u) + ((lidx + (k * 256u)) % 64u)) < cfg.d) {
                    f1 = phi[(((t + ((lidx + (k * 256u)) / 64u)) * cfg.d) + (wid.x * 64u)) + ((lidx + (k * 256u)) % 64u)];
                }
            }
            wg_a[(lidx + (k * 256u)) & 2047u] = f0;
            wg_b[(lidx + (k * 256u)) & 2047u] = f1;
        }
        workgroupBarrier();
        for (kk = 0u; kk < 4u; kk = kk + 1u) {
            for (k = 0u; k < 16u; k = k + 1u) {
                pb[k & 63u] = 0.0f;
            }
            for (k = 0u; k < 4u; k = k + 1u) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                        pb[((k2 * 4u) + k3) & 63u] = pb[((k2 * 4u) + k3) & 63u] + (wg_a[((((kk * 4u) + k) * 64u) + (lid.y * 4u) + k2) & 2047u] * wg_b[((((kk * 4u) + k) * 64u) + (lid.x * 4u) + k3) & 2047u]);
                    }
                }
            }
            for (k = 0u; k < 16u; k = k + 1u) {
                pa[k & 63u] = pa[k & 63u] + pb[k & 63u];
            }
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if (((gid.y * 4u) + min(k, 3u)) < cfg.r_rows && ((gid.x * 4u) + min(k2, 3u)) < cfg.d && (cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d) {
                h[(((gid.y * 4u) + min(k, 3u)) * cfg.d) + ((gid.x * 4u) + min(k2, 3u))] = h[(((gid.y * 4u) + min(k, 3u)) * cfg.d) + ((gid.x * 4u) + min(k2, 3u))] + pa[((k * 4u) + k2) & 63u];
            }
        }
    }
}
