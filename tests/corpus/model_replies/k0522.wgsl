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
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            f0 = 0.0f;
            f1 = 0.0f;
            if ((t + ((lidx + (256u * k)) >> 6u)) < cfg.m) {
                if (((wid.x * 64u) + ((lidx + (256u * k)) & 63u)) < cfg.d) {
                    f0 = phi[(((t + ((lidx + (256u * k)) >> 6u)) * cfg.d) + (wid.x * 64u)) + ((lidx + (256u * k)) & 63u)];
                }
                if ((((wid.y * 64u) + ((lidx + (256u * k)) & 63u)) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + ((lidx + (256u * k)) & 63u)) < cfg.d)) {
                    f1 = phi[((((t + ((lidx + (256u * k)) >> 6u)) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + ((lidx + (256u * k)) & 63u)];
                }
            }
            wg_a[(lidx + (256u * k)) & 2047u] = f0;
            wg_b[(lidx + (256u * k)) & 2047u] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0f;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    pb[(k2 * 4u) + k3] = pb[(k2 * 4u) + k3] + (wg_b[(k * 64u) + (lid.y * 4u) + k2] * wg_a[(k * 64u) + (lid.x * 4u) + k3]);
                }
            }
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            pa[(((t >> 5u) & 1u) * 16u) + k] = pa[(((t >> 5u) & 1u) * 16u) + k] + pb[k];
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        if ((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + (gid.y * 4u) + k) < cfg.d)) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((gid.x * 4u) + k2) < cfg.d) {
                    let idx = (((gid.y * 4u) + k) * cfg.d) + (gid.x * 4u) + k2;
                    h[idx] = h[idx] + (pa[(k * 4u) + k2] + pa[16u + (k * 4u) + k2]);
                }
            }
        }
    }
}
