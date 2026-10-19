// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    for (k = 0u; k < 32u; k = k + 1u) {
        pc[k] = 0.0;
    }
    for (t = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (k = 0u; k < 32u; k = k + 1u) {
            pa[k] = 0.0;
            pb[k] = 0.0;
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            let e = lidx + k * 128u;
            let item = t * 32u + e / 64u;
            let cc = e % 64u;
            f0 = 0.0;
            if (item < cfg.m && (wid.x * 64u + cc) < cfg.d) {
                f0 = phi[item * cfg.d + wid.x * 64u + cc];
            }
            wg_a[e & 2047u] = f0;
            f1 = 0.0;
            if (item < cfg.m && (wid.y * 64u + cc) < cfg.r_rows && (cfg.r_off + wid.y * 64u + cc) < cfg.d) {
                f1 = phi[item * cfg.d + cfg.r_off + wid.y * 64u + cc];
            }
            wg_b[e & 2047u] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 32u; k = k + 2u) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    let q = k2 * 4u + k3;
                    pa[q] = pa[q] + wg_b[(k * 64u + lid.y * 8u + k2) & 2047u] * wg_a[(k * 64u + lid.x * 4u + k3) & 2047u];
                    pb[q] = pb[q] + wg_b[((k + 1u) * 64u + lid.y * 8u + k2) & 2047u] * wg_a[((k + 1u) * 64u + lid.x * 4u + k3) & 2047u];
                }
            }
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            pc[k] = pc[k] + (pa[k] + pb[k]);
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if (((gid.y * 8u) + k) < cfg.r_rows && ((gid.x * 4u) + k2) < cfg.d && (cfg.r_off + (gid.y * 8u) + k) < cfg.d) {
                let idx = ((gid.y * 8u) + k) * cfg.d + (gid.x * 4u) + k2;
                h[idx] = h[idx] + pc[k * 4u + k2];
            }
        }
    }
}
