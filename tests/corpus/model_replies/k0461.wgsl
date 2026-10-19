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
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 8u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 6u);
            let j = idx & 63u;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (item < cfg.m) {
                let lr = wid.y * 64u + j;
                if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                    a = phi[item * cfg.d + cfg.r_off + lr];
                }
                let c = wid.x * 64u + j;
                if (c < cfg.d) {
                    b = phi[item * cfg.d + c];
                }
            }
            wg_a[idx] = a;
            wg_b[idx] = b;
        }
        workgroupBarrier();
        for (k = 0u; k < 4u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    let q = k2 * 4u + k3;
                    pa[q] = fma(wg_a[(2u * k) * 64u + lid.y + 16u * k2], wg_b[(2u * k) * 64u + lid.x + 16u * k3], pa[q]);
                    pb[q] = fma(wg_a[(2u * k + 1u) * 64u + lid.y + 16u * k2], wg_b[(2u * k + 1u) * 64u + lid.x + 16u * k3], pb[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = wid.x * 64u + lid.x + 16u * k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + (pa[k * 4u + k2] + pb[k * 4u + k2]);
                }
            }
        }
    }
}
