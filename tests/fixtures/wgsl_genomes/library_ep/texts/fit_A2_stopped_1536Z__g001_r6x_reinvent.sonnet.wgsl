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
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 256u * k;
            let it = t + (e >> 6u);
            let cc = e & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (it < cfg.m) {
                if (colBase + cc < cfg.d) {
                    f0 = phi[it * cfg.d + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < cfg.d) {
                    f1 = phi[it * cfg.d + cfg.r_off + rowBase + cc];
                }
            }
            wg_a[e] = f0;
            wg_b[e] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let bv = wg_b[k * 64u + lid.y + 16u * k2];
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    pb[k2 * 4u + k3] = pb[k2 * 4u + k3] + bv * wg_a[k * 64u + lid.x + 16u * k3];
                }
            }
        }
        for (k = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
        let lr = rowBase + lid.y + 16u * k2;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                let c = colBase + lid.x + 16u * k3;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pa[k2 * 4u + k3];
                }
            }
        }
    }
}
