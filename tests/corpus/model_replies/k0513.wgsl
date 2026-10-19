// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tt = idx >> 6u;
            let cc = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (t + tt < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    f0 = phi[(t + tt) * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    f1 = phi[(t + tt) * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let ob = k * 64u + lid.y * 4u;
            let oa = k * 64u + lid.x * 4u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let bv = wg_b[ob + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pb[i * 4u + j] = pb[i * 4u + j] + bv * wg_a[oa + j];
                }
            }
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x * 4u + j;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[i * 4u + j];
                }
            }
        }
    }
}
