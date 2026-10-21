// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tr = t + (idx >> 6u);
            let cc = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (tr < cfg.m) {
                if (wid.x * 64u + cc < cfg.d) {
                    fa = phi[tr * cfg.d + wid.x * 64u + cc];
                }
                if (wid.y * 64u + cc < cfg.r_rows && cfg.r_off + wid.y * 64u + cc < cfg.d) {
                    fb = phi[tr * cfg.d + cfg.r_off + wid.y * 64u + cc];
                }
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pb[k] = 0.0;
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_a[k * 64u + lid.x];
            let a1 = wg_a[k * 64u + lid.x + 32u];
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let bv = wg_b[k * 64u + lid.y + 8u * i];
                pb[i * 2u] = pb[i * 2u] + bv * a0;
                pb[i * 2u + 1u] = pb[i * 2u + 1u] + bv * a1;
            }
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pa[k] = pa[k] + pb[k];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 8u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 32u * j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + pa[i * 2u + j];
                }
            }
        }
    }
}
