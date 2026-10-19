// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 32>;
    var pb: array<f32, 32>;
    var ra: array<f32, 4>;
    var rb: array<f32, 8>;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + k * 128u;
            let it = idx / 64u;
            let c = idx % 64u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (t + it < cfg.m) {
                let lr = wid.y * 64u + c;
                if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                    f0 = phi[(t + it) * cfg.d + cfg.r_off + lr];
                }
                let gc = wid.x * 64u + c;
                if (gc < cfg.d) {
                    f1 = phi[(t + it) * cfg.d + gc];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                pb[k] = 0.0;
            }
            for (var s: u32 = 0u; s < 4u; s = s + 1u) {
                let kk = q * 4u + s;
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    ra[i] = wg_a[kk * 64u + lid.y + 16u * i];
                }
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    rb[j] = wg_b[kk * 64u + lid.x + 8u * j];
                }
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                        pb[i * 8u + j] = pb[i * 8u + j] + ra[i] * rb[j];
                    }
                }
            }
            for (var k: u32 = 0u; k < 32u; k = k + 1u) {
                pa[k] = pa[k] + pb[k];
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let c = wid.x * 64u + lid.x + 8u * j;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[i * 8u + j];
            }
        }
    }
}
