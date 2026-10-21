// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    let lin = lid.y * 16u + lid.x;
    let d = cfg.d;
    for (var t = 0u; t < cfg.m; t = t + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let idx = lin + k * 256u;
            let it = t + (idx >> 6u);
            let c6 = idx & 63u;
            var f0 = 0.0;
            var f1 = 0.0;
            if (it < cfg.m) {
                let lrw = wid.y * 64u + c6;
                if (lrw < cfg.r_rows && cfg.r_off + lrw < d) {
                    f0 = phi[it * d + cfg.r_off + lrw];
                }
                let cw = wid.x * 64u + c6;
                if (cw < d) {
                    f1 = phi[it * d + cw];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 2u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            var rc: array<f32, 4>;
            var rd: array<f32, 4>;
            for (var q = 0u; q < 4u; q = q + 1u) {
                ra[q] = wg_a[k * 64u + lid.y + 16u * q];
                rb[q] = wg_b[k * 64u + lid.x + 16u * q];
                rc[q] = wg_a[k * 64u + 64u + lid.y + 16u * q];
                rd[q] = wg_b[k * 64u + 64u + lid.x + 16u * q];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + ra[i] * rb[j];
                    pb[i * 4u + j] = pb[i * 4u + j] + rc[i] * rd[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + (pa[i * 4u + j] + pb[i * 4u + j]);
                }
            }
        }
    }
}
