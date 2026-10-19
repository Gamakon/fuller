// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var ra: array<f32, 64>;
    var rb: array<f32, 64>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        pa[q] = 0.0;
        pb[q] = 0.0;
    }
    let lin = lid.y * 16u + lid.x;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var kk: u32 = 0u; kk < 2u; kk = kk + 1u) {
            let idx = lin + 256u * kk;
            let item = t + (idx >> 6u);
            let cl = idx & 63u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < cfg.m) {
                let lr = wid.y * 64u + cl;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    f0 = phi[item * cfg.d + cfg.r_off + lr];
                }
                let cc = wid.x * 64u + cl;
                if (cc < cfg.d) {
                    f1 = phi[item * cfg.d + cc];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let o0 = (2u * k) * 64u;
            let o1 = (2u * k + 1u) * 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                ra[i] = wg_a[o0 + lid.y * 4u + i];
                ra[4u + i] = wg_a[o1 + lid.y * 4u + i];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                rb[j] = wg_b[o0 + lid.x * 4u + j];
                rb[4u + j] = wg_b[o1 + lid.x * 4u + j];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = fma(ra[i], rb[j], pa[i * 4u + j]);
                    pb[i * 4u + j] = fma(ra[4u + i], rb[4u + j], pb[i * 4u + j]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let cc = wid.x * 64u + lid.x * 4u + j;
                if (cc < cfg.d) {
                    let ix = lr * cfg.d + cc;
                    h[ix] = h[ix] + (pa[i * 4u + j] + pb[i * 4u + j]);
                }
            }
        }
    }
}
