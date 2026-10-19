// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    var k: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    var t: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    let rowbase = wid.y * 64u;
    let colbase = wid.x * 64u;
    for (t = 0u; t < ((cfg.m + 15u) / 16u); t = t + 1u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let item = t * 16u + (e / 64u);
            let c = e % 64u;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < cfg.m) {
                let lr = rowbase + c;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    f0 = phi[item * cfg.d + cfg.r_off + lr];
                }
                if ((colbase + c) < cfg.d) {
                    f1 = phi[item * cfg.d + colbase + c];
                }
            }
            wg_a[e] = f0;
            wg_b[e] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 2u) {
            for (i = 0u; i < 8u; i = i + 1u) {
                let a0 = wg_a[k * 64u + lid.y + 8u * i];
                let a1 = wg_a[k * 64u + 64u + lid.y + 8u * i];
                for (j = 0u; j < 2u; j = j + 1u) {
                    let b0 = wg_b[k * 64u + lid.x + 32u * j];
                    let b1 = wg_b[k * 64u + 64u + lid.x + 32u * j];
                    pa[i * 2u + j] = pa[i * 2u + j] + a0 * b0;
                    pb[i * 2u + j] = pb[i * 2u + j] + a1 * b1;
                }
            }
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 8u; i = i + 1u) {
        let lr = rowbase + lid.y + 8u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 2u; j = j + 1u) {
                let c = colbase + lid.x + 32u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + (pa[i * 2u + j] + pb[i * 2u + j]);
                }
            }
        }
    }
}
