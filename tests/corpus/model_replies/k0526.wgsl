// grid 8 32 8 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(8, 32, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var t: u32 = 0u; var k: u32 = 0u; var i: u32 = 0u; var j: u32 = 0u; var n: u32 = 0u;
    var f0: f32 = 0.0; var f1: f32 = 0.0; var bv: f32 = 0.0;
    for (n = 0u; n < 32u; n = n + 1u) {
        pa[n] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 256u * k;
            let row = t + (e >> 6u);
            let cc = wid.x * 64u + (e & 63u);
            let rr = wid.y * 64u + (e & 63u);
            f0 = 0.0;
            f1 = 0.0;
            if (row < cfg.m) {
                if (cc < cfg.d) {
                    f0 = phi[row * cfg.d + cc];
                }
                if (rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) {
                    f1 = phi[row * cfg.d + cfg.r_off + rr];
                }
            }
            wg_a[e] = f0;
            wg_b[e] = f1;
        }
        workgroupBarrier();
        for (n = 0u; n < 16u; n = n + 1u) {
            pb[n] = 0.0;
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            for (i = 0u; i < 2u; i = i + 1u) {
                bv = wg_b[k * 64u + lid.y + 32u * i];
                for (j = 0u; j < 8u; j = j + 1u) {
                    pb[i * 8u + j] = pb[i * 8u + j] + bv * wg_a[k * 64u + lid.x + 8u * j];
                }
            }
        }
        let par = (t >> 5u) & 1u;
        for (n = 0u; n < 16u; n = n + 1u) {
            pa[par * 16u + n] = pa[par * 16u + n] + pb[n];
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 2u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 32u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 8u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 8u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + (pa[i * 8u + j] + pa[i * 8u + j + 16u]);
                }
            }
        }
    }
}
