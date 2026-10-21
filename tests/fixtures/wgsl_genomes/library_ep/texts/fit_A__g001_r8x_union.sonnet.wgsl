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
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    let li = lid.y * 16u + lid.x;
    for (t = 0u; t < cfg.m; t = t + 8u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let id = li + 256u * k;
            let tr = t + (id >> 6u);
            let cc = id & 63u;
            let lrow = wid.y * 64u + cc;
            let ccol = wid.x * 64u + cc;
            f0 = 0.0;
            f1 = 0.0;
            if (tr < cfg.m) {
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < cfg.d) {
                    f0 = phi[tr * cfg.d + cfg.r_off + lrow];
                }
                if (ccol < cfg.d) {
                    f1 = phi[tr * cfg.d + ccol];
                }
            }
            wg_a[id] = f0;
            wg_b[id] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 4u; k = k + 1u) {
            let o0 = (2u * k) * 64u;
            let o1 = (2u * k + 1u) * 64u;
            for (i = 0u; i < 4u; i = i + 1u) {
                let a0 = wg_a[o0 + lid.y + 16u * i];
                let a1 = wg_a[o1 + lid.y + 16u * i];
                for (j = 0u; j < 4u; j = j + 1u) {
                    let n = i * 4u + j;
                    pa[n] = fma(a0, wg_b[o0 + lid.x + 16u * j], pa[n]);
                    pb[n] = fma(a1, wg_b[o1 + lid.x + 16u * j], pb[n]);
                }
            }
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        let row = wid.y * 64u + lid.y + 16u * i;
        if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 64u + lid.x + 16u * j;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + (pa[i * 4u + j] + pb[i * 4u + j]);
                }
            }
        }
    }
}
