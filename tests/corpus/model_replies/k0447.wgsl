// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var q: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let row0 = cfg.r_off + wid.y * 64u;
    let col0 = wid.x * 64u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (q = 0u; q < 4u; q = q + 1u) {
            let e = lidx + q * 256u;
            let kk = e / 64u;
            let cc = e % 64u;
            let tt = t + kk;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tt < cfg.m) {
                if (row0 + cc < cfg.d) {
                    va = phi[tt * cfg.d + row0 + cc];
                }
                if (col0 + cc < cfg.d) {
                    vb = phi[tt * cfg.d + col0 + cc];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_a[k * 64u + lid.y];
            let a1 = wg_a[k * 64u + lid.y + 16u];
            let a2 = wg_a[k * 64u + lid.y + 32u];
            let a3 = wg_a[k * 64u + lid.y + 48u];
            let b0 = wg_b[k * 64u + lid.x];
            let b1 = wg_b[k * 64u + lid.x + 16u];
            let b2 = wg_b[k * 64u + lid.x + 32u];
            let b3 = wg_b[k * 64u + lid.x + 48u];
            pa[0] = pa[0] + a0 * b0;
            pa[1] = pa[1] + a0 * b1;
            pa[2] = pa[2] + a0 * b2;
            pa[3] = pa[3] + a0 * b3;
            pa[4] = pa[4] + a1 * b0;
            pa[5] = pa[5] + a1 * b1;
            pa[6] = pa[6] + a1 * b2;
            pa[7] = pa[7] + a1 * b3;
            pa[8] = pa[8] + a2 * b0;
            pa[9] = pa[9] + a2 * b1;
            pa[10] = pa[10] + a2 * b2;
            pa[11] = pa[11] + a2 * b3;
            pa[12] = pa[12] + a3 * b0;
            pa[13] = pa[13] + a3 * b1;
            pa[14] = pa[14] + a3 * b2;
            pa[15] = pa[15] + a3 * b3;
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + i * 16u;
        for (j = 0u; j < 4u; j = j + 1u) {
            let c = col0 + lid.x + j * 16u;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
