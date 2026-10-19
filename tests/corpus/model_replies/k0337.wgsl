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
    let r0 = wid.y * 64u;
    let c0 = wid.x * 64u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (q = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + q * 256u;
            let tt = idx / 64u;
            let r = idx % 64u;
            let it = t + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m) {
                if (r0 + r < cfg.r_rows && cfg.r_off + r0 + r < cfg.d) {
                    va = phi[it * cfg.d + cfg.r_off + r0 + r];
                }
                if (c0 + r < cfg.d) {
                    vb = phi[it * cfg.d + c0 + r];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let a = wg_a[k * 64u + lid.y * 4u + j];
                for (i = 0u; i < 4u; i = i + 1u) {
                    pa[j * 4u + i] = pa[j * 4u + i] + a * wg_b[k * 64u + lid.x * 4u + i];
                }
            }
        }
        workgroupBarrier();
    }
    for (j = 0u; j < 4u; j = j + 1u) {
        let lr = r0 + lid.y * 4u + j;
        for (i = 0u; i < 4u; i = i + 1u) {
            let c = c0 + lid.x * 4u + i;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[j * 4u + i];
            }
        }
    }
}
