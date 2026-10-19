// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (i = 0u; i < 16u; i = i + 1u) { pa[i] = 0.0; }
    let ntiles = (cfg.m + 15u) / 16u;
    for (t = 0u; t < ntiles; t = t + 1u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 16u + idx / 64u;
            let cc = idx % 64u;
            let gc = wid.x * 64u + cc;
            let lr = wid.y * 64u + cc;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (item < cfg.m && gc < cfg.d) {
                fa = phi[item * cfg.d + gc];
            }
            if (item < cfg.m && lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                fb = phi[item * cfg.d + cfg.r_off + lr];
            }
            wg_a[idx] = fa;
            wg_b[idx] = fb;
        }
        workgroupBarrier();
        for (i = 0u; i < 16u; i = i + 1u) { pb[i] = 0.0; }
        for (k = 0u; k < 16u; k = k + 1u) {
            for (i = 0u; i < 4u; i = i + 1u) {
                let bv = wg_b[k * 64u + lid.y + 16u * i];
                for (j = 0u; j < 4u; j = j + 1u) {
                    let av = wg_a[k * 64u + lid.x + 16u * j];
                    pb[i * 4u + j] = pb[i * 4u + j] + bv * av;
                }
            }
        }
        for (i = 0u; i < 16u; i = i + 1u) { pa[i] = pa[i] + pb[i]; }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        let r = wid.y * 64u + lid.y + 16u * i;
        if (r < cfg.r_rows && (cfg.r_off + r) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 16u * j;
                if (c < cfg.d) {
                    let o = r * cfg.d + c;
                    h[o] = h[o] + pa[i * 4u + j];
                }
            }
        }
    }
}
