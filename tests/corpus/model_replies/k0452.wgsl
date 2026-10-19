// grid 16 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 32>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let e = lidx;
    let ca = cfg.r_off + wid.y * 32u + (e % 32u);
    let cb = wid.x * 64u + (e % 64u);
    let eo = e / 64u;
    let eo2 = e / 32u;
    let ya = lid.y * 4u;
    let xb = lid.x * 4u;
    for (t = 0u; t < cfg.m; t = t + 2u) {
        let it = t + eo;
        let it2 = t + eo2;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (e < 64u && it2 < cfg.m && ca < cfg.d) {
            va = phi[it2 * cfg.d + ca];
        }
        if (it < cfg.m && cb < cfg.d) {
            vb = phi[it * cfg.d + cb];
        }
        if (e < 64u) {
            wg_a[e] = va;
        }
        wg_b[e] = vb;
        workgroupBarrier();
        for (k = 0u; k < 2u; k = k + 1u) {
            let pb = k * 16u;
            let ab = k * 32u + ya;
            let bb = k * 64u + xb;
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let av = wg_a[ab + k2];
                let pk = pb + k2 * 4u;
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    pa[pk + k3] = pa[pk + k3] + av * wg_b[bb + k3];
                }
            }
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let lr = gid.y * 4u + k;
            let c = gid.x * 4u + k2;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && c < cfg.d) {
                let j = k * 4u + k2;
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + (pa[j] + pa[j + 16u]);
            }
        }
    }
}
