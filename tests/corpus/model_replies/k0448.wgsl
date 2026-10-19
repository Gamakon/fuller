// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    for (k = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 8u) {
        let par = ((t / 8u) & 1u) * 32u;
        for (k = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 128u;
            let it = t + e / 64u;
            let ca = cfg.r_off + wid.y * 64u + (e % 64u);
            let cb = wid.x * 64u + (e % 64u);
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m && ca < cfg.d) {
                va = phi[it * cfg.d + ca];
            }
            if (it < cfg.m && cb < cfg.d) {
                vb = phi[it * cfg.d + cb];
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                let av = wg_a[k * 64u + lid.y * 8u + k2];
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    let ix = par + k2 * 4u + k3;
                    pa[ix] = pa[ix] + av * wg_b[k * 64u + lid.x * 4u + k3];
                }
            }
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let lr = gid.y * 8u + k;
            let c = gid.x * 4u + k2;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && c < cfg.d) {
                let j = k * 4u + k2;
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + (pa[j] + pa[j + 32u]);
            }
        }
    }
}
