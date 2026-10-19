// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let nt = (cfg.m + 7u) / 8u;
    for (var tt: u32 = 0u; tt < nt; tt = tt + 1u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let item = idx / 64u;
            let cc = idx % 64u;
            let gt = tt * 8u + item;
            let gc = col0 + cc;
            let lrr = row0 + cc;
            let grr = cfg.r_off + lrr;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (gt < cfg.m && gc < cfg.d) {
                va = phi[gt * cfg.d + gc];
            }
            if (gt < cfg.m && lrr < cfg.r_rows && grr < cfg.d) {
                vb = phi[gt * cfg.d + grr];
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                ra[i] = wg_a[kk * 64u + lid.x + 16u * i];
                rb[i] = wg_b[kk * 64u + lid.y + 16u * i];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    pa[j * 4u + i] = pa[j * 4u + i] + rb[j] * ra[i];
                }
            }
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let lr = row0 + lid.y + 16u * j;
            let c = col0 + lid.x + 16u * i;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[j * 4u + i];
            }
        }
    }
}
