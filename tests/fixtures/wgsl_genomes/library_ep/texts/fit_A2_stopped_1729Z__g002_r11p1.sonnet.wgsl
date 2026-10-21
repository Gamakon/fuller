// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    var av: array<f32, 64>;
    var bv: array<f32, 64>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        acc[k] = 0.0;
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tb: u32 = 0u; tb < ntiles; tb = tb + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + q * 256u;
            let tt = e / 64u;
            let cc = e % 64u;
            let t = tb * 16u + tt;
            let r = row0 + cc;
            let gr = cfg.r_off + r;
            let c = col0 + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (r < cfg.r_rows && gr < cfg.d) {
                    va = phi[t * cfg.d + gr];
                }
                if (c < cfg.d) {
                    vb = phi[t * cfg.d + c];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                av[i] = wg_a[kk * 64u + lid.y * 4u + i];
                bv[i] = wg_b[kk * 64u + lid.x * 4u + i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + av[i] * bv[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = row0 + lid.y * 4u + i;
            let c = col0 + lid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[i * 4u + j];
            }
        }
    }
}
