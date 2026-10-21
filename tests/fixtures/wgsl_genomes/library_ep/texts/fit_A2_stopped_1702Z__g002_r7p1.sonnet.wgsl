// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    var av: array<f32, 4>;
    var bv: array<f32, 4>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        acc[z] = 0.0;
    }
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tile: u32 = 0u; tile < ntiles; tile = tile + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let tt = idx / 64u;
            let cc = idx % 64u;
            let t = tile * 16u + tt;
            let lrr = rowBase + cc;
            let grr = cfg.r_off + lrr;
            let cl = colBase + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (lrr < cfg.r_rows && grr < cfg.d) {
                    va = phi[t * cfg.d + grr];
                }
                if (cl < cfg.d) {
                    vb = phi[t * cfg.d + cl];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                av[i] = wg_a[k * 64u + lid.y * 4u + i];
                bv[i] = wg_b[k * 64u + lid.x * 4u + i];
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
            let lr = rowBase + lid.y * 4u + i;
            let c = colBase + lid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + acc[i * 4u + j];
            }
        }
    }
}
