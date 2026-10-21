// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 32>;
    for (var z: u32 = 0u; z < 32u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let ntiles = (cfg.m + 15u) / 16u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let idx = lidx + q * 128u;
            let kk = idx / 64u;
            let r = idx % 64u;
            let t = tt * 16u + kk;
            let lrow = wid.y * 64u + r;
            let grow = cfg.r_off + lrow;
            let col = wid.x * 64u + r;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m && lrow < cfg.r_rows && grow < cfg.d) {
                va = phi[t * cfg.d + grow];
            }
            if (t < cfg.m && col < cfg.d) {
                vb = phi[t * cfg.d + col];
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let a = wg_a[kk * 64u + lid.y + 8u * i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = wg_b[kk * 64u + lid.x + 16u * j];
                    pa[i * 4u + j] = pa[i * 4u + j] + a * b;
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 8u * i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = wid.x * 64u + lid.x + 16u * j;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[i * 4u + j];
            }
        }
    }
}
