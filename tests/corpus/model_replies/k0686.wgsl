// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let ntiles = (cfg.m + 7u) / 8u;
    for (var tb: u32 = 0u; tb < ntiles; tb = tb + 1u) {
        let t0 = tb * 8u;
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let e = lidx + 256u * q;
            let kk = e / 64u;
            let p = e % 64u;
            let t = t0 + kk;
            let lrow = wid.y * 64u + p;
            let grow = cfg.r_off + lrow;
            let col = wid.x * 64u + p;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (lrow < cfg.r_rows && grow < cfg.d) {
                    va = phi[t * cfg.d + grow];
                }
                if (col < cfg.d) {
                    vb = phi[t * cfg.d + col];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[kk * 64u + lid.y * 4u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + a * wg_b[kk * 64u + lid.x * 4u + j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = (wid.y * 16u + lid.y) * 4u + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = (wid.x * 16u + lid.x) * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
