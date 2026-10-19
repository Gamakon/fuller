// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var ar: array<f32, 64>;
    var br: array<f32, 64>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        pa[q] = 0.0;
    }
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        for (var e: u32 = 0u; e < 4u; e = e + 1u) {
            let idx = lidx + e * 256u;
            let kk = idx / 64u;
            let i = idx % 64u;
            let t = t0 + kk;
            let lr = wid.y * 64u + i;
            let c = wid.x * 64u + i;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    va = phi[t * cfg.d + cfg.r_off + lr];
                }
                if (c < cfg.d) {
                    vb = phi[t * cfg.d + c];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            for (var a: u32 = 0u; a < 4u; a = a + 1u) {
                ar[a] = wg_a[kk * 64u + lid.y * 4u + a];
                br[a] = wg_b[kk * 64u + lid.x * 4u + a];
            }
            for (var a: u32 = 0u; a < 4u; a = a + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[a * 4u + j] = pa[a * 4u + j] + ar[a] * br[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var a: u32 = 0u; a < 4u; a = a + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = gid.y * 4u + a;
            let c = gid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[a * 4u + j];
            }
        }
    }
}
