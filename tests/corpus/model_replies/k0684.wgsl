// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let tx = lid.x;
    let ty = lid.y;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let ntiles = (cfg.m + 31u) / 32u;
    for (var tt: u32 = 0u; tt < ntiles; tt = tt + 1u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let idx = lidx + k * 128u;
            let ti = idx / 64u;
            let j = idx % 64u;
            let t = tt * 32u + ti;
            let lrow = row0 + j;
            let gr = cfg.r_off + lrow;
            let c = col0 + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m && lrow < cfg.r_rows && gr < cfg.d) {
                va = phi[t * cfg.d + gr];
            }
            if (t < cfg.m && c < cfg.d) {
                vb = phi[t * cfg.d + c];
            }
            wg_a[ti * 64u + j] = va;
            wg_b[ti * 64u + j] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let a = wg_a[kk * 64u + ty * 8u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + a * wg_b[kk * 64u + tx * 4u + j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + ty * 8u + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + tx * 4u + j;
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 4u + j];
            }
        }
    }
}
