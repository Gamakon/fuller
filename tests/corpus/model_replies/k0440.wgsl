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
    var pb: array<f32, 64>;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
        pb[k] = 0.0;
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let nchunk = (cfg.m + 31u) / 32u;
    for (var ch: u32 = 0u; ch < nchunk; ch = ch + 1u) {
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let idx = lidx + k * 128u;
            let it = idx / 64u;
            let cc = idx % 64u;
            let item = ch * 32u + it;
            let c = col0 + cc;
            var va: f32 = 0.0;
            if (item < cfg.m && c < cfg.d) {
                va = phi[item * cfg.d + c];
            }
            wg_a[idx] = va;
            let lr = row0 + cc;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            if (item < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                vb = phi[item * cfg.d + gr];
            }
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var it: u32 = 0u; it < 32u; it = it + 2u) {
            let ba = it * 64u;
            let bb = ba + 64u;
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let r0 = wg_b[ba + lid.y * 8u + i];
                let r1 = wg_b[bb + lid.y * 8u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + r0 * wg_a[ba + lid.x * 4u + j];
                    pb[i * 4u + j] = pb[i * 4u + j] + r1 * wg_a[bb + lid.x * 4u + j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + lid.y * 8u + i;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = col0 + lid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + (pa[i * 4u + j] + pb[i * 4u + j]);
            }
        }
    }
}
