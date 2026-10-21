// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var pc: array<f32, 64>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) {
        pa[q] = 0.0;
        pb[q] = 0.0;
        pc[q] = 0.0;
    }
    let nch = (cfg.m + 31u) / 32u;
    for (var t0: u32 = 0u; t0 < nch; t0 = t0 + 1u) {
        let base = t0 * 32u;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let e = lidx + 128u * k;
            let it = e >> 6u;
            let cc = e & 63u;
            let item = base + it;
            let gc = wid.x * 64u + cc;
            var va: f32 = 0.0;
            if (item < cfg.m && gc < cfg.d) {
                va = phi[item * cfg.d + gc];
            }
            wg_a[e] = va;
            let lrw = wid.y * 64u + cc;
            let gr = cfg.r_off + lrw;
            var vb: f32 = 0.0;
            if (item < cfg.m && lrw < cfg.r_rows && gr < cfg.d) {
                vb = phi[item * cfg.d + gr];
            }
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (var q: u32 = 0u; q < 32u; q = q + 1u) {
            pb[q] = 0.0;
        }
        for (var q: u32 = 32u; q < 64u; q = q + 1u) {
            pb[q] = 0.0;
        }
        for (var it: u32 = 0u; it < 32u; it = it + 1u) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                pc[j] = wg_a[it * 64u + lid.x * 8u + j];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let bv = wg_b[it * 64u + lid.y * 4u + i];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    pb[i * 8u + j] = pb[i * 8u + j] + bv * pc[j];
                }
            }
        }
        for (var q: u32 = 0u; q < 32u; q = q + 1u) {
            pa[q] = pa[q] + pb[q];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let c = wid.x * 64u + lid.x * 8u + j;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[i * 8u + j];
            }
        }
    }
}
