// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        pa[q] = 0.0;
    }
    let r0 = wid.y * 64u;
    let c0 = wid.x * 64u;
    let nt = (cfg.m + 7u) / 8u;
    for (var tt: u32 = 0u; tt < nt; tt = tt + 1u) {
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let idx = lidx + s * 256u;
            let kk = idx / 64u;
            let cc = idx % 64u;
            let t = tt * 8u + kk;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < cfg.m) {
                if (r0 + cc < cfg.r_rows && cfg.r_off + r0 + cc < cfg.d) {
                    av = phi[t * cfg.d + cfg.r_off + r0 + cc];
                }
                if (c0 + cc < cfg.d) {
                    bv = phi[t * cfg.d + c0 + cc];
                }
            }
            wg_a[idx] = av;
            wg_b[idx] = bv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[k * 64u + lid.y * 4u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + a * wg_b[k * 64u + lid.x * 4u + j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = r0 + lid.y * 4u + i;
            let c = c0 + lid.x * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[i * 4u + j];
            }
        }
    }
}
