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
    var bv: array<f32, 64>;
    var t0: u32 = 0u;
    var q: u32 = 0u;
    var kk: u32 = 0u;
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (i = 0u; i < 16u; i = i + 1u) {
        pa[i] = 0.0;
    }
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (t0 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        for (q = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let ti = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + ti;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < cfg.m) {
                if (colBase + cc < cfg.d) {
                    vb = phi[t * cfg.d + colBase + cc];
                }
                if (rowBase + cc < cfg.r_rows && cfg.r_off + rowBase + cc < cfg.d) {
                    va = phi[t * cfg.d + cfg.r_off + rowBase + cc];
                }
            }
            wg_b[ti * 64u + cc] = vb;
            wg_a[ti * 64u + cc] = va;
        }
        workgroupBarrier();
        for (kk = 0u; kk < 16u; kk = kk + 1u) {
            for (i = 0u; i < 4u; i = i + 1u) {
                bv[i] = wg_b[kk * 64u + lid.x * 4u + i];
            }
            for (j = 0u; j < 4u; j = j + 1u) {
                let av = wg_a[kk * 64u + lid.y * 4u + j];
                for (i = 0u; i < 4u; i = i + 1u) {
                    pa[j * 4u + i] = pa[j * 4u + i] + av * bv[i];
                }
            }
        }
        workgroupBarrier();
    }
    for (j = 0u; j < 4u; j = j + 1u) {
        for (i = 0u; i < 4u; i = i + 1u) {
            let lr = rowBase + lid.y * 4u + j;
            let c = colBase + lid.x * 4u + i;
            if (lr < cfg.r_rows && c < cfg.d && cfg.r_off + lr < cfg.d) {
                h[lr * cfg.d + c] = h[lr * cfg.d + c] + pa[j * 4u + i];
            }
        }
    }
}
