// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc0: array<vec4<f32>, 16>;
    var acc1: array<vec4<f32>, 16>;
    for (var ii: u32 = 0u; ii < 8u; ii = ii + 1u) {
        acc0[ii] = vec4<f32>(0.0);
        acc1[ii] = vec4<f32>(0.0);
    }
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 32u) {
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let idx = i * 128u + lidx;
            let kk = idx / 64u;
            let j = idx % 64u;
            let item = t0 + kk;
            let c = col0 + j;
            var vb: f32 = 0.0;
            if (item < cfg.m && c < cfg.d) {
                vb = phi[item * cfg.d + c];
            }
            wg_b[idx] = vb;
            let lr = row0 + j;
            let gr = cfg.r_off + lr;
            var va: f32 = 0.0;
            if (item < cfg.m && lr < cfg.r_rows && gr < cfg.d) {
                va = phi[item * cfg.d + gr];
            }
            wg_a[idx] = va;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ke = (2u * k) * 64u;
            let ko = ke + 64u;
            let be = vec4<f32>(wg_b[ke + lid.x], wg_b[ke + lid.x + 16u], wg_b[ke + lid.x + 32u], wg_b[ke + lid.x + 48u]);
            let bo = vec4<f32>(wg_b[ko + lid.x], wg_b[ko + lid.x + 16u], wg_b[ko + lid.x + 32u], wg_b[ko + lid.x + 48u]);
            for (var ii: u32 = 0u; ii < 8u; ii = ii + 1u) {
                let ae = wg_a[ke + lid.y + 8u * ii];
                let ao = wg_a[ko + lid.y + 8u * ii];
                acc0[ii] = acc0[ii] + ae * be;
                acc1[ii] = acc1[ii] + ao * bo;
            }
        }
        workgroupBarrier();
    }
    for (var ii: u32 = 0u; ii < 8u; ii = ii + 1u) {
        let lr = row0 + lid.y + 8u * ii;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < cfg.d) {
            let s = acc0[ii] + acc1[ii];
            for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
                let c = col0 + lid.x + 16u * jj;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + s[jj];
                }
            }
        }
    }
}
