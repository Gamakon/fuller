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
    for (var z: u32 = 0u; z < 16u; z = z + 1u) {
        acc[z] = 0.0;
    }
    var ra: array<f32, 4>;
    var rb: array<f32, 4>;
    let tx = lid.x;
    let ty = lid.y;
    let nt = (cfg.m + 15u) / 16u;
    for (var t0: u32 = 0u; t0 < nt; t0 = t0 + 1u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let tt = idx / 64u;
            let cc = idx % 64u;
            let item = t0 * 16u + tt;
            let col = wid.x * 64u + cc;
            let gr = cfg.r_off + wid.y * 64u + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < cfg.m) {
                if (col < cfg.d) {
                    vb = phi[item * cfg.d + col];
                }
                if (gr < cfg.d) {
                    va = phi[item * cfg.d + gr];
                }
            }
            wg_a[idx] = va;
            wg_b[idx] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                ra[i] = wg_a[kk * 64u + ty * 4u + i];
                rb[i] = wg_b[kk * 64u + tx * 4u + i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + ra[i] * rb[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = wid.y * 64u + ty * 4u + i;
            let c = wid.x * 64u + tx * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + acc[i * 4u + j];
            }
        }
    }
}
