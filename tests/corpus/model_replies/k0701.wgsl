// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var z: u32 = 0u; z < 32u; z = z + 1u) {
        pa[z] = 0.0;
    }
    let tx = lid.x;
    let ty = lid.y;
    let nt = (cfg.m + 15u) / 16u;
    for (var t0: u32 = 0u; t0 < nt; t0 = t0 + 1u) {
        for (var l: u32 = 0u; l < 8u; l = l + 1u) {
            let idx = lidx + l * 128u;
            let kk = idx / 64u;
            let j = idx % 64u;
            let t = t0 * 16u + kk;
            let g = cfg.r_off + wid.y * 64u + j;
            let c = wid.x * 64u + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (g < cfg.d) {
                    va = phi[t * cfg.d + g];
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
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let a = wg_a[kk * 64u + ty * 8u + i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b = wg_b[kk * 64u + tx * 4u + j];
                    pa[i * 4u + j] = pa[i * 4u + j] + a * b;
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let lr = wid.y * 64u + ty * 8u + i;
            let c = wid.x * 64u + tx * 4u + j;
            if (lr < cfg.r_rows && c < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + pa[i * 4u + j];
            }
        }
    }
}
