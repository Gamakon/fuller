// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var tt: u32 = 0u; var q: u32 = 0u; var kk: u32 = 0u;
    let row0 = wid.y * 32u;
    let col0 = wid.x * 32u;
    for (tt = 0u; tt < (cfg.m + 15u) / 16u; tt = tt + 1u) {
        for (q = 0u; q < 2u; q = q + 1u) {
            let e = lidx + q * 256u;
            let ki = e / 32u;
            let cc = e % 32u;
            let t = tt * 16u + ki;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < cfg.m) {
                if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < cfg.d) {
                    va = phi[t * cfg.d + cfg.r_off + row0 + cc];
                }
                if (col0 + cc < cfg.d) {
                    vb = phi[t * cfg.d + col0 + cc];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        for (kk = 0u; kk < 16u; kk = kk + 1u) {
            let a0 = wg_a[kk * 32u + lid.y];
            let a1 = wg_a[kk * 32u + lid.y + 16u];
            let b0 = wg_b[kk * 32u + lid.x];
            let b1 = wg_b[kk * 32u + lid.x + 16u];
            f0 = f0 + a0 * b0;
            f1 = f1 + a0 * b1;
            f2 = f2 + a1 * b0;
            f3 = f3 + a1 * b1;
        }
        workgroupBarrier();
    }
    let lr0 = row0 + lid.y;
    let lr1 = row0 + lid.y + 16u;
    let c0 = col0 + lid.x;
    let c1 = col0 + lid.x + 16u;
    if (lr0 < cfg.r_rows && c0 < cfg.d) { h[lr0 * cfg.d + c0] = h[lr0 * cfg.d + c0] + f0; }
    if (lr0 < cfg.r_rows && c1 < cfg.d) { h[lr0 * cfg.d + c1] = h[lr0 * cfg.d + c1] + f1; }
    if (lr1 < cfg.r_rows && c0 < cfg.d) { h[lr1 * cfg.d + c0] = h[lr1 * cfg.d + c0] + f2; }
    if (lr1 < cfg.r_rows && c1 < cfg.d) { h[lr1 * cfg.d + c1] = h[lr1 * cfg.d + c1] + f3; }
}
