// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let tt = idx >> 5u;
            let cc = idx & 31u;
            let t = t0 + tt;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let col = wid.x * 32u + cc;
                if (col < d) {
                    vb = phi[t * d + col];
                }
                let lrr = wid.y * 32u + cc;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sb[idx] = vb;
            sa[idx] = va;
        }
        workgroupBarrier();
        var p00: f32 = 0.0;
        var p01: f32 = 0.0;
        var p10: f32 = 0.0;
        var p11: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a0 = sa[k * 32u + lid.y];
            let a1 = sa[k * 32u + lid.y + 16u];
            let b0 = sb[k * 32u + lid.x];
            let b1 = sb[k * 32u + lid.x + 16u];
            p00 = p00 + a0 * b0;
            p01 = p01 + a0 * b1;
            p10 = p10 + a1 * b0;
            p11 = p11 + a1 * b1;
        }
        a00 = a00 + p00;
        a01 = a01 + p01;
        a10 = a10 + p10;
        a11 = a11 + p11;
        workgroupBarrier();
    }
    let r0 = wid.y * 32u + lid.y;
    let r1 = r0 + 16u;
    let c0 = wid.x * 32u + lid.x;
    let c1 = c0 + 16u;
    let ok0 = r0 < cfg.r_rows && (cfg.r_off + r0) < d;
    let ok1 = r1 < cfg.r_rows && (cfg.r_off + r1) < d;
    if (ok0 && c0 < d) { let i = r0 * d + c0; h[i] = h[i] + a00; }
    if (ok0 && c1 < d) { let i = r0 * d + c1; h[i] = h[i] + a01; }
    if (ok1 && c0 < d) { let i = r1 * d + c0; h[i] = h[i] + a10; }
    if (ok1 && c1 < d) { let i = r1 * d + c1; h[i] = h[i] + a11; }
}
