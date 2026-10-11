// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<f32, 1024>;
var<workgroup> wb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let row0 = wid.y * 32u;
    let col0 = wid.x * 32u;
    var a00: f32 = 0.0; var a01: f32 = 0.0; var a10: f32 = 0.0; var a11: f32 = 0.0;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 256u * k;
            let tt = e >> 5u;
            let cc = e & 31u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if (t + tt < m) {
                let base = (t + tt) * d;
                let lr = row0 + cc;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                    fa = phi[base + cfg.r_off + lr];
                }
                let c = col0 + cc;
                if (c < d) {
                    fb = phi[base + c];
                }
            }
            wa[e] = fa;
            wb[e] = fb;
        }
        workgroupBarrier();
        var p00: f32 = 0.0; var p01: f32 = 0.0; var p10: f32 = 0.0; var p11: f32 = 0.0;
        var q00: f32 = 0.0; var q01: f32 = 0.0; var q10: f32 = 0.0; var q11: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            let o = k * 32u;
            let r0 = wa[o + lid.y];
            let r1 = wa[o + lid.y + 16u];
            let c0 = wb[o + lid.x];
            let c1 = wb[o + lid.x + 16u];
            p00 = p00 + r0 * c0; p01 = p01 + r0 * c1;
            p10 = p10 + r1 * c0; p11 = p11 + r1 * c1;
            let o2 = o + 32u;
            let s0 = wa[o2 + lid.y];
            let s1 = wa[o2 + lid.y + 16u];
            let d0 = wb[o2 + lid.x];
            let d1 = wb[o2 + lid.x + 16u];
            q00 = q00 + s0 * d0; q01 = q01 + s0 * d1;
            q10 = q10 + s1 * d0; q11 = q11 + s1 * d1;
        }
        a00 = a00 + (p00 + q00); a01 = a01 + (p01 + q01);
        a10 = a10 + (p10 + q10); a11 = a11 + (p11 + q11);
        workgroupBarrier();
    }
    let ra = row0 + lid.y;
    let rb = ra + 16u;
    let ca = col0 + lid.x;
    let cb = ca + 16u;
    let okra = ra < cfg.r_rows && cfg.r_off + ra < d;
    let okrb = rb < cfg.r_rows && cfg.r_off + rb < d;
    let okca = ca < d;
    let okcb = cb < d;
    if (okra && okca) { let i = ra * d + ca; h[i] = h[i] + a00; }
    if (okra && okcb) { let i = ra * d + cb; h[i] = h[i] + a01; }
    if (okrb && okca) { let i = rb * d + ca; h[i] = h[i] + a10; }
    if (okrb && okcb) { let i = rb * d + cb; h[i] = h[i] + a11; }
}
