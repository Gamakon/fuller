// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let e = lidx + 256u * j;
            let tt = e >> 5u;
            let cc = e & 31u;
            let t = t0 + tt;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < cfg.m) {
                let c = colBase + cc;
                if (c < cfg.d) {
                    vb = phi[t * cfg.d + c];
                }
                let lr = rowBase + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) {
                    va = phi[t * cfg.d + gr];
                }
            }
            sa[e] = va;
            sb[e] = vb;
        }
        workgroupBarrier();
        var p00: f32 = 0.0;
        var p01: f32 = 0.0;
        var p10: f32 = 0.0;
        var p11: f32 = 0.0;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ia = k * 32u + lid.y * 2u;
            let ib = k * 32u + lid.x * 2u;
            let r0 = sa[ia];
            let r1 = sa[ia + 1u];
            let c0 = sb[ib];
            let c1 = sb[ib + 1u];
            p00 = p00 + r0 * c0;
            p01 = p01 + r0 * c1;
            p10 = p10 + r1 * c0;
            p11 = p11 + r1 * c1;
        }
        a00 = a00 + p00;
        a01 = a01 + p01;
        a10 = a10 + p10;
        a11 = a11 + p11;
        workgroupBarrier();
    }
    let lr0 = rowBase + lid.y * 2u;
    let c0 = colBase + lid.x * 2u;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let lr = lr0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let c = c0 + j;
                if (c < cfg.d) {
                    var v: f32 = a00;
                    if (i == 0u && j == 1u) { v = a01; }
                    if (i == 1u && j == 0u) { v = a10; }
                    if (i == 1u && j == 1u) { v = a11; }
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + v;
                }
            }
        }
    }
}
