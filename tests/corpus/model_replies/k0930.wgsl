// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let rbase = wid.y * 32u;
    let cbase = wid.x * 32u;
    let d = cfg.d;
    let m = cfg.m;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lid + k * 256u;
            let tt = idx >> 5u;
            let j = idx & 31u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = rbase + j;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let col = cbase + j;
                if (col < d) {
                    vb = phi[t * d + col];
                }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 32u;
            let x0 = sa[(o + ty) & 511u];
            let x1 = sa[(o + ty + 16u) & 511u];
            let y0 = sb[(o + tx) & 511u];
            let y1 = sb[(o + tx + 16u) & 511u];
            a00 = a00 + x0 * y0;
            a01 = a01 + x0 * y1;
            a10 = a10 + x1 * y0;
            a11 = a11 + x1 * y1;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let r0 = rbase + ty;
    let r1 = r0 + 16u;
    let c0 = cbase + tx;
    let c1 = c0 + 16u;
    if (r0 < cfg.r_rows) {
        if (c0 < d) { h[r0 * d + c0] = h[r0 * d + c0] + a00; }
        if (c1 < d) { h[r0 * d + c1] = h[r0 * d + c1] + a01; }
    }
    if (r1 < cfg.r_rows) {
        if (c0 < d) { h[r1 * d + c0] = h[r1 * d + c0] + a10; }
        if (c1 < d) { h[r1 * d + c1] = h[r1 * d + c1] + a11; }
    }
}
