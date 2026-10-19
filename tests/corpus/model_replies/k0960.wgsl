// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 512>;
var<workgroup> sB: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let lk = l >> 5u;
    let li = l & 31u;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;

    let ldCol = colBase + li;
    let ldLr = rowBase + li;
    let ldGr = cfg.r_off + ldLr;
    let colOk = ldCol < d;
    let rowOk = (ldLr < cfg.r_rows) && (ldGr < d);

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let k = lk + e * 8u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (rowOk) { va = phi[base + ldGr]; }
                if (colOk) { vb = phi[base + ldCol]; }
            }
            sA[(k * 32u + li) & 511u] = va;
            sB[(k * 32u + li) & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let r0 = sA[(o + ty) & 511u];
            let r1 = sA[(o + ty + 16u) & 511u];
            let c0 = sB[(o + tx) & 511u];
            let c1 = sB[(o + tx + 16u) & 511u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr0 = rowBase + ty;
    let lr1 = lr0 + 16u;
    let cc0 = colBase + tx;
    let cc1 = cc0 + 16u;
    let ok0 = (lr0 < cfg.r_rows) && ((cfg.r_off + lr0) < d);
    let ok1 = (lr1 < cfg.r_rows) && ((cfg.r_off + lr1) < d);
    if (ok0) {
        if (cc0 < d) { h[lr0 * d + cc0] = h[lr0 * d + cc0] + a00; }
        if (cc1 < d) { h[lr0 * d + cc1] = h[lr0 * d + cc1] + a01; }
    }
    if (ok1) {
        if (cc0 < d) { h[lr1 * d + cc0] = h[lr1 * d + cc0] + a10; }
        if (cc1 < d) { h[lr1 * d + cc1] = h[lr1 * d + cc1] + a11; }
    }
}
