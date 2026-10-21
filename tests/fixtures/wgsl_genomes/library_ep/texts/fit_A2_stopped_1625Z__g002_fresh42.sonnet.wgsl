// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let lx = lid.x;
    let ly = lid.y;
    let c0 = wid.x * 32u;
    let r0 = wid.y * 32u;
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    let d = cfg.d;
    let m = cfg.m;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let tt = idx >> 5u;
            let k = idx & 31u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let col = c0 + k;
                if (col < d) {
                    va = phi[t * d + col];
                }
                let lr = r0 + k;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[t * d + gr];
                }
            }
            sa[idx] = va;
            sb[idx] = vb;
        }
        workgroupBarrier();
        var p00: f32 = 0.0;
        var p01: f32 = 0.0;
        var p10: f32 = 0.0;
        var p11: f32 = 0.0;
        for (var tt: u32 = 0u; tt < 32u; tt = tt + 1u) {
            let o = tt * 32u;
            let a0 = sa[o + lx];
            let a1 = sa[o + lx + 16u];
            let b0 = sb[o + ly];
            let b1 = sb[o + ly + 16u];
            p00 = p00 + b0 * a0;
            p01 = p01 + b0 * a1;
            p10 = p10 + b1 * a0;
            p11 = p11 + b1 * a1;
        }
        acc00 = acc00 + p00;
        acc01 = acc01 + p01;
        acc10 = acc10 + p10;
        acc11 = acc11 + p11;
        workgroupBarrier();
    }
    let ca = c0 + lx;
    let cb = c0 + lx + 16u;
    let ra = r0 + ly;
    let rb = r0 + ly + 16u;
    if (ra < cfg.r_rows) {
        if (ca < d) {
            h[ra * d + ca] = h[ra * d + ca] + acc00;
        }
        if (cb < d) {
            h[ra * d + cb] = h[ra * d + cb] + acc01;
        }
    }
    if (rb < cfg.r_rows) {
        if (ca < d) {
            h[rb * d + ca] = h[rb * d + ca] + acc10;
        }
        if (cb < d) {
            h[rb * d + cb] = h[rb * d + cb] + acc11;
        }
    }
}
