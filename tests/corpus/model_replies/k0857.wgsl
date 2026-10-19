// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> rt: array<f32, 256>;
var<workgroup> ct: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let lk = tid & 31u;
    let lt = tid >> 5u;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;

    let lcol = colBase + lk;
    let lrow = rowBase + lk;
    let lgr = cfg.r_off + lrow;
    let colOk = lcol < cfg.d;
    let rowOk = (lrow < cfg.r_rows) && (lgr < cfg.d);

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= cfg.m) { break; }
        let item = t0 + lt;
        var cv: f32 = 0.0;
        var rv: f32 = 0.0;
        if (item < cfg.m) {
            let base = item * cfg.d;
            if (colOk) { cv = phi[base + lcol]; }
            if (rowOk) { rv = phi[base + lgr]; }
        }
        ct[lt * 32u + lk] = cv;
        rt[lt * 32u + lk] = rv;
        workgroupBarrier();

        for (var t: u32 = 0u; t < 8u; t = t + 1u) {
            let o = t * 32u;
            let r0 = rt[o + ly * 2u];
            let r1 = rt[o + ly * 2u + 1u];
            let c0 = ct[o + lx * 2u];
            let c1 = ct[o + lx * 2u + 1u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let lr0 = rowBase + ly * 2u;
    let lr1 = lr0 + 1u;
    let cc0 = colBase + lx * 2u;
    let cc1 = cc0 + 1u;

    if (lr0 < cfg.r_rows) {
        if (cc0 < cfg.d) {
            let i = lr0 * cfg.d + cc0;
            h[i] = h[i] + a00;
        }
        if (cc1 < cfg.d) {
            let i = lr0 * cfg.d + cc1;
            h[i] = h[i] + a01;
        }
    }
    if (lr1 < cfg.r_rows) {
        if (cc0 < cfg.d) {
            let i = lr1 * cfg.d + cc0;
            h[i] = h[i] + a10;
        }
        if (cc1 < cfg.d) {
            let i = lr1 * cfg.d + cc1;
            h[i] = h[i] + a11;
        }
    }
}