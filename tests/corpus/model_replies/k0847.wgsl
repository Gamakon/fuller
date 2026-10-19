// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let lk = tid >> 5u;
    let lj = tid & 31u;
    let col0 = wg.x * 32u;
    let row0 = wg.y * 32u;

    let ldc = col0 + lj;
    let ldlr = row0 + lj;
    let ldgr = cfg.r_off + ldlr;
    let rowOk = ldlr < cfg.r_rows && ldgr < d;
    let colOk = ldc < d;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        let t = t0 + lk;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < m) {
            if (rowOk) { va = phi[t * d + ldgr]; }
            if (colOk) { vb = phi[t * d + ldc]; }
        }
        sa[lk * 32u + lj] = va;
        sb[lk * 32u + lj] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let r0 = sa[k * 32u + ly];
            let r1 = sa[k * 32u + ly + 16u];
            let c0 = sb[k * 32u + lx];
            let c1 = sb[k * 32u + lx + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
    }

    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
        let lr = row0 + ly + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 2u; i = i + 1u) {
                let c = col0 + lx + 16u * i;
                if (c < d) {
                    var v: f32;
                    if (j == 0u) {
                        if (i == 0u) { v = a00; } else { v = a01; }
                    } else {
                        if (i == 0u) { v = a10; } else { v = a11; }
                    }
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v;
                }
            }
        }
    }
}