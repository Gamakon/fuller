// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sr: array<f32, 256>;
var<workgroup> sc: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let idx = ly * 16u + lx;
    let tt_l = idx / 32u;
    let k_l = idx % 32u;
    let row_base = wg.y * 32u;
    let col_base = wg.x * 32u;
    let d = cfg.d;
    let m = cfg.m;

    let lrl = row_base + k_l;
    let grl = cfg.r_off + lrl;
    let row_ok = lrl < cfg.r_rows && grl < d;
    let cl = col_base + k_l;
    let col_ok = cl < d;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        let t = t0 + tt_l;
        var rv: f32 = 0.0;
        var cv: f32 = 0.0;
        if (t < m) {
            if (row_ok) { rv = phi[t * d + grl]; }
            if (col_ok) { cv = phi[t * d + cl]; }
        }
        sr[idx] = rv;
        sc[idx] = cv;
        workgroupBarrier();
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let o = j * 32u;
            let r0 = sr[o + ly];
            let r1 = sr[o + ly + 16u];
            let c0 = sc[o + lx];
            let c1 = sc[o + lx + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
    }

    let lr0 = row_base + ly;
    let lr1 = lr0 + 16u;
    let cc0 = col_base + lx;
    let cc1 = cc0 + 16u;
    if (lr0 < cfg.r_rows) {
        if (cc0 < d) { let i = lr0 * d + cc0; h[i] = h[i] + a00; }
        if (cc1 < d) { let i = lr0 * d + cc1; h[i] = h[i] + a01; }
    }
    if (lr1 < cfg.r_rows) {
        if (cc0 < d) { let i = lr1 * d + cc0; h[i] = h[i] + a10; }
        if (cc1 < d) { let i = lr1 * d + cc1; h[i] = h[i] + a11; }
    }
}