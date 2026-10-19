// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ta: array<f32, 256>; // 8 items x 32 rows
var<workgroup> tb: array<f32, 256>; // 8 items x 32 cols

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let li = ty * 16u + tx;
    let ld_t = li / 32u;
    let ld_k = li % 32u;
    let row_base = wid.y * 32u;
    let col_base = wid.x * 32u;
    let d = cfg.d;
    let m = cfg.m;

    let lr_ld = row_base + ld_k;
    let gr_ld = cfg.r_off + lr_ld;
    let row_ok = lr_ld < cfg.r_rows && gr_ld < d;
    let c_ld = col_base + ld_k;
    let col_ok = c_ld < d;

    var a00 = 0.0; var a01 = 0.0; var a10 = 0.0; var a11 = 0.0;
    var b00 = 0.0; var b01 = 0.0; var b10 = 0.0; var b11 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ld_t;
        var va = 0.0;
        var vb = 0.0;
        if (t < m) {
            if (row_ok) { va = phi[t * d + gr_ld]; }
            if (col_ok) { vb = phi[t * d + c_ld]; }
        }
        ta[li] = va;
        tb[li] = vb;
        workgroupBarrier();
        for (var s: u32 = 0u; s < 8u; s = s + 2u) {
            let o0 = s * 32u;
            let o1 = o0 + 32u;
            let r0 = ta[o0 + ty];
            let r1 = ta[o0 + ty + 16u];
            let c0 = tb[o0 + tx];
            let c1 = tb[o0 + tx + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
            let q0 = ta[o1 + ty];
            let q1 = ta[o1 + ty + 16u];
            let e0 = tb[o1 + tx];
            let e1 = tb[o1 + tx + 16u];
            b00 = b00 + q0 * e0;
            b01 = b01 + q0 * e1;
            b10 = b10 + q1 * e0;
            b11 = b11 + q1 * e1;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let s00 = a00 + b00;
    let s01 = a01 + b01;
    let s10 = a10 + b10;
    let s11 = a11 + b11;

    let lrA = row_base + ty;
    let lrB = lrA + 16u;
    let cA = col_base + tx;
    let cB = cA + 16u;
    let okRA = lrA < cfg.r_rows && (cfg.r_off + lrA) < d;
    let okRB = lrB < cfg.r_rows && (cfg.r_off + lrB) < d;
    let okCA = cA < d;
    let okCB = cB < d;

    if (okRA && okCA) { let i = lrA * d + cA; h[i] = h[i] + s00; }
    if (okRA && okCB) { let i = lrA * d + cB; h[i] = h[i] + s01; }
    if (okRB && okCA) { let i = lrB * d + cA; h[i] = h[i] + s10; }
    if (okRB && okCB) { let i = lrB * d + cB; h[i] = h[i] + s11; }
}
