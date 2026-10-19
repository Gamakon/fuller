// grid 32 8 2 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var tot0 = vec4<f32>(0.0);
    var tot1 = vec4<f32>(0.0);
    var tot2 = vec4<f32>(0.0);
    var tot3 = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let lidx = lid.y * 32u + lid.x;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = idx / 64u;
            let col = idx % 64u;
            let item = t + row;
            let it = min(item, cfg.m - 1u);
            let ca = min(colbase + col, cfg.d - 1u);
            let cb = min(cfg.r_off + rowbase + col, cfg.d - 1u);
            let ma = f32(min(cfg.m - min(item, cfg.m), 1u));
            wg_a[idx] = phi[it * cfg.d + ca] * ma;
            wg_b[idx] = phi[it * cfg.d + cb];
        }
        workgroupBarrier();
        var acc0 = vec4<f32>(0.0);
        var acc1 = vec4<f32>(0.0);
        var acc2 = vec4<f32>(0.0);
        var acc3 = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a0 = wg_a[kk * 64u + lid.x];
            let a1 = wg_a[kk * 64u + lid.x + 32u];
            let av = vec4<f32>(a0, a1, a0, a1);
            let bo = kk * 64u + lid.y * 8u;
            let b0 = wg_b[bo];
            let b1 = wg_b[bo + 1u];
            let b2 = wg_b[bo + 2u];
            let b3 = wg_b[bo + 3u];
            let b4 = wg_b[bo + 4u];
            let b5 = wg_b[bo + 5u];
            let b6 = wg_b[bo + 6u];
            let b7 = wg_b[bo + 7u];
            acc0 = acc0 + vec4<f32>(b0, b0, b1, b1) * av;
            acc1 = acc1 + vec4<f32>(b2, b2, b3, b3) * av;
            acc2 = acc2 + vec4<f32>(b4, b4, b5, b5) * av;
            acc3 = acc3 + vec4<f32>(b6, b6, b7, b7) * av;
        }
        tot0 = tot0 + acc0;
        tot1 = tot1 + acc1;
        tot2 = tot2 + acc2;
        tot3 = tot3 + acc3;
        workgroupBarrier();
    }
    let c0 = colbase + lid.x;
    let c1 = c0 + 32u;
    let rb = rowbase + lid.y * 8u;

    let q0r0 = rb;
    let q0r1 = rb + 1u;
    let q0ok0 = q0r0 < cfg.r_rows && cfg.r_off + q0r0 < cfg.d;
    let q0ok1 = q0r1 < cfg.r_rows && cfg.r_off + q0r1 < cfg.d;
    if (q0ok0 && c0 < cfg.d) { h[q0r0 * cfg.d + c0] = h[q0r0 * cfg.d + c0] + tot0.x; }
    if (q0ok0 && c1 < cfg.d) { h[q0r0 * cfg.d + c1] = h[q0r0 * cfg.d + c1] + tot0.y; }
    if (q0ok1 && c0 < cfg.d) { h[q0r1 * cfg.d + c0] = h[q0r1 * cfg.d + c0] + tot0.z; }
    if (q0ok1 && c1 < cfg.d) { h[q0r1 * cfg.d + c1] = h[q0r1 * cfg.d + c1] + tot0.w; }

    let q1r0 = rb + 2u;
    let q1r1 = rb + 3u;
    let q1ok0 = q1r0 < cfg.r_rows && cfg.r_off + q1r0 < cfg.d;
    let q1ok1 = q1r1 < cfg.r_rows && cfg.r_off + q1r1 < cfg.d;
    if (q1ok0 && c0 < cfg.d) { h[q1r0 * cfg.d + c0] = h[q1r0 * cfg.d + c0] + tot1.x; }
    if (q1ok0 && c1 < cfg.d) { h[q1r0 * cfg.d + c1] = h[q1r0 * cfg.d + c1] + tot1.y; }
    if (q1ok1 && c0 < cfg.d) { h[q1r1 * cfg.d + c0] = h[q1r1 * cfg.d + c0] + tot1.z; }
    if (q1ok1 && c1 < cfg.d) { h[q1r1 * cfg.d + c1] = h[q1r1 * cfg.d + c1] + tot1.w; }

    let q2r0 = rb + 4u;
    let q2r1 = rb + 5u;
    let q2ok0 = q2r0 < cfg.r_rows && cfg.r_off + q2r0 < cfg.d;
    let q2ok1 = q2r1 < cfg.r_rows && cfg.r_off + q2r1 < cfg.d;
    if (q2ok0 && c0 < cfg.d) { h[q2r0 * cfg.d + c0] = h[q2r0 * cfg.d + c0] + tot2.x; }
    if (q2ok0 && c1 < cfg.d) { h[q2r0 * cfg.d + c1] = h[q2r0 * cfg.d + c1] + tot2.y; }
    if (q2ok1 && c0 < cfg.d) { h[q2r1 * cfg.d + c0] = h[q2r1 * cfg.d + c0] + tot2.z; }
    if (q2ok1 && c1 < cfg.d) { h[q2r1 * cfg.d + c1] = h[q2r1 * cfg.d + c1] + tot2.w; }

    let q3r0 = rb + 6u;
    let q3r1 = rb + 7u;
    let q3ok0 = q3r0 < cfg.r_rows && cfg.r_off + q3r0 < cfg.d;
    let q3ok1 = q3r1 < cfg.r_rows && cfg.r_off + q3r1 < cfg.d;
    if (q3ok0 && c0 < cfg.d) { h[q3r0 * cfg.d + c0] = h[q3r0 * cfg.d + c0] + tot3.x; }
    if (q3ok0 && c1 < cfg.d) { h[q3r0 * cfg.d + c1] = h[q3r0 * cfg.d + c1] + tot3.y; }
    if (q3ok1 && c0 < cfg.d) { h[q3r1 * cfg.d + c0] = h[q3r1 * cfg.d + c0] + tot3.z; }
    if (q3ok1 && c1 < cfg.d) { h[q3r1 * cfg.d + c1] = h[q3r1 * cfg.d + c1] + tot3.w; }
}
