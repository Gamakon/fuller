// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    var acc: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let slot = lid.y * 16u + lid.x;
    let lk = lid.y;
    let lj = lid.x * 4u;
    let colb = wid.x * 64u + lj;
    let rowl = wid.y * 64u + lj;
    for (var t0: u32 = 0u; t0 < (m + 15u) / 16u; t0 = t0 + 1u) {
        let t = t0 * 16u + lk;
        let tok = t < m;
        let rg0 = ro + rowl;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        var b2: f32 = 0.0;
        var b3: f32 = 0.0;
        if (tok && colb < d) { a0 = phi[t * d + colb]; }
        if (tok && colb + 1u < d) { a1 = phi[t * d + (colb + 1u)]; }
        if (tok && colb + 2u < d) { a2 = phi[t * d + (colb + 2u)]; }
        if (tok && colb + 3u < d) { a3 = phi[t * d + (colb + 3u)]; }
        if (tok && rowl < rr && rg0 < d) { b0 = phi[t * d + rg0]; }
        if (tok && rowl + 1u < rr && rg0 + 1u < d) { b1 = phi[t * d + (rg0 + 1u)]; }
        if (tok && rowl + 2u < rr && rg0 + 2u < d) { b2 = phi[t * d + (rg0 + 2u)]; }
        if (tok && rowl + 3u < rr && rg0 + 3u < d) { b3 = phi[t * d + (rg0 + 3u)]; }
        wa[slot] = vec4<f32>(a0, a1, a2, a3);
        wb[slot] = vec4<f32>(b0, b1, b2, b3);
        workgroupBarrier();
        var t0v: vec4<f32> = vec4<f32>(0.0);
        var t1v: vec4<f32> = vec4<f32>(0.0);
        var t2v: vec4<f32> = vec4<f32>(0.0);
        var t3v: vec4<f32> = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wa[k * 16u + lid.x];
            let bv = wb[k * 16u + lid.y];
            t0v = t0v + av * bv.x;
            t1v = t1v + av * bv.y;
            t2v = t2v + av * bv.z;
            t3v = t3v + av * bv.w;
        }
        acc[0] = acc[0] + t0v;
        acc[1] = acc[1] + t1v;
        acc[2] = acc[2] + t2v;
        acc[3] = acc[3] + t3v;
        workgroupBarrier();
    }
    let c0 = (wid.x * 16u + lid.x) * 4u;
    let r0 = (wid.y * 16u + lid.y) * 4u;
    let c1 = c0 + 1u;
    let c2 = c0 + 2u;
    let c3 = c0 + 3u;
    let rl0 = r0;
    let rl1 = r0 + 1u;
    let rl2 = r0 + 2u;
    let rl3 = r0 + 3u;
    let ok0 = rl0 < rr && ro + rl0 < d;
    let ok1 = rl1 < rr && ro + rl1 < d;
    let ok2 = rl2 < rr && ro + rl2 < d;
    let ok3 = rl3 < rr && ro + rl3 < d;
    let v0 = acc[0];
    let v1 = acc[1];
    let v2 = acc[2];
    let v3 = acc[3];
    if (ok0 && c0 < d) { h[rl0 * d + c0] = h[rl0 * d + c0] + v0.x; }
    if (ok0 && c1 < d) { h[rl0 * d + c1] = h[rl0 * d + c1] + v0.y; }
    if (ok0 && c2 < d) { h[rl0 * d + c2] = h[rl0 * d + c2] + v0.z; }
    if (ok0 && c3 < d) { h[rl0 * d + c3] = h[rl0 * d + c3] + v0.w; }
    if (ok1 && c0 < d) { h[rl1 * d + c0] = h[rl1 * d + c0] + v1.x; }
    if (ok1 && c1 < d) { h[rl1 * d + c1] = h[rl1 * d + c1] + v1.y; }
    if (ok1 && c2 < d) { h[rl1 * d + c2] = h[rl1 * d + c2] + v1.z; }
    if (ok1 && c3 < d) { h[rl1 * d + c3] = h[rl1 * d + c3] + v1.w; }
    if (ok2 && c0 < d) { h[rl2 * d + c0] = h[rl2 * d + c0] + v2.x; }
    if (ok2 && c1 < d) { h[rl2 * d + c1] = h[rl2 * d + c1] + v2.y; }
    if (ok2 && c2 < d) { h[rl2 * d + c2] = h[rl2 * d + c2] + v2.z; }
    if (ok2 && c3 < d) { h[rl2 * d + c3] = h[rl2 * d + c3] + v2.w; }
    if (ok3 && c0 < d) { h[rl3 * d + c0] = h[rl3 * d + c0] + v3.x; }
    if (ok3 && c1 < d) { h[rl3 * d + c1] = h[rl3 * d + c1] + v3.y; }
    if (ok3 && c2 < d) { h[rl3 * d + c2] = h[rl3 * d + c2] + v3.z; }
    if (ok3 && c3 < d) { h[rl3 * d + c3] = h[rl3 * d + c3] + v3.w; }
}
