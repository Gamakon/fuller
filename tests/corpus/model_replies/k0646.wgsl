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
        if (tok && colb + 1u < d) { a1 = phi[t * d + colb + 1u]; }
        if (tok && colb + 2u < d) { a2 = phi[t * d + colb + 2u]; }
        if (tok && colb + 3u < d) { a3 = phi[t * d + colb + 3u]; }
        if (tok && rowl < rr && rg0 < d) { b0 = phi[t * d + rg0]; }
        if (tok && rowl + 1u < rr && rg0 + 1u < d) { b1 = phi[t * d + rg0 + 1u]; }
        if (tok && rowl + 2u < rr && rg0 + 2u < d) { b2 = phi[t * d + rg0 + 2u]; }
        if (tok && rowl + 3u < rr && rg0 + 3u < d) { b3 = phi[t * d + rg0 + 3u]; }
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
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let v = acc[i];
        let rowok = r0 + i < rr && ro + r0 + i < d;
        if (rowok && c0 < d) { h[(r0 + i) * d + c0] = h[(r0 + i) * d + c0] + v.x; }
        if (rowok && c0 + 1u < d) { h[(r0 + i) * d + c0 + 1u] = h[(r0 + i) * d + c0 + 1u] + v.y; }
        if (rowok && c0 + 2u < d) { h[(r0 + i) * d + c0 + 2u] = h[(r0 + i) * d + c0 + 2u] + v.z; }
        if (rowok && c0 + 3u < d) { h[(r0 + i) * d + c0 + 3u] = h[(r0 + i) * d + c0 + 3u] + v.w; }
    }
}
