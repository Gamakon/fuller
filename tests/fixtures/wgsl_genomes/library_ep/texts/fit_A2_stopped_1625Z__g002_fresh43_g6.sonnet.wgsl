// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    var acc: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let lk = lidx >> 4u;
    let lj = (lidx & 15u) * 4u;
    let colb = wid.x * 64u + lj;
    let rowl = wid.y * 64u + lj;
    for (var t0: u32 = 0u; t0 < (m + 15u) / 16u; t0 = t0 + 1u) {
        let t = t0 * 16u + lk;
        let base = t * d;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        var b2: f32 = 0.0;
        var b3: f32 = 0.0;
        let tok = t < m;
        let ca1 = colb + 1u;
        let ca2 = colb + 2u;
        let ca3 = colb + 3u;
        let rg0 = ro + rowl;
        let rg1 = rg0 + 1u;
        let rg2 = rg0 + 2u;
        let rg3 = rg0 + 3u;
        let ok0 = tok && colb < d;
        let ok1 = tok && ca1 < d;
        let ok2 = tok && ca2 < d;
        let ok3 = tok && ca3 < d;
        let rk0 = tok && rowl < rr && rg0 < d;
        let rk1 = tok && rowl + 1u < rr && rg1 < d;
        let rk2 = tok && rowl + 2u < rr && rg2 < d;
        let rk3 = tok && rowl + 3u < rr && rg3 < d;
        if (ok0) { a0 = phi[base + colb]; }
        if (ok1) { a1 = phi[base + ca1]; }
        if (ok2) { a2 = phi[base + ca2]; }
        if (ok3) { a3 = phi[base + ca3]; }
        if (rk0) { b0 = phi[base + rg0]; }
        if (rk1) { b1 = phi[base + rg1]; }
        if (rk2) { b2 = phi[base + rg2]; }
        if (rk3) { b3 = phi[base + rg3]; }
        wa[lidx] = vec4<f32>(a0, a1, a2, a3);
        wb[lidx] = vec4<f32>(b0, b1, b2, b3);
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
        let lr = r0 + i;
        let v = acc[i];
        let rb = lr * d + c0;
        let rowok = lr < rr && ro + lr < d;
        if (rowok && c0 < d) { h[rb] = h[rb] + v.x; }
        if (rowok && c0 + 1u < d) { h[rb + 1u] = h[rb + 1u] + v.y; }
        if (rowok && c0 + 2u < d) { h[rb + 2u] = h[rb + 2u] + v.z; }
        if (rowok && c0 + 3u < d) { h[rb + 3u] = h[rb + 3u] + v.w; }
    }
}
