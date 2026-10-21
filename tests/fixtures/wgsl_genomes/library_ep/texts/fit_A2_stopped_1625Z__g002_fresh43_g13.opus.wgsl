// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    var s0: vec4<f32> = vec4<f32>(0.0);
    var s1: vec4<f32> = vec4<f32>(0.0);
    var s2: vec4<f32> = vec4<f32>(0.0);
    var s3: vec4<f32> = vec4<f32>(0.0);
    let slot = lid.y * 16u + lid.x;
    let lk = lid.y;
    let lj = lid.x * 4u;
    let colb = wid.x * 64u + lj;
    let rowl = wid.y * 64u + lj;
    let rg0 = ro + rowl;
    let ca0 = colb < d;
    let ca1 = colb + 1u < d;
    let ca2 = colb + 2u < d;
    let ca3 = colb + 3u < d;
    let cb0 = rowl < rr && rg0 < d;
    let cb1 = rowl + 1u < rr && rg0 + 1u < d;
    let cb2 = rowl + 2u < rr && rg0 + 2u < d;
    let cb3 = rowl + 3u < rr && rg0 + 3u < d;
    for (var t0: u32 = 0u; t0 < (m + 15u) / 16u; t0 = t0 + 1u) {
        let t = t0 * 16u + lk;
        let tb = t * d;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        var b2: f32 = 0.0;
        var b3: f32 = 0.0;
        if (t < m) {
            if (ca0) { a0 = phi[tb + colb]; }
            if (ca1) { a1 = phi[tb + colb + 1u]; }
            if (ca2) { a2 = phi[tb + colb + 2u]; }
            if (ca3) { a3 = phi[tb + colb + 3u]; }
            if (cb0) { b0 = phi[tb + rg0]; }
            if (cb1) { b1 = phi[tb + rg0 + 1u]; }
            if (cb2) { b2 = phi[tb + rg0 + 2u]; }
            if (cb3) { b3 = phi[tb + rg0 + 3u]; }
        }
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
        s0 = s0 + t0v;
        s1 = s1 + t1v;
        s2 = s2 + t2v;
        s3 = s3 + t3v;
        workgroupBarrier();
    }
    let acc: array<vec4<f32>, 4> = array<vec4<f32>, 4>(s0, s1, s2, s3);
    let c0 = gid.x * 4u;
    let r0 = gid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let v = acc[i];
        if (r0 + i < rr && ro + r0 + i < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                if (c0 + j < d) {
                    h[(r0 + i) * d + (c0 + j)] = h[(r0 + i) * d + (c0 + j)] + v[j];
                }
            }
        }
    }
}
