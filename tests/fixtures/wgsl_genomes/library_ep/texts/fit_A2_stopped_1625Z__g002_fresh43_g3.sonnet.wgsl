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
        var av4 = vec4<f32>(0.0);
        var bv4 = vec4<f32>(0.0);
        if (t < m) {
            if (colb < d) { av4.x = phi[base + colb]; }
            if (colb + 1u < d) { av4.y = phi[base + colb + 1u]; }
            if (colb + 2u < d) { av4.z = phi[base + colb + 2u]; }
            if (colb + 3u < d) { av4.w = phi[base + colb + 3u]; }
            if (rowl < rr) {
                if (ro + rowl < d) { bv4.x = phi[base + ro + rowl]; }
            }
            if (rowl + 1u < rr) {
                if (ro + rowl + 1u < d) { bv4.y = phi[base + ro + rowl + 1u]; }
            }
            if (rowl + 2u < rr) {
                if (ro + rowl + 2u < d) { bv4.z = phi[base + ro + rowl + 2u]; }
            }
            if (rowl + 3u < rr) {
                if (ro + rowl + 3u < d) { bv4.w = phi[base + ro + rowl + 3u]; }
            }
        }
        wa[lidx] = av4;
        wb[lidx] = bv4;
        workgroupBarrier();
        var t0v = vec4<f32>(0.0);
        var t1v = vec4<f32>(0.0);
        var t2v = vec4<f32>(0.0);
        var t3v = vec4<f32>(0.0);
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
        if (lr < rr) {
            if (ro + lr < d) {
                let rb = lr * d + c0;
                if (c0 < d) { h[rb] = h[rb] + v.x; }
                if (c0 + 1u < d) { h[rb + 1u] = h[rb + 1u] + v.y; }
                if (c0 + 2u < d) { h[rb + 2u] = h[rb + 2u] + v.z; }
                if (c0 + 3u < d) { h[rb + 3u] = h[rb + 3u] + v.w; }
            }
        }
    }
}
