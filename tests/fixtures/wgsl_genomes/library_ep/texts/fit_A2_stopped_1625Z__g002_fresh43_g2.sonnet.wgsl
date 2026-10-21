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
    var tl: array<vec4<f32>, 4> = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let lk = lidx >> 4u;
    let lj = (lidx & 15u) * 4u;
    let colb = wid.x * 64u + lj;
    let rowl = wid.y * 64u + lj;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + lk;
        let base = t * d;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var a2: f32 = 0.0;
        var a3: f32 = 0.0;
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        var b2: f32 = 0.0;
        var b3: f32 = 0.0;
        if (t < m && colb < d) { a0 = phi[base + colb]; }
        if (t < m && colb + 1u < d) { a1 = phi[base + colb + 1u]; }
        if (t < m && colb + 2u < d) { a2 = phi[base + colb + 2u]; }
        if (t < m && colb + 3u < d) { a3 = phi[base + colb + 3u]; }
        if (t < m && rowl < rr && ro + rowl < d) { b0 = phi[base + ro + rowl]; }
        if (t < m && rowl + 1u < rr && ro + rowl + 1u < d) { b1 = phi[base + ro + rowl + 1u]; }
        if (t < m && rowl + 2u < rr && ro + rowl + 2u < d) { b2 = phi[base + ro + rowl + 2u]; }
        if (t < m && rowl + 3u < rr && ro + rowl + 3u < d) { b3 = phi[base + ro + rowl + 3u]; }
        wa[lidx] = vec4<f32>(a0, a1, a2, a3);
        wb[lidx] = vec4<f32>(b0, b1, b2, b3);
        workgroupBarrier();
        tl[0] = vec4<f32>(0.0);
        tl[1] = vec4<f32>(0.0);
        tl[2] = vec4<f32>(0.0);
        tl[3] = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wa[k * 16u + lid.x];
            let bv = wb[k * 16u + lid.y];
            tl[0] = tl[0] + av * bv.x;
            tl[1] = tl[1] + av * bv.y;
            tl[2] = tl[2] + av * bv.z;
            tl[3] = tl[3] + av * bv.w;
        }
        acc[0] = acc[0] + tl[0];
        acc[1] = acc[1] + tl[1];
        acc[2] = acc[2] + tl[2];
        acc[3] = acc[3] + tl[3];
        workgroupBarrier();
    }
    let c0 = (wid.x * 16u + lid.x) * 4u;
    let r0 = (wid.y * 16u + lid.y) * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        let v = acc[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = c0 + j;
            let idx = lr * d + c;
            if (lr < rr && ro + lr < d && c < d) {
                h[idx] = h[idx] + v[j];
            }
        }
    }
}
