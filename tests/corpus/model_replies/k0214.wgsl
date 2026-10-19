// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 256u * k;
            let item = t + (e >> 4u);
            let g = e & 15u;
            let cb = wid.x * 64u + g * 4u;
            let rl = wid.y * 64u + g * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                if (cb < d) { a.x = phi[base + cb]; }
                if (cb + 1u < d) { a.y = phi[base + cb + 1u]; }
                if (cb + 2u < d) { a.z = phi[base + cb + 2u]; }
                if (cb + 3u < d) { a.w = phi[base + cb + 3u]; }
                if (rl < r_rows && r_off + rl < d) { b.x = phi[base + r_off + rl]; }
                if (rl + 1u < r_rows && r_off + rl + 1u < d) { b.y = phi[base + r_off + rl + 1u]; }
                if (rl + 2u < r_rows && r_off + rl + 2u < d) { b.z = phi[base + r_off + rl + 2u]; }
                if (rl + 3u < r_rows && r_off + rl + 3u < d) { b.w = phi[base + r_off + rl + 3u]; }
            }
            wg_va[e & 511u] = a;
            wg_vb[e & 511u] = b;
        }
        workgroupBarrier();
        var a0 = acc[0];
        var a1 = acc[1];
        var a2 = acc[2];
        var a3 = acc[3];
        for (var i: u32 = 0u; i < 32u; i = i + 1u) {
            let va = wg_va[(i * 16u + lid.x) & 511u];
            let vb = wg_vb[(i * 16u + lid.y) & 511u];
            a0 = a0 + vec4<f32>(vb.x) * va;
            a1 = a1 + vec4<f32>(vb.y) * va;
            a2 = a2 + vec4<f32>(vb.z) * va;
            a3 = a3 + vec4<f32>(vb.w) * va;
        }
        acc[0] = a0;
        acc[1] = a1;
        acc[2] = a2;
        acc[3] = a3;
        workgroupBarrier();
    }
    let rbase = (wid.y * 16u + lid.y) * 4u;
    let cbase = (wid.x * 16u + lid.x) * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rbase + j;
        if (lr < r_rows && r_off + lr < d) {
            let v = acc[j & 3u];
            let ro = lr * d;
            if (cbase < d) { h[ro + cbase] = h[ro + cbase] + v.x; }
            if (cbase + 1u < d) { h[ro + cbase + 1u] = h[ro + cbase + 1u] + v.y; }
            if (cbase + 2u < d) { h[ro + cbase + 2u] = h[ro + cbase + 2u] + v.z; }
            if (cbase + 3u < d) { h[ro + cbase + 3u] = h[ro + cbase + 3u] + v.w; }
        }
    }
}
