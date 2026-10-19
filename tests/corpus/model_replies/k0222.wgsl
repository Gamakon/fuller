// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let item = lidx >> 4u;
    let col = (lidx & 15u) * 4u;
    let cbase = wid.x * 64u + col;
    let rbase = wid.y * 64u + col;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        let tm = t + item;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (tm < m) {
            let rowoff = tm * d;
            if (cbase < d) { a.x = phi[rowoff + cbase]; }
            if (cbase + 1u < d) { a.y = phi[rowoff + cbase + 1u]; }
            if (cbase + 2u < d) { a.z = phi[rowoff + cbase + 2u]; }
            if (cbase + 3u < d) { a.w = phi[rowoff + cbase + 3u]; }
            if (rbase < cfg.r_rows && cfg.r_off + rbase < d) { b.x = phi[rowoff + cfg.r_off + rbase]; }
            if (rbase + 1u < cfg.r_rows && cfg.r_off + rbase + 1u < d) { b.y = phi[rowoff + cfg.r_off + rbase + 1u]; }
            if (rbase + 2u < cfg.r_rows && cfg.r_off + rbase + 2u < d) { b.z = phi[rowoff + cfg.r_off + rbase + 2u]; }
            if (rbase + 3u < cfg.r_rows && cfg.r_off + rbase + 3u < d) { b.w = phi[rowoff + cfg.r_off + rbase + 3u]; }
        }
        wg_a[lidx & 255u] = a;
        wg_b[lidx & 255u] = b;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let av = wg_a[(k * 16u + tx) & 255u];
            let bv = wg_b[(k * 16u + ty) & 255u];
            acc0 = acc0 + vec4<f32>(bv.x) * av;
            acc1 = acc1 + vec4<f32>(bv.y) * av;
            acc2 = acc2 + vec4<f32>(bv.z) * av;
            acc3 = acc3 + vec4<f32>(bv.w) * av;
        }
        workgroupBarrier();
    }
    var accs: array<vec4<f32>, 4>;
    accs[0u] = acc0;
    accs[1u] = acc1;
    accs[2u] = acc2;
    accs[3u] = acc3;
    let c0 = wid.x * 64u + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let av = accs[i & 3u];
            let ro = lr * d;
            if (c0 < d) { h[ro + c0] = h[ro + c0] + av.x; }
            if (c0 + 1u < d) { h[ro + c0 + 1u] = h[ro + c0 + 1u] + av.y; }
            if (c0 + 2u < d) { h[ro + c0 + 2u] = h[ro + c0 + 2u] + av.z; }
            if (c0 + 3u < d) { h[ro + c0 + 3u] = h[ro + c0 + 3u] + av.w; }
        }
    }
}
