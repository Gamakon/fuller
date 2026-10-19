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
    let tid = lid.y * 16u + lid.x;
    let litem = tid >> 4u;
    let lcq = (tid & 15u) * 4u;
    let colbase = wid.x * 64u + lcq;
    let rowbase = wid.y * 64u + lcq;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        let it = t + litem;
        if (it < m) {
            let base = it * d;
            if (colbase + 0u < d) { va.x = phi[base + colbase + 0u]; }
            if (colbase + 1u < d) { va.y = phi[base + colbase + 1u]; }
            if (colbase + 2u < d) { va.z = phi[base + colbase + 2u]; }
            if (colbase + 3u < d) { va.w = phi[base + colbase + 3u]; }
            let g0 = r_off + rowbase;
            if (rowbase + 0u < r_rows && g0 + 0u < d) { vb.x = phi[base + g0 + 0u]; }
            if (rowbase + 1u < r_rows && g0 + 1u < d) { vb.y = phi[base + g0 + 1u]; }
            if (rowbase + 2u < r_rows && g0 + 2u < d) { vb.z = phi[base + g0 + 2u]; }
            if (rowbase + 3u < r_rows && g0 + 3u < d) { vb.w = phi[base + g0 + 3u]; }
        }
        wa[tid & 255u] = va;
        wb[tid & 255u] = vb;
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let av = wa[(kk * 16u + lid.x) & 255u];
            let bv = wb[(kk * 16u + lid.y) & 255u];
            acc0 = acc0 + bv.x * av;
            acc1 = acc1 + bv.y * av;
            acc2 = acc2 + bv.z * av;
            acc3 = acc3 + bv.w * av;
        }
        workgroupBarrier();
    }
    let c0 = wid.x * 64u + lid.x * 4u;
    let r0 = wid.y * 64u + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < r_rows && r_off + lr < d) {
            var a = acc0;
            if (i == 1u) { a = acc1; }
            if (i == 2u) { a = acc2; }
            if (i == 3u) { a = acc3; }
            let rb = lr * d;
            if (c0 + 0u < d) { h[rb + c0 + 0u] = h[rb + c0 + 0u] + a.x; }
            if (c0 + 1u < d) { h[rb + c0 + 1u] = h[rb + c0 + 1u] + a.y; }
            if (c0 + 2u < d) { h[rb + c0 + 2u] = h[rb + c0 + 2u] + a.z; }
            if (c0 + 3u < d) { h[rb + c0 + 3u] = h[rb + c0 + 3u] + a.w; }
        }
    }
}
