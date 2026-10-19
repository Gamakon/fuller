// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 256u * k;
            let item = e >> 4u;
            let c4 = (e & 15u) * 4u;
            let gt = t + item;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (gt < m) {
                let base = gt * d;
                let c0 = colbase + c4;
                if (c0 < d) { a.x = phi[base + c0]; }
                if (c0 + 1u < d) { a.y = phi[base + c0 + 1u]; }
                if (c0 + 2u < d) { a.z = phi[base + c0 + 2u]; }
                if (c0 + 3u < d) { a.w = phi[base + c0 + 3u]; }
                let l0 = rowbase + c4;
                if (l0 < r_rows && r_off + l0 < d) { b.x = phi[base + r_off + l0]; }
                if (l0 + 1u < r_rows && r_off + l0 + 1u < d) { b.y = phi[base + r_off + l0 + 1u]; }
                if (l0 + 2u < r_rows && r_off + l0 + 2u < d) { b.z = phi[base + r_off + l0 + 2u]; }
                if (l0 + 3u < r_rows && r_off + l0 + 3u < d) { b.w = phi[base + r_off + l0 + 3u]; }
            }
            wg_a[e & 511u] = a;
            wg_b[e & 511u] = b;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 1u) {
            let a = wg_a[(s * 16u + lid.x) & 511u];
            let b = wg_b[(s * 16u + lid.y) & 511u];
            acc0 = acc0 + vec4<f32>(b.x) * a;
            acc1 = acc1 + vec4<f32>(b.y) * a;
            acc2 = acc2 + vec4<f32>(b.z) * a;
            acc3 = acc3 + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    let c0 = colbase + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < r_rows && r_off + lr < d) {
            var v = acc0;
            if (i == 1u) { v = acc1; }
            if (i == 2u) { v = acc2; }
            if (i == 3u) { v = acc3; }
            let ro = lr * d;
            if (c0 < d) { h[ro + c0] = h[ro + c0] + v.x; }
            if (c0 + 1u < d) { h[ro + c0 + 1u] = h[ro + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[ro + c0 + 2u] = h[ro + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[ro + c0 + 3u] = h[ro + c0 + 3u] + v.w; }
        }
    }
}
