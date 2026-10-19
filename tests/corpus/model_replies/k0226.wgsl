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
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    let ltt = lidx >> 4u;
    let lcv = lidx & 15u;
    let colbase = wid.x * 64u + lcv * 4u;
    let rowbase = wid.y * 64u + lcv * 4u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + ltt;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            if (colbase < d) { va.x = phi[base + colbase]; }
            if (colbase + 1u < d) { va.y = phi[base + colbase + 1u]; }
            if (colbase + 2u < d) { va.z = phi[base + colbase + 2u]; }
            if (colbase + 3u < d) { va.w = phi[base + colbase + 3u]; }
            let g = r_off + rowbase;
            if (rowbase < r_rows && g < d) { vb.x = phi[base + g]; }
            if (rowbase + 1u < r_rows && g + 1u < d) { vb.y = phi[base + g + 1u]; }
            if (rowbase + 2u < r_rows && g + 2u < d) { vb.z = phi[base + g + 2u]; }
            if (rowbase + 3u < r_rows && g + 3u < d) { vb.w = phi[base + g + 3u]; }
        }
        wa[lidx & 255u] = va;
        wb[lidx & 255u] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wa[(k * 16u + lid.x) & 255u];
            let b = wb[(k * 16u + lid.y) & 255u];
            a0 = a0 + vec4<f32>(b.x) * a;
            a1 = a1 + vec4<f32>(b.y) * a;
            a2 = a2 + vec4<f32>(b.z) * a;
            a3 = a3 + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    let c0 = wid.x * 64u + lid.x * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < r_rows && r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let ro = lr * d;
            if (c0 < d) { h[ro + c0] = h[ro + c0] + v.x; }
            if (c0 + 1u < d) { h[ro + c0 + 1u] = h[ro + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[ro + c0 + 2u] = h[ro + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[ro + c0 + 3u] = h[ro + c0 + 3u] + v.w; }
        }
    }
}
