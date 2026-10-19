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
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    // staging mapping: thread loads one vec4 (4 consecutive columns / rows) of one item per tile
    let st_item = lidx >> 4u;
    let st_c4 = lidx & 15u;
    let col_base = wid.x * 64u + st_c4 * 4u;
    let row_base = wid.y * 64u + st_c4 * 4u;   // local row
    let grow_base = cfg.r_off + row_base;       // global row

    for (var t: u32 = 0u; t < m; t = t + 16u) {
        let it = t + st_item;
        var vb = vec4<f32>(0.0);
        var va = vec4<f32>(0.0);
        if (it < m) {
            let base = it * d;
            if (col_base + 3u < d) {
                vb = vec4<f32>(phi[base + col_base], phi[base + col_base + 1u], phi[base + col_base + 2u], phi[base + col_base + 3u]);
            } else {
                if (col_base < d) { vb.x = phi[base + col_base]; }
                if (col_base + 1u < d) { vb.y = phi[base + col_base + 1u]; }
                if (col_base + 2u < d) { vb.z = phi[base + col_base + 2u]; }
            }
            if (row_base + 3u < cfg.r_rows && grow_base + 3u < d) {
                va = vec4<f32>(phi[base + grow_base], phi[base + grow_base + 1u], phi[base + grow_base + 2u], phi[base + grow_base + 3u]);
            } else {
                if (row_base < cfg.r_rows && grow_base < d) { va.x = phi[base + grow_base]; }
                if (row_base + 1u < cfg.r_rows && grow_base + 1u < d) { va.y = phi[base + grow_base + 1u]; }
                if (row_base + 2u < cfg.r_rows && grow_base + 2u < d) { va.z = phi[base + grow_base + 2u]; }
            }
        }
        wg_a[lidx & 255u] = va;
        wg_b[lidx & 255u] = vb;
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a = wg_a[(kk * 16u + lid.y) & 255u];
            let b = wg_b[(kk * 16u + lid.x) & 255u];
            acc0 = fma(vec4<f32>(a.x), b, acc0);
            acc1 = fma(vec4<f32>(a.y), b, acc1);
            acc2 = fma(vec4<f32>(a.z), b, acc2);
            acc3 = fma(vec4<f32>(a.w), b, acc3);
        }
        workgroupBarrier();
    }

    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var rv = acc0;
            if (i == 1u) { rv = acc1; }
            if (i == 2u) { rv = acc2; }
            if (i == 3u) { rv = acc3; }
            let o = lr * d + col0;
            if (col0 < d) { h[o] = h[o] + rv.x; }
            if (col0 + 1u < d) { h[o + 1u] = h[o + 1u] + rv.y; }
            if (col0 + 2u < d) { h[o + 2u] = h[o + 2u] + rv.z; }
            if (col0 + 3u < d) { h[o + 3u] = h[o + 3u] + rv.w; }
        }
    }
}
