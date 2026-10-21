// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var tot: array<vec4<f32>, 4>;
    tot[0] = vec4<f32>(0.0);
    tot[1] = vec4<f32>(0.0);
    tot[2] = vec4<f32>(0.0);
    tot[3] = vec4<f32>(0.0);
    let litem = lidx >> 4u;
    let lc4 = (lidx & 15u) * 4u;
    let colbase = wid.x * 64u + lc4;
    let rowloc = wid.y * 64u + lc4;
    let rowglob = cfg.r_off + rowloc;
    let nt = (cfg.m + 15u) / 16u;
    for (var it: u32 = 0u; it < nt; it = it + 1u) {
        let item = it * 16u + litem;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (item < cfg.m) {
            let base = item * cfg.d;
            if (colbase < cfg.d) { a.x = phi[base + colbase]; }
            if (colbase + 1u < cfg.d) { a.y = phi[base + colbase + 1u]; }
            if (colbase + 2u < cfg.d) { a.z = phi[base + colbase + 2u]; }
            if (colbase + 3u < cfg.d) { a.w = phi[base + colbase + 3u]; }
            if (rowloc < cfg.r_rows && rowglob < cfg.d) { b.x = phi[base + rowglob]; }
            if (rowloc + 1u < cfg.r_rows && rowglob + 1u < cfg.d) { b.y = phi[base + rowglob + 1u]; }
            if (rowloc + 2u < cfg.r_rows && rowglob + 2u < cfg.d) { b.z = phi[base + rowglob + 2u]; }
            if (rowloc + 3u < cfg.r_rows && rowglob + 3u < cfg.d) { b.w = phi[base + rowglob + 3u]; }
        }
        wg_va[lidx] = a;
        wg_vb[lidx] = b;
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            p0 = p0 + vec4<f32>(vb.x) * va;
            p1 = p1 + vec4<f32>(vb.y) * va;
            p2 = p2 + vec4<f32>(vb.z) * va;
            p3 = p3 + vec4<f32>(vb.w) * va;
        }
        tot[0] = tot[0] + p0;
        tot[1] = tot[1] + p1;
        tot[2] = tot[2] + p2;
        tot[3] = tot[3] + p3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let v = tot[i];
            let c0 = wid.x * 64u + lid.x * 4u;
            let ro = lr * cfg.d;
            if (c0 < cfg.d) { h[ro + c0] = h[ro + c0] + v.x; }
            if (c0 + 1u < cfg.d) { h[ro + c0 + 1u] = h[ro + c0 + 1u] + v.y; }
            if (c0 + 2u < cfg.d) { h[ro + c0 + 2u] = h[ro + c0 + 2u] + v.z; }
            if (c0 + 3u < cfg.d) { h[ro + c0 + 3u] = h[ro + c0 + 3u] + v.w; }
        }
    }
}
