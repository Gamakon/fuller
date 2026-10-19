// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let lgrp = (lidx & 15u) * 4u;
    let litem = lidx >> 4u;
    let lrow = row0 + lgrp;
    let la = cfg.r_off + lrow;
    let lb = col0 + lgrp;
    let va0 = (lrow < cfg.r_rows) && (la < cfg.d);
    let va1 = (lrow + 1u < cfg.r_rows) && (la + 1u < cfg.d);
    let va2 = (lrow + 2u < cfg.r_rows) && (la + 2u < cfg.d);
    let va3 = (lrow + 3u < cfg.r_rows) && (la + 3u < cfg.d);
    let vb0 = lb < cfg.d;
    let vb1 = lb + 1u < cfg.d;
    let vb2 = lb + 2u < cfg.d;
    let vb3 = lb + 3u < cfg.d;
    var acc0: vec4<f32> = vec4<f32>(0.0); var acc1: vec4<f32> = vec4<f32>(0.0);
    var acc2: vec4<f32> = vec4<f32>(0.0); var acc3: vec4<f32> = vec4<f32>(0.0);
    var acc4: vec4<f32> = vec4<f32>(0.0); var acc5: vec4<f32> = vec4<f32>(0.0);
    var acc6: vec4<f32> = vec4<f32>(0.0); var acc7: vec4<f32> = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let it = t + litem + k * 16u;
            var va: vec4<f32> = vec4<f32>(0.0);
            var vb: vec4<f32> = vec4<f32>(0.0);
            if (it < cfg.m) {
                let base = it * cfg.d;
                if (va0) { va.x = phi[base + la]; }
                if (va1) { va.y = phi[base + la + 1u]; }
                if (va2) { va.z = phi[base + la + 2u]; }
                if (va3) { va.w = phi[base + la + 3u]; }
                if (vb0) { vb.x = phi[base + lb]; }
                if (vb1) { vb.y = phi[base + lb + 1u]; }
                if (vb2) { vb.z = phi[base + lb + 2u]; }
                if (vb3) { vb.w = phi[base + lb + 3u]; }
            }
            wg_va[lidx + k * 256u] = va;
            wg_vb[lidx + k * 256u] = vb;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 32u; s = s + 2u) {
            let a0 = wg_va[s * 16u + lid.y];
            let b0 = wg_vb[s * 16u + lid.x];
            let a1 = wg_va[s * 16u + 16u + lid.y];
            let b1 = wg_vb[s * 16u + 16u + lid.x];
            acc0 = fma(vec4<f32>(a0.x), b0, acc0);
            acc1 = fma(vec4<f32>(a0.y), b0, acc1);
            acc2 = fma(vec4<f32>(a0.z), b0, acc2);
            acc3 = fma(vec4<f32>(a0.w), b0, acc3);
            acc4 = fma(vec4<f32>(a1.x), b1, acc4);
            acc5 = fma(vec4<f32>(a1.y), b1, acc5);
            acc6 = fma(vec4<f32>(a1.z), b1, acc6);
            acc7 = fma(vec4<f32>(a1.w), b1, acc7);
        }
        workgroupBarrier();
    }
    let orow = row0 + lid.y * 4u;
    let ocol = col0 + lid.x * 4u;
    let c0 = ocol < cfg.d;
    let c1 = ocol + 1u < cfg.d;
    let c2 = ocol + 2u < cfg.d;
    let c3 = ocol + 3u < cfg.d;
    let r0 = acc0 + acc4;
    let r1 = acc1 + acc5;
    let r2 = acc2 + acc6;
    let r3 = acc3 + acc7;
    if (orow < cfg.r_rows && cfg.r_off + orow < cfg.d) {
        let idx = orow * cfg.d + ocol;
        if (c0) { h[idx] = h[idx] + r0.x; }
        if (c1) { h[idx + 1u] = h[idx + 1u] + r0.y; }
        if (c2) { h[idx + 2u] = h[idx + 2u] + r0.z; }
        if (c3) { h[idx + 3u] = h[idx + 3u] + r0.w; }
    }
    if (orow + 1u < cfg.r_rows && cfg.r_off + orow + 1u < cfg.d) {
        let idx = (orow + 1u) * cfg.d + ocol;
        if (c0) { h[idx] = h[idx] + r1.x; }
        if (c1) { h[idx + 1u] = h[idx + 1u] + r1.y; }
        if (c2) { h[idx + 2u] = h[idx + 2u] + r1.z; }
        if (c3) { h[idx + 3u] = h[idx + 3u] + r1.w; }
    }
    if (orow + 2u < cfg.r_rows && cfg.r_off + orow + 2u < cfg.d) {
        let idx = (orow + 2u) * cfg.d + ocol;
        if (c0) { h[idx] = h[idx] + r2.x; }
        if (c1) { h[idx + 1u] = h[idx + 1u] + r2.y; }
        if (c2) { h[idx + 2u] = h[idx + 2u] + r2.z; }
        if (c3) { h[idx + 3u] = h[idx + 3u] + r2.w; }
    }
    if (orow + 3u < cfg.r_rows && cfg.r_off + orow + 3u < cfg.d) {
        let idx = (orow + 3u) * cfg.d + ocol;
        if (c0) { h[idx] = h[idx] + r3.x; }
        if (c1) { h[idx + 1u] = h[idx + 1u] + r3.y; }
        if (c2) { h[idx + 2u] = h[idx + 2u] + r3.z; }
        if (c3) { h[idx + 3u] = h[idx + 3u] + r3.w; }
    }
}
