// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0);
    var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0);
    var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let cl = (lidx & 15u) * 4u;
    let ir = lidx >> 4u;
    let ca = wid.x * 64u + cl;
    let rb = wid.y * 64u + cl;
    let ra = cfg.r_off + rb;
    let a1 = ca < cfg.d;
    let a2 = (ca + 1u) < cfg.d;
    let a3 = (ca + 2u) < cfg.d;
    let a4 = (ca + 3u) < cfg.d;
    let c1 = (rb < cfg.r_rows) && (ra < cfg.d);
    let c2 = ((rb + 1u) < cfg.r_rows) && ((ra + 1u) < cfg.d);
    let c3 = ((rb + 2u) < cfg.r_rows) && ((ra + 2u) < cfg.d);
    let c4 = ((rb + 3u) < cfg.r_rows) && ((ra + 3u) < cfg.d);
    for (t = 0u; t < cfg.m; t = t + 128u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let i8 = t + k * 32u + ir;
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let it = i8 + k2 * 8u;
                f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0;
                f4 = 0.0; f5 = 0.0; f6 = 0.0; f7 = 0.0;
                if (it < cfg.m) {
                    let base = it * cfg.d;
                    let pa = base + ca;
                    if (a1) { f0 = phi[pa]; }
                    if (a2) { f1 = phi[pa + 1u]; }
                    if (a3) { f2 = phi[pa + 2u]; }
                    if (a4) { f3 = phi[pa + 3u]; }
                    let pb = base + ra;
                    if (c1) { f4 = phi[pb]; }
                    if (c2) { f5 = phi[pb + 1u]; }
                    if (c3) { f6 = phi[pb + 2u]; }
                    if (c4) { f7 = phi[pb + 3u]; }
                }
                let sl = (lidx + 128u * k2) & 511u;
                wg_va[sl] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[sl] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            v0 = vec4<f32>(0.0); v1 = vec4<f32>(0.0); v2 = vec4<f32>(0.0); v3 = vec4<f32>(0.0);
            v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
            for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i17 = k2 * 32u;
                let i18 = i17 + lid.y * 2u;
                v8 = wg_va[(i17 + lid.x) & 511u];
                v9 = wg_va[(i17 + 16u + lid.x) & 511u];
                v10 = wg_vb[i18 & 511u];
                v11 = wg_vb[(i18 + 1u) & 511u];
                v12 = wg_vb[(i18 + 16u) & 511u];
                v13 = wg_vb[(i18 + 17u) & 511u];
                v0 = v0 + (vec4<f32>(v10[0]) * v8 + vec4<f32>(v12[0]) * v9);
                v1 = v1 + (vec4<f32>(v10[1]) * v8 + vec4<f32>(v12[1]) * v9);
                v2 = v2 + (vec4<f32>(v10[2]) * v8 + vec4<f32>(v12[2]) * v9);
                v3 = v3 + (vec4<f32>(v10[3]) * v8 + vec4<f32>(v12[3]) * v9);
                v4 = v4 + (vec4<f32>(v11[0]) * v8 + vec4<f32>(v13[0]) * v9);
                v5 = v5 + (vec4<f32>(v11[1]) * v8 + vec4<f32>(v13[1]) * v9);
                v6 = v6 + (vec4<f32>(v11[2]) * v8 + vec4<f32>(v13[2]) * v9);
                v7 = v7 + (vec4<f32>(v11[3]) * v8 + vec4<f32>(v13[3]) * v9);
            }
            v14 = v14 + v0;
            v15 = v15 + v1;
            v16 = v16 + v2;
            v17 = v17 + v3;
            v18 = v18 + v4;
            v19 = v19 + v5;
            v20 = v20 + v6;
            v21 = v21 + v7;
            workgroupBarrier();
        }
        pv[0] = pv[0] + v14;
        pv[1] = pv[1] + v15;
        pv[2] = pv[2] + v16;
        pv[3] = pv[3] + v17;
        pv[4] = pv[4] + v18;
        pv[5] = pv[5] + v19;
        pv[6] = pv[6] + v20;
        pv[7] = pv[7] + v21;
        v14 = vec4<f32>(0.0); v15 = vec4<f32>(0.0); v16 = vec4<f32>(0.0); v17 = vec4<f32>(0.0);
        v18 = vec4<f32>(0.0); v19 = vec4<f32>(0.0); v20 = vec4<f32>(0.0); v21 = vec4<f32>(0.0);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        if ((gid.y * 8u + k) < cfg.r_rows && (cfg.r_off + gid.y * 8u + k) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if ((gid.x * 4u + k2) < cfg.d) {
                    let idx = (gid.y * 8u + k) * cfg.d + gid.x * 4u + k2;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
