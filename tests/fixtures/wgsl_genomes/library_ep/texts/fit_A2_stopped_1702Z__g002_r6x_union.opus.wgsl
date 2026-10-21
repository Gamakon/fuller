// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0);
    var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0);
    var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let tb = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i4 = lidx + 128u * k2;
                let item = tb + (i4 >> 4u);
                let i6 = (i4 & 15u) * 4u;
                let cb = wid.x * 64u + i6;
                let rb = wid.y * 64u + i6;
                let gb = cfg.r_off + rb;
                f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0;
                f4 = 0.0; f5 = 0.0; f6 = 0.0; f7 = 0.0;
                if (item < cfg.m) {
                    let ia = item * cfg.d + cb;
                    if (cb < cfg.d) { f0 = phi[ia]; }
                    if (cb + 1u < cfg.d) { f1 = phi[ia + 1u]; }
                    if (cb + 2u < cfg.d) { f2 = phi[ia + 2u]; }
                    if (cb + 3u < cfg.d) { f3 = phi[ia + 3u]; }
                    let ib = item * cfg.d + gb;
                    if (rb < cfg.r_rows && gb < cfg.d) { f4 = phi[ib]; }
                    if (rb + 1u < cfg.r_rows && gb + 1u < cfg.d) { f5 = phi[ib + 1u]; }
                    if (rb + 2u < cfg.r_rows && gb + 2u < cfg.d) { f6 = phi[ib + 2u]; }
                    if (rb + 3u < cfg.r_rows && gb + 3u < cfg.d) { f7 = phi[ib + 3u]; }
                }
                wg_va[i4 & 511u] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[i4 & 511u] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            v0 = vec4<f32>(0.0); v1 = vec4<f32>(0.0); v2 = vec4<f32>(0.0); v3 = vec4<f32>(0.0);
            v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
            for (var k2: u32 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i8 = k2 * 32u;
                let i9 = i8 + lid.y * 2u;
                v8 = wg_va[(i8 + lid.x) & 511u];
                v9 = wg_va[(i8 + 16u + lid.x) & 511u];
                v10 = wg_vb[i9 & 511u];
                v11 = wg_vb[(i9 + 1u) & 511u];
                v12 = wg_vb[(i9 + 16u) & 511u];
                v13 = wg_vb[(i9 + 17u) & 511u];
                v0 = v0 + ((vec4<f32>(v10[0]) * v8) + (vec4<f32>(v12[0]) * v9));
                v1 = v1 + ((vec4<f32>(v10[1]) * v8) + (vec4<f32>(v12[1]) * v9));
                v2 = v2 + ((vec4<f32>(v10[2]) * v8) + (vec4<f32>(v12[2]) * v9));
                v3 = v3 + ((vec4<f32>(v10[3]) * v8) + (vec4<f32>(v12[3]) * v9));
                v4 = v4 + ((vec4<f32>(v11[0]) * v8) + (vec4<f32>(v13[0]) * v9));
                v5 = v5 + ((vec4<f32>(v11[1]) * v8) + (vec4<f32>(v13[1]) * v9));
                v6 = v6 + ((vec4<f32>(v11[2]) * v8) + (vec4<f32>(v13[2]) * v9));
                v7 = v7 + ((vec4<f32>(v11[3]) * v8) + (vec4<f32>(v13[3]) * v9));
            }
            v14 = v14 + v0; v15 = v15 + v1; v16 = v16 + v2; v17 = v17 + v3;
            v18 = v18 + v4; v19 = v19 + v5; v20 = v20 + v6; v21 = v21 + v7;
            workgroupBarrier();
        }
        pv[0u] = pv[0u] + v14; pv[1u] = pv[1u] + v15; pv[2u] = pv[2u] + v16; pv[3u] = pv[3u] + v17;
        pv[4u] = pv[4u] + v18; pv[5u] = pv[5u] + v19; pv[6u] = pv[6u] + v20; pv[7u] = pv[7u] + v21;
        v14 = vec4<f32>(0.0); v15 = vec4<f32>(0.0); v16 = vec4<f32>(0.0); v17 = vec4<f32>(0.0);
        v18 = vec4<f32>(0.0); v19 = vec4<f32>(0.0); v20 = vec4<f32>(0.0); v21 = vec4<f32>(0.0);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let r = gid.y * 8u + k;
        if (r < cfg.r_rows && cfg.r_off + r < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = r * cfg.d + c;
                    h[idx] = h[idx] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
