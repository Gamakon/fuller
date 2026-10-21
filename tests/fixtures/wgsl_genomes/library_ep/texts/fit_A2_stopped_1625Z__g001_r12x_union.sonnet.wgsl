// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32>; var v1: vec4<f32>; var v2: vec4<f32>; var v3: vec4<f32>;
    var v4: vec4<f32>; var v5: vec4<f32>; var v6: vec4<f32>; var v7: vec4<f32>;
    var v8: vec4<f32>; var v9: vec4<f32>; var v10: vec4<f32>; var v11: vec4<f32>;
    var v12: vec4<f32>; var v13: vec4<f32>;
    var v14 = vec4<f32>(0.0); var v15 = vec4<f32>(0.0); var v16 = vec4<f32>(0.0); var v17 = vec4<f32>(0.0);
    var v18 = vec4<f32>(0.0); var v19 = vec4<f32>(0.0); var v20 = vec4<f32>(0.0); var v21 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            let i12 = t + (k * 32u);
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i19 = lidx + (128u * k2);
                let i20 = i12 + (i19 >> 4u);
                let i21 = (i19 & 15u) * 4u;
                var f0 = 0.0f; var f1 = 0.0f; var f2 = 0.0f; var f3 = 0.0f;
                var f4 = 0.0f; var f5 = 0.0f; var f6 = 0.0f; var f7 = 0.0f;
                if (i20 < cfg.m) {
                    let ca = (wid.x * 64u) + i21;
                    let i22 = (i20 * cfg.d) + ca;
                    if (ca < cfg.d) { f0 = phi[i22]; }
                    if (ca + 1u < cfg.d) { f1 = phi[i22 + 1u]; }
                    if (ca + 2u < cfg.d) { f2 = phi[i22 + 2u]; }
                    if (ca + 3u < cfg.d) { f3 = phi[i22 + 3u]; }
                    let lb = (wid.y * 64u) + i21;
                    let i23 = (i20 * cfg.d) + cfg.r_off + lb;
                    if (lb < cfg.r_rows && cfg.r_off + lb < cfg.d) { f4 = phi[i23]; }
                    if (lb + 1u < cfg.r_rows && cfg.r_off + lb + 1u < cfg.d) { f5 = phi[i23 + 1u]; }
                    if (lb + 2u < cfg.r_rows && cfg.r_off + lb + 2u < cfg.d) { f6 = phi[i23 + 2u]; }
                    if (lb + 3u < cfg.r_rows && cfg.r_off + lb + 3u < cfg.d) { f7 = phi[i23 + 3u]; }
                }
                wg_va[i19 & 511u] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[i19 & 511u] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
            v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
            for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
                let i23 = (k2 * 32u) + (lid.x * 2u);
                v8 = wg_va[i23 & 511u];
                v9 = wg_va[(i23 + 1u) & 511u];
                v10 = wg_va[(i23 + 16u) & 511u];
                v11 = wg_va[(i23 + 17u) & 511u];
                v12 = wg_vb[((k2 * 32u) + lid.y) & 511u];
                v13 = wg_vb[((k2 * 32u) + 16u + lid.y) & 511u];
                v0 = v0 + ((vec4<f32>(v12[0u]) * v8) + (vec4<f32>(v13[0u]) * v10));
                v1 = v1 + ((vec4<f32>(v12[0u]) * v9) + (vec4<f32>(v13[0u]) * v11));
                v2 = v2 + ((vec4<f32>(v12[1u]) * v8) + (vec4<f32>(v13[1u]) * v10));
                v3 = v3 + ((vec4<f32>(v12[1u]) * v9) + (vec4<f32>(v13[1u]) * v11));
                v4 = v4 + ((vec4<f32>(v12[2u]) * v8) + (vec4<f32>(v13[2u]) * v10));
                v5 = v5 + ((vec4<f32>(v12[2u]) * v9) + (vec4<f32>(v13[2u]) * v11));
                v6 = v6 + ((vec4<f32>(v12[3u]) * v8) + (vec4<f32>(v13[3u]) * v10));
                v7 = v7 + ((vec4<f32>(v12[3u]) * v9) + (vec4<f32>(v13[3u]) * v11));
            }
            v14 = v14 + v0; v15 = v15 + v1; v16 = v16 + v2; v17 = v17 + v3;
            v18 = v18 + v4; v19 = v19 + v5; v20 = v20 + v6; v21 = v21 + v7;
            workgroupBarrier();
        }
        pv[0u] = pv[0u] + v14; pv[1u] = pv[1u] + v15; pv[2u] = pv[2u] + v16; pv[3u] = pv[3u] + v17;
        pv[4u] = pv[4u] + v18; pv[5u] = pv[5u] + v19; pv[6u] = pv[6u] + v20; pv[7u] = pv[7u] + v21;
        v14 = vec4<f32>(0.0f); v15 = vec4<f32>(0.0f); v16 = vec4<f32>(0.0f); v17 = vec4<f32>(0.0f);
        v18 = vec4<f32>(0.0f); v19 = vec4<f32>(0.0f); v20 = vec4<f32>(0.0f); v21 = vec4<f32>(0.0f);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = (gid.y * 4u) + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                let c = (gid.x * 8u) + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[((k * 2u) + (k2 >> 2u)) & 15u][k2 & 3u];
                }
            }
        }
    }
}
