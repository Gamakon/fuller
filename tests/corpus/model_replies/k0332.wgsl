// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0f);
        pw[k & 15u] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 4u; k = k + 1u) {
            i0 = lidx + (128u * k);
            i1 = t + (i0 >> 4u);
            i2 = (i0 & 15u) * 4u;
            f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
            f4 = 0.0f; f5 = 0.0f; f6 = 0.0f; f7 = 0.0f;
            if (i1 < cfg.m) {
                i3 = (wid.x * 64u) + i2;
                i4 = (i1 * cfg.d) + i3;
                if (i3 < cfg.d) { f0 = phi[i4]; }
                if ((i3 + 1u) < cfg.d) { f1 = phi[i4 + 1u]; }
                if ((i3 + 2u) < cfg.d) { f2 = phi[i4 + 2u]; }
                if ((i3 + 3u) < cfg.d) { f3 = phi[i4 + 3u]; }
                i3 = (wid.y * 64u) + i2;
                i5 = cfg.r_off + i3;
                i4 = (i1 * cfg.d) + i5;
                if ((i3 < cfg.r_rows) && (i5 < cfg.d)) { f4 = phi[i4]; }
                if (((i3 + 1u) < cfg.r_rows) && ((i5 + 1u) < cfg.d)) { f5 = phi[i4 + 1u]; }
                if (((i3 + 2u) < cfg.r_rows) && ((i5 + 2u) < cfg.d)) { f6 = phi[i4 + 2u]; }
                if (((i3 + 3u) < cfg.r_rows) && ((i5 + 3u) < cfg.d)) { f7 = phi[i4 + 3u]; }
            }
            wg_va[i0 & 511u] = vec4<f32>(f0, f1, f2, f3);
            wg_vb[i0 & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
        v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
        for (k = 0u; k < 16u; k = k + 1u) {
            i0 = k * 32u;
            v8 = wg_va[(i0 + (lid.x * 2u)) & 511u];
            v9 = wg_va[((i0 + (lid.x * 2u)) + 1u) & 511u];
            v10 = wg_vb[(i0 + lid.y) & 511u];
            v11 = wg_va[((i0 + 16u) + (lid.x * 2u)) & 511u];
            v12 = wg_va[(((i0 + 16u) + (lid.x * 2u)) + 1u) & 511u];
            v13 = wg_vb[((i0 + 16u) + lid.y) & 511u];
            v0 = v0 + ((vec4<f32>(v10.x) * v8) + (vec4<f32>(v13.x) * v11));
            v1 = v1 + ((vec4<f32>(v10.x) * v9) + (vec4<f32>(v13.x) * v12));
            v2 = v2 + ((vec4<f32>(v10.y) * v8) + (vec4<f32>(v13.y) * v11));
            v3 = v3 + ((vec4<f32>(v10.y) * v9) + (vec4<f32>(v13.y) * v12));
            v4 = v4 + ((vec4<f32>(v10.z) * v8) + (vec4<f32>(v13.z) * v11));
            v5 = v5 + ((vec4<f32>(v10.z) * v9) + (vec4<f32>(v13.z) * v12));
            v6 = v6 + ((vec4<f32>(v10.w) * v8) + (vec4<f32>(v13.w) * v11));
            v7 = v7 + ((vec4<f32>(v10.w) * v9) + (vec4<f32>(v13.w) * v12));
        }
        if (((t >> 5u) & 1u) == 0u) {
            pv[0u] = pv[0u] + v0; pv[1u] = pv[1u] + v1; pv[2u] = pv[2u] + v2; pv[3u] = pv[3u] + v3;
            pv[4u] = pv[4u] + v4; pv[5u] = pv[5u] + v5; pv[6u] = pv[6u] + v6; pv[7u] = pv[7u] + v7;
        } else {
            pw[0u] = pw[0u] + v0; pw[1u] = pw[1u] + v1; pw[2u] = pw[2u] + v2; pw[3u] = pw[3u] + v3;
            pw[4u] = pw[4u] + v4; pw[5u] = pw[5u] + v5; pw[6u] = pw[6u] + v6; pw[7u] = pw[7u] + v7;
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = pv[k & 15u] + pw[k & 15u];
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        if ((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d)) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                if (((gid.x * 8u) + k2) < cfg.d) {
                    if ((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) && (((gid.x * 8u) + min(k2, 7u)) < cfg.d) && ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d)) {
                        h[(((gid.y * 4u) + min(k, 3u)) * cfg.d) + ((gid.x * 8u) + min(k2, 7u))] = h[(((gid.y * 4u) + min(k, 3u)) * cfg.d) + ((gid.x * 8u) + min(k2, 7u))] + pv[((k * 2u) + (k2 >> 2u)) & 15u][k2 & 3u];
                    }
                }
            }
        }
    }
}
