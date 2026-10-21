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
    var i8: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u; var i16: u32 = 0u; var i17: u32 = 0u; var i18: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var ps: array<vec4<f32>, 16>;
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0);
        pw[k & 15u] = vec4<f32>(0.0);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            ps[k & 15u] = vec4<f32>(0.0);
        }
        for (k = 0u; k < 4u; k = k + 1u) {
            i8 = t + (k * 32u);
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                i13 = lidx + (128u * k2);
                i14 = i8 + (i13 >> 4u);
                i15 = (i13 & 15u) * 4u;
                f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0;
                f4 = 0.0; f5 = 0.0; f6 = 0.0; f7 = 0.0;
                if (i14 < cfg.m) {
                    i16 = ((i14 * cfg.d) + (wid.x * 64u)) + i15;
                    if (((wid.x * 64u) + i15) < cfg.d) { f0 = phi[i16]; }
                    if ((((wid.x * 64u) + i15) + 1u) < cfg.d) { f1 = phi[i16 + 1u]; }
                    if ((((wid.x * 64u) + i15) + 2u) < cfg.d) { f2 = phi[i16 + 2u]; }
                    if ((((wid.x * 64u) + i15) + 3u) < cfg.d) { f3 = phi[i16 + 3u]; }
                    i16 = (((i14 * cfg.d) + cfg.r_off) + (wid.y * 64u)) + i15;
                    if ((((wid.y * 64u) + i15) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + i15) < cfg.d)) { f4 = phi[i16]; }
                    if (((((wid.y * 64u) + i15) + 1u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i15) + 1u) < cfg.d)) { f5 = phi[i16 + 1u]; }
                    if (((((wid.y * 64u) + i15) + 2u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i15) + 2u) < cfg.d)) { f6 = phi[i16 + 2u]; }
                    if (((((wid.y * 64u) + i15) + 3u) < cfg.r_rows) && ((((cfg.r_off + (wid.y * 64u)) + i15) + 3u) < cfg.d)) { f7 = phi[i16 + 3u]; }
                }
                wg_va[i13 & 511u] = vec4<f32>(f0, f1, f2, f3);
                wg_vb[i13 & 511u] = vec4<f32>(f4, f5, f6, f7);
            }
            workgroupBarrier();
            v0 = vec4<f32>(0.0); v1 = vec4<f32>(0.0); v2 = vec4<f32>(0.0); v3 = vec4<f32>(0.0);
            v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
            for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
                i17 = k2 * 32u;
                i18 = i17 + (lid.y * 2u);
                v8 = wg_va[(i17 + lid.x) & 511u];
                v9 = wg_va[((i17 + 16u) + lid.x) & 511u];
                v10 = wg_vb[i18 & 511u];
                v11 = wg_vb[(i18 + 1u) & 511u];
                v12 = wg_vb[(i18 + 16u) & 511u];
                v13 = wg_vb[(i18 + 17u) & 511u];
                v0 = v0 + ((vec4<f32>(v10.x) * v8) + (vec4<f32>(v12.x) * v9));
                v1 = v1 + ((vec4<f32>(v10.y) * v8) + (vec4<f32>(v12.y) * v9));
                v2 = v2 + ((vec4<f32>(v10.z) * v8) + (vec4<f32>(v12.z) * v9));
                v3 = v3 + ((vec4<f32>(v10.w) * v8) + (vec4<f32>(v12.w) * v9));
                v4 = v4 + ((vec4<f32>(v11.x) * v8) + (vec4<f32>(v13.x) * v9));
                v5 = v5 + ((vec4<f32>(v11.y) * v8) + (vec4<f32>(v13.y) * v9));
                v6 = v6 + ((vec4<f32>(v11.z) * v8) + (vec4<f32>(v13.z) * v9));
                v7 = v7 + ((vec4<f32>(v11.w) * v8) + (vec4<f32>(v13.w) * v9));
            }
            ps[0u] = ps[0u] + v0;
            ps[1u] = ps[1u] + v1;
            ps[2u] = ps[2u] + v2;
            ps[3u] = ps[3u] + v3;
            ps[4u] = ps[4u] + v4;
            ps[5u] = ps[5u] + v5;
            ps[6u] = ps[6u] + v6;
            ps[7u] = ps[7u] + v7;
            workgroupBarrier();
        }
        if (((t >> 7u) & 1u) == 0u) {
            for (k = 0u; k < 8u; k = k + 1u) {
                pv[k & 15u] = pv[k & 15u] + ps[k & 15u];
            }
        } else {
            for (k = 0u; k < 8u; k = k + 1u) {
                pw[k & 15u] = pw[k & 15u] + ps[k & 15u];
            }
        }
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        pv[k & 15u] = pv[k & 15u] + pw[k & 15u];
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        if ((((gid.y * 8u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 8u) + k)) < cfg.d)) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((gid.x * 4u) + k2) < cfg.d) {
                    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[((k) & 15u)][(k2) & 3u]; }
                }
            }
        }
    }
}
