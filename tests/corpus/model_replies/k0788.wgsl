// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0);
    var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i4: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u;
    var i14: u32 = 0u; var i15: u32 = 0u; var i16: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
        v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
        v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
        for (k = 0u; k < 8u; k = k + 1u) {
            i4 = (t + (k * 16u));
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                i7 = (lidx + (256u * k2));
                i8 = (i4 + (i7 >> 5u));
                i9 = ((wid.x * 128u) + ((i7 & 31u) * 4u));
                f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
                if ((i8 < cfg.m)) {
                    i10 = ((i8 * cfg.d) + i9);
                    if ((i9 < cfg.d)) { f0 = phi[i10]; }
                    if (((i9 + 1u) < cfg.d)) { f1 = phi[(i10 + 1u)]; }
                    if (((i9 + 2u) < cfg.d)) { f2 = phi[(i10 + 2u)]; }
                    if (((i9 + 3u) < cfg.d)) { f3 = phi[(i10 + 3u)]; }
                }
                wg_va[((i7) & 511u)] = vec4<f32>(f0, f1, f2, f3);
            }
            i8 = (i4 + (lidx >> 4u));
            i9 = ((wid.y * 64u) + ((lidx & 15u) * 4u));
            f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
            if ((i8 < cfg.m)) {
                i10 = (((i8 * cfg.d) + cfg.r_off) + i9);
                if (((i9 < cfg.r_rows) && ((cfg.r_off + i9) < cfg.d))) { f0 = phi[i10]; }
                if ((((i9 + 1u) < cfg.r_rows) && (((cfg.r_off + i9) + 1u) < cfg.d))) { f1 = phi[(i10 + 1u)]; }
                if ((((i9 + 2u) < cfg.r_rows) && (((cfg.r_off + i9) + 2u) < cfg.d))) { f2 = phi[(i10 + 2u)]; }
                if ((((i9 + 3u) < cfg.r_rows) && (((cfg.r_off + i9) + 3u) < cfg.d))) { f3 = phi[(i10 + 3u)]; }
            }
            wg_vb[((lidx) & 511u)] = vec4<f32>(f0, f1, f2, f3);
            workgroupBarrier();
            v8 = vec4<f32>(0.0f); v9 = vec4<f32>(0.0f); v10 = vec4<f32>(0.0f); v11 = vec4<f32>(0.0f);
            v12 = vec4<f32>(0.0f); v13 = vec4<f32>(0.0f); v14 = vec4<f32>(0.0f); v15 = vec4<f32>(0.0f);
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                i11 = (k2 * 64u);
                i12 = ((k2 * 32u) + lid.y);
                v16 = wg_va[(((i11 + lid.x)) & 511u)];
                v17 = wg_va[((((i11 + 16u) + lid.x)) & 511u)];
                v18 = wg_va[((((i11 + 32u) + lid.x)) & 511u)];
                v19 = wg_va[((((i11 + 48u) + lid.x)) & 511u)];
                v20 = wg_vb[((i12) & 511u)];
                v21 = wg_vb[(((i12 + 16u)) & 511u)];
                v8 = (v8 + ((vec4<f32>(v20[(0u) & 3u]) * v16) + (vec4<f32>(v21[(0u) & 3u]) * v18)));
                v9 = (v9 + ((vec4<f32>(v20[(1u) & 3u]) * v16) + (vec4<f32>(v21[(1u) & 3u]) * v18)));
                v10 = (v10 + ((vec4<f32>(v20[(2u) & 3u]) * v16) + (vec4<f32>(v21[(2u) & 3u]) * v18)));
                v11 = (v11 + ((vec4<f32>(v20[(3u) & 3u]) * v16) + (vec4<f32>(v21[(3u) & 3u]) * v18)));
                v12 = (v12 + ((vec4<f32>(v20[(0u) & 3u]) * v17) + (vec4<f32>(v21[(0u) & 3u]) * v19)));
                v13 = (v13 + ((vec4<f32>(v20[(1u) & 3u]) * v17) + (vec4<f32>(v21[(1u) & 3u]) * v19)));
                v14 = (v14 + ((vec4<f32>(v20[(2u) & 3u]) * v17) + (vec4<f32>(v21[(2u) & 3u]) * v19)));
                v15 = (v15 + ((vec4<f32>(v20[(3u) & 3u]) * v17) + (vec4<f32>(v21[(3u) & 3u]) * v19)));
            }
            v0 = (v0 + v8); v1 = (v1 + v9); v2 = (v2 + v10); v3 = (v3 + v11);
            v4 = (v4 + v12); v5 = (v5 + v13); v6 = (v6 + v14); v7 = (v7 + v15);
            workgroupBarrier();
        }
        i13 = (((t >> 7u) & 1u) * 8u);
        pv[((i13) & 15u)] = (pv[((i13) & 15u)] + v0);
        pv[(((i13 + 1u)) & 15u)] = (pv[(((i13 + 1u)) & 15u)] + v1);
        pv[(((i13 + 2u)) & 15u)] = (pv[(((i13 + 2u)) & 15u)] + v2);
        pv[(((i13 + 3u)) & 15u)] = (pv[(((i13 + 3u)) & 15u)] + v3);
        pv[(((i13 + 4u)) & 15u)] = (pv[(((i13 + 4u)) & 15u)] + v4);
        pv[(((i13 + 5u)) & 15u)] = (pv[(((i13 + 5u)) & 15u)] + v5);
        pv[(((i13 + 6u)) & 15u)] = (pv[(((i13 + 6u)) & 15u)] + v6);
        pv[(((i13 + 7u)) & 15u)] = (pv[(((i13 + 7u)) & 15u)] + v7);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        i14 = ((gid.y * 4u) + k);
        if (((i14 < cfg.r_rows) && ((cfg.r_off + i14) < cfg.d))) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                i15 = ((k2 >> 2u) * 4u) + k;
                v8 = (pv[((i15) & 15u)] + pv[(((i15 + 8u)) & 15u)]);
                i16 = (((wid.x * 128u) + ((k2 >> 2u) * 64u)) + (lid.x * 4u)) + (k2 & 3u);
                if ((i16 < cfg.d)) {
                    h[((i14 * cfg.d) + i16)] = h[((i14 * cfg.d) + i16)] + v8[(k2) & 3u];
                }
            }
        }
    }
}
