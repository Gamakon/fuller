// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0);
    var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i2: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u;
    var rowb: u32 = 0u; var colb: u32 = 0u; var row: u32 = 0u; var col: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    colb = wid.x * 128u;
    rowb = wid.y * 64u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pv[(k) & 15u] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 128u) {
        v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
        v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
        for (k = 0u; k < 8u; k = k + 1u) {
            i2 = t + (k * 16u);
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                i4 = lidx + (256u * k2);
                i5 = i2 + (i4 >> 5u);
                i6 = colb + ((i4 & 31u) * 4u);
                f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
                if (i5 < cfg.m) {
                    i7 = (i5 * cfg.d) + i6;
                    if (i6 < cfg.d) { f0 = phi[i7]; }
                    if ((i6 + 1u) < cfg.d) { f1 = phi[i7 + 1u]; }
                    if ((i6 + 2u) < cfg.d) { f2 = phi[i7 + 2u]; }
                    if ((i6 + 3u) < cfg.d) { f3 = phi[i7 + 3u]; }
                }
                wg_va[(i4) & 511u] = vec4<f32>(f0, f1, f2, f3);
            }
            i4 = lidx;
            i5 = i2 + (i4 >> 4u);
            i6 = rowb + ((i4 & 15u) * 4u);
            f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
            if (i5 < cfg.m) {
                i7 = (i5 * cfg.d) + cfg.r_off + i6;
                if ((i6 < cfg.r_rows) && ((cfg.r_off + i6) < cfg.d)) { f0 = phi[i7]; }
                if (((i6 + 1u) < cfg.r_rows) && ((cfg.r_off + i6 + 1u) < cfg.d)) { f1 = phi[i7 + 1u]; }
                if (((i6 + 2u) < cfg.r_rows) && ((cfg.r_off + i6 + 2u) < cfg.d)) { f2 = phi[i7 + 2u]; }
                if (((i6 + 3u) < cfg.r_rows) && ((cfg.r_off + i6 + 3u) < cfg.d)) { f3 = phi[i7 + 3u]; }
            }
            wg_vb[(i4) & 511u] = vec4<f32>(f0, f1, f2, f3);
            workgroupBarrier();
            v8 = vec4<f32>(0.0f); v9 = vec4<f32>(0.0f); v10 = vec4<f32>(0.0f); v11 = vec4<f32>(0.0f);
            v12 = vec4<f32>(0.0f); v13 = vec4<f32>(0.0f); v14 = vec4<f32>(0.0f); v15 = vec4<f32>(0.0f);
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                i8 = k2 * 64u;
                i9 = (k2 * 32u) + (lid.y * 2u);
                v16 = wg_va[(i8 + lid.x) & 511u];
                v17 = wg_va[(i8 + 32u + lid.x) & 511u];
                v18 = wg_vb[(i9) & 511u];
                v19 = wg_vb[(i9 + 1u) & 511u];
                v20 = wg_vb[(i9 + 16u) & 511u];
                v21 = wg_vb[(i9 + 17u) & 511u];
                v8 = (v8 + ((vec4<f32>(v18.x) * v16) + (vec4<f32>(v20.x) * v17)));
                v9 = (v9 + ((vec4<f32>(v18.y) * v16) + (vec4<f32>(v20.y) * v17)));
                v10 = (v10 + ((vec4<f32>(v18.z) * v16) + (vec4<f32>(v20.z) * v17)));
                v11 = (v11 + ((vec4<f32>(v18.w) * v16) + (vec4<f32>(v20.w) * v17)));
                v12 = (v12 + ((vec4<f32>(v19.x) * v16) + (vec4<f32>(v21.x) * v17)));
                v13 = (v13 + ((vec4<f32>(v19.y) * v16) + (vec4<f32>(v21.y) * v17)));
                v14 = (v14 + ((vec4<f32>(v19.z) * v16) + (vec4<f32>(v21.z) * v17)));
                v15 = (v15 + ((vec4<f32>(v19.w) * v16) + (vec4<f32>(v21.w) * v17)));
            }
            v0 = (v0 + v8); v1 = (v1 + v9); v2 = (v2 + v10); v3 = (v3 + v11);
            v4 = (v4 + v12); v5 = (v5 + v13); v6 = (v6 + v14); v7 = (v7 + v15);
            workgroupBarrier();
        }
        i10 = ((t >> 7u) & 1u) * 8u;
        pv[(i10) & 15u] = (pv[(i10) & 15u] + v0);
        pv[(i10 + 1u) & 15u] = (pv[(i10 + 1u) & 15u] + v1);
        pv[(i10 + 2u) & 15u] = (pv[(i10 + 2u) & 15u] + v2);
        pv[(i10 + 3u) & 15u] = (pv[(i10 + 3u) & 15u] + v3);
        pv[(i10 + 4u) & 15u] = (pv[(i10 + 4u) & 15u] + v4);
        pv[(i10 + 5u) & 15u] = (pv[(i10 + 5u) & 15u] + v5);
        pv[(i10 + 6u) & 15u] = (pv[(i10 + 6u) & 15u] + v6);
        pv[(i10 + 7u) & 15u] = (pv[(i10 + 7u) & 15u] + v7);
    }
    for (k = 0u; k < 8u; k = k + 1u) {
        row = rowb + (lid.y * 8u) + k;
        if ((row < cfg.r_rows) && ((cfg.r_off + row) < cfg.d)) {
            v8 = (pv[(k) & 15u] + pv[(k + 8u) & 15u]);
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                col = colb + (lid.x * 4u) + k2;
                if (col < cfg.d) {
                    h[(row * cfg.d) + col] = h[(row * cfg.d) + col] + v8[(k2) & 3u];
                }
            }
        }
    }
}
