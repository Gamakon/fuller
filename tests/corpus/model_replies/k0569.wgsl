// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0);
    var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0);
    var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var i7: u32 = 0u; var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
            i7 = lidx + (128u * k2);
            i8 = t + (i7 >> 4u);
            i9 = (i7 & 15u) * 4u;
            f0 = 0.0; f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0; f6 = 0.0; f7 = 0.0;
            if (i8 < cfg.m) {
                i10 = (i8 * cfg.d) + (wid.x * 64u) + i9;
                if (((wid.x * 64u) + i9) < cfg.d) { f0 = phi[i10]; }
                if (((wid.x * 64u) + i9 + 1u) < cfg.d) { f1 = phi[i10 + 1u]; }
                if (((wid.x * 64u) + i9 + 2u) < cfg.d) { f2 = phi[i10 + 2u]; }
                if (((wid.x * 64u) + i9 + 3u) < cfg.d) { f3 = phi[i10 + 3u]; }
                i10 = (i8 * cfg.d) + cfg.r_off + (wid.y * 64u) + i9;
                if (((wid.y * 64u) + i9) < cfg.r_rows) {
                    if ((cfg.r_off + (wid.y * 64u) + i9) < cfg.d) { f4 = phi[i10]; }
                }
                if (((wid.y * 64u) + i9 + 1u) < cfg.r_rows) {
                    if ((cfg.r_off + (wid.y * 64u) + i9 + 1u) < cfg.d) { f5 = phi[i10 + 1u]; }
                }
                if (((wid.y * 64u) + i9 + 2u) < cfg.r_rows) {
                    if ((cfg.r_off + (wid.y * 64u) + i9 + 2u) < cfg.d) { f6 = phi[i10 + 2u]; }
                }
                if (((wid.y * 64u) + i9 + 3u) < cfg.r_rows) {
                    if ((cfg.r_off + (wid.y * 64u) + i9 + 3u) < cfg.d) { f7 = phi[i10 + 3u]; }
                }
            }
            wg_va[i7 & 511u] = vec4<f32>(f0, f1, f2, f3);
            wg_vb[i7 & 511u] = vec4<f32>(f4, f5, f6, f7);
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0); v1 = vec4<f32>(0.0); v2 = vec4<f32>(0.0); v3 = vec4<f32>(0.0);
        v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
        for (var k3: u32 = 0u; k3 < 16u; k3 = k3 + 1u) {
            i11 = (k3 * 32u) + (lid.x * 2u);
            v8 = wg_va[i11 & 511u];
            v9 = wg_va[(i11 + 1u) & 511u];
            v10 = wg_va[(i11 + 16u) & 511u];
            v11 = wg_va[(i11 + 17u) & 511u];
            v12 = wg_vb[((k3 * 32u) + lid.y) & 511u];
            v13 = wg_vb[((k3 * 32u) + 16u + lid.y) & 511u];
            v0 = v0 + ((vec4<f32>(v12.x) * v8) + (vec4<f32>(v13.x) * v10));
            v1 = v1 + ((vec4<f32>(v12.x) * v9) + (vec4<f32>(v13.x) * v11));
            v2 = v2 + ((vec4<f32>(v12.y) * v8) + (vec4<f32>(v13.y) * v10));
            v3 = v3 + ((vec4<f32>(v12.y) * v9) + (vec4<f32>(v13.y) * v11));
            v4 = v4 + ((vec4<f32>(v12.z) * v8) + (vec4<f32>(v13.z) * v10));
            v5 = v5 + ((vec4<f32>(v12.z) * v9) + (vec4<f32>(v13.z) * v11));
            v6 = v6 + ((vec4<f32>(v12.w) * v8) + (vec4<f32>(v13.w) * v10));
            v7 = v7 + ((vec4<f32>(v12.w) * v9) + (vec4<f32>(v13.w) * v11));
        }
        v14 = v14 + v0; v15 = v15 + v1; v16 = v16 + v2; v17 = v17 + v3;
        v18 = v18 + v4; v19 = v19 + v5; v20 = v20 + v6; v21 = v21 + v7;
        workgroupBarrier();
    }
    pv[0] = v14; pv[1] = v15; pv[2] = v16; pv[3] = v17;
    pv[4] = v18; pv[5] = v19; pv[6] = v20; pv[7] = v21;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let lr = (wid.y * 64u) + (lid.y * 4u) + r;
        if (lr < cfg.r_rows) {
            if ((cfg.r_off + lr) < cfg.d) {
                for (var c: u32 = 0u; c < 8u; c = c + 1u) {
                    let col = (wid.x * 64u) + (lid.x * 8u) + c;
                    if (col < cfg.d) {
                        h[(lr * cfg.d) + col] = h[(lr * cfg.d) + col] + pv[((r * 2u) + (c >> 2u)) & 15u][c & 3u];
                    }
                }
            }
        }
    }
}
