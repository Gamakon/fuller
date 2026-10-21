// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 16u; k = k + 1u) {
        pv[k & 15u] = vec4<f32>(0.0f);
        pw[k & 15u] = vec4<f32>(0.0f);
    }
    i0 = wid.x * 128u;
    i1 = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 128u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                i2 = lidx + (k2 * 256u);
                i3 = (t + (k * 16u)) + (i2 >> 5u);
                i4 = i0 + ((i2 & 31u) * 4u);
                f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
                if (i3 < cfg.m) {
                    i5 = (i3 * cfg.d) + i4;
                    if (i4 < cfg.d) { f0 = phi[i5]; }
                    if ((i4 + 1u) < cfg.d) { f1 = phi[i5 + 1u]; }
                    if ((i4 + 2u) < cfg.d) { f2 = phi[i5 + 2u]; }
                    if ((i4 + 3u) < cfg.d) { f3 = phi[i5 + 3u]; }
                }
                wg_va[i2 & 511u] = vec4<f32>(f0, f1, f2, f3);
            }
            i3 = (t + (k * 16u)) + (lidx >> 4u);
            i4 = i1 + ((lidx & 15u) * 4u);
            f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
            if (i3 < cfg.m) {
                i5 = ((i3 * cfg.d) + cfg.r_off) + i4;
                if ((i4 < cfg.r_rows) && ((cfg.r_off + i4) < cfg.d)) { f0 = phi[i5]; }
                if (((i4 + 1u) < cfg.r_rows) && ((cfg.r_off + i4 + 1u) < cfg.d)) { f1 = phi[i5 + 1u]; }
                if (((i4 + 2u) < cfg.r_rows) && ((cfg.r_off + i4 + 2u) < cfg.d)) { f2 = phi[i5 + 2u]; }
                if (((i4 + 3u) < cfg.r_rows) && ((cfg.r_off + i4 + 3u) < cfg.d)) { f3 = phi[i5 + 3u]; }
            }
            wg_vb[lidx & 511u] = vec4<f32>(f0, f1, f2, f3);
            workgroupBarrier();
            v0 = vec4<f32>(0.0f); v1 = vec4<f32>(0.0f); v2 = vec4<f32>(0.0f); v3 = vec4<f32>(0.0f);
            v4 = vec4<f32>(0.0f); v5 = vec4<f32>(0.0f); v6 = vec4<f32>(0.0f); v7 = vec4<f32>(0.0f);
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                i6 = k2 * 64u;
                i7 = k2 * 32u;
                v8 = wg_va[(i6 + lid.x) & 511u];
                v9 = wg_va[(i6 + 16u + lid.x) & 511u];
                v10 = wg_va[(i6 + 32u + lid.x) & 511u];
                v11 = wg_va[(i6 + 48u + lid.x) & 511u];
                v12 = wg_vb[(i7 + lid.y) & 511u];
                v13 = wg_vb[(i7 + 16u + lid.y) & 511u];
                v0 = v0 + ((v8 * vec4<f32>(v12.x)) + (v10 * vec4<f32>(v13.x)));
                v1 = v1 + ((v9 * vec4<f32>(v12.x)) + (v11 * vec4<f32>(v13.x)));
                v2 = v2 + ((v8 * vec4<f32>(v12.y)) + (v10 * vec4<f32>(v13.y)));
                v3 = v3 + ((v9 * vec4<f32>(v12.y)) + (v11 * vec4<f32>(v13.y)));
                v4 = v4 + ((v8 * vec4<f32>(v12.z)) + (v10 * vec4<f32>(v13.z)));
                v5 = v5 + ((v9 * vec4<f32>(v12.z)) + (v11 * vec4<f32>(v13.z)));
                v6 = v6 + ((v8 * vec4<f32>(v12.w)) + (v10 * vec4<f32>(v13.w)));
                v7 = v7 + ((v9 * vec4<f32>(v12.w)) + (v11 * vec4<f32>(v13.w)));
            }
            pw[0u] = pw[0u] + v0;
            pw[1u] = pw[1u] + v1;
            pw[2u] = pw[2u] + v2;
            pw[3u] = pw[3u] + v3;
            pw[4u] = pw[4u] + v4;
            pw[5u] = pw[5u] + v5;
            pw[6u] = pw[6u] + v6;
            pw[7u] = pw[7u] + v7;
            workgroupBarrier();
        }
        pv[0u] = pv[0u] + pw[0u];
        pv[1u] = pv[1u] + pw[1u];
        pv[2u] = pv[2u] + pw[2u];
        pv[3u] = pv[3u] + pw[3u];
        pv[4u] = pv[4u] + pw[4u];
        pv[5u] = pv[5u] + pw[5u];
        pv[6u] = pv[6u] + pw[6u];
        pv[7u] = pv[7u] + pw[7u];
        pw[0u] = vec4<f32>(0.0f);
        pw[1u] = vec4<f32>(0.0f);
        pw[2u] = vec4<f32>(0.0f);
        pw[3u] = vec4<f32>(0.0f);
        pw[4u] = vec4<f32>(0.0f);
        pw[5u] = vec4<f32>(0.0f);
        pw[6u] = vec4<f32>(0.0f);
        pw[7u] = vec4<f32>(0.0f);
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        i2 = i1 + (lid.y * 4u) + k;
        if ((i2 < cfg.r_rows) && ((cfg.r_off + i2) < cfg.d)) {
            for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
                for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
                    i3 = i0 + (k2 * 64u) + (lid.x * 4u) + k3;
                    if (i3 < cfg.d) {
                        i4 = (i2 * cfg.d) + i3;
                        h[i4] = h[i4] + pv[((k * 2u) + k2) & 15u][k3 & 3u];
                    }
                }
            }
        }
    }
}
