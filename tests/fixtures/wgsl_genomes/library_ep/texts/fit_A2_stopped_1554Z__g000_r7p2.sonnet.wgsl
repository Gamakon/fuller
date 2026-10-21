// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
    var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0;
    var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 64u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            v0 = vec4<f32>(0.0f);
            v1 = vec4<f32>(0.0f);
            if ((t + ((lidx + (256u * k)) >> 4u)) < cfg.m) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    if (((wid.x * 64u) + (((lidx + (256u * k)) & 15u) * 4u) + k2) < cfg.d) {
                        v0[k2 & 3u] = phi[((t + ((lidx + (256u * k)) >> 4u)) * cfg.d) + (wid.x * 64u) + (((lidx + (256u * k)) & 15u) * 4u) + k2];
                    }
                    if ((((wid.y * 64u) + (((lidx + (256u * k)) & 15u) * 4u) + k2) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + (((lidx + (256u * k)) & 15u) * 4u) + k2) < cfg.d)) {
                        v1[k2 & 3u] = phi[((t + ((lidx + (256u * k)) >> 4u)) * cfg.d) + cfg.r_off + (wid.y * 64u) + (((lidx + (256u * k)) & 15u) * 4u) + k2];
                    }
                }
            }
            wg_va[(lidx + (256u * k)) & 511u] = v0;
            wg_vb[(lidx + (256u * k)) & 511u] = v1;
        }
        workgroupBarrier();
        f0 = 0.0f; f1 = 0.0f; f2 = 0.0f; f3 = 0.0f;
        f4 = 0.0f; f5 = 0.0f; f6 = 0.0f; f7 = 0.0f;
        f8 = 0.0f; f9 = 0.0f; f10 = 0.0f; f11 = 0.0f;
        f12 = 0.0f; f13 = 0.0f; f14 = 0.0f; f15 = 0.0f;
        for (k = 0u; k < 32u; k = k + 1u) {
            v2 = wg_vb[((k * 16u) + lid.y) & 511u];
            v3 = wg_va[((k * 16u) + lid.x) & 511u];
            f0 = f0 + v2.x * v3.x;
            f1 = f1 + v2.x * v3.y;
            f2 = f2 + v2.x * v3.z;
            f3 = f3 + v2.x * v3.w;
            f4 = f4 + v2.y * v3.x;
            f5 = f5 + v2.y * v3.y;
            f6 = f6 + v2.y * v3.z;
            f7 = f7 + v2.y * v3.w;
            f8 = f8 + v2.z * v3.x;
            f9 = f9 + v2.z * v3.y;
            f10 = f10 + v2.z * v3.z;
            f11 = f11 + v2.z * v3.w;
            f12 = f12 + v2.w * v3.x;
            f13 = f13 + v2.w * v3.y;
            f14 = f14 + v2.w * v3.z;
            f15 = f15 + v2.w * v3.w;
        }
        i0 = ((t >> 5u) & 3u) * 16u;
        pa[(i0 + 0u) & 63u] = pa[(i0 + 0u) & 63u] + f0;
        pa[(i0 + 1u) & 63u] = pa[(i0 + 1u) & 63u] + f1;
        pa[(i0 + 2u) & 63u] = pa[(i0 + 2u) & 63u] + f2;
        pa[(i0 + 3u) & 63u] = pa[(i0 + 3u) & 63u] + f3;
        pa[(i0 + 4u) & 63u] = pa[(i0 + 4u) & 63u] + f4;
        pa[(i0 + 5u) & 63u] = pa[(i0 + 5u) & 63u] + f5;
        pa[(i0 + 6u) & 63u] = pa[(i0 + 6u) & 63u] + f6;
        pa[(i0 + 7u) & 63u] = pa[(i0 + 7u) & 63u] + f7;
        pa[(i0 + 8u) & 63u] = pa[(i0 + 8u) & 63u] + f8;
        pa[(i0 + 9u) & 63u] = pa[(i0 + 9u) & 63u] + f9;
        pa[(i0 + 10u) & 63u] = pa[(i0 + 10u) & 63u] + f10;
        pa[(i0 + 11u) & 63u] = pa[(i0 + 11u) & 63u] + f11;
        pa[(i0 + 12u) & 63u] = pa[(i0 + 12u) & 63u] + f12;
        pa[(i0 + 13u) & 63u] = pa[(i0 + 13u) & 63u] + f13;
        pa[(i0 + 14u) & 63u] = pa[(i0 + 14u) & 63u] + f14;
        pa[(i0 + 15u) & 63u] = pa[(i0 + 15u) & 63u] + f15;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        if (((wid.y * 64u) + (lid.y * 4u) + k) < cfg.r_rows && (cfg.r_off + (wid.y * 64u) + (lid.y * 4u) + k) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((wid.x * 64u) + (lid.x * 4u) + k2) < cfg.d) {
                    h[(((wid.y * 64u) + (lid.y * 4u) + k) * cfg.d) + (wid.x * 64u) + (lid.x * 4u) + k2] = h[(((wid.y * 64u) + (lid.y * 4u) + k) * cfg.d) + (wid.x * 64u) + (lid.x * 4u) + k2] + ((pa[((k * 4u) + k2) & 63u] + pa[(((k * 4u) + k2) + 16u) & 63u]) + (pa[(((k * 4u) + k2) + 32u) & 63u] + pa[(((k * 4u) + k2) + 48u) & 63u]));
                }
            }
        }
    }
}
