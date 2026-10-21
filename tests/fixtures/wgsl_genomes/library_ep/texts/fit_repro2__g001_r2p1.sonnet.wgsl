// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0; var f20: f32 = 0.0; var f21: f32 = 0.0; var f22: f32 = 0.0; var f23: f32 = 0.0;
    var f24: f32 = 0.0; var f25: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            f0 = 0.0f;
            f1 = 0.0f;
            i0 = lidx + (256u * k);
            if ((t + (i0 >> 6u)) < cfg.m) {
                if (((wid.x * 64u) + (i0 & 63u)) < cfg.d) {
                    f0 = phi[((t + (i0 >> 6u)) * cfg.d) + (wid.x * 64u) + (i0 & 63u)];
                }
                if ((((wid.y * 64u) + (i0 & 63u)) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + (i0 & 63u)) < cfg.d)) {
                    f1 = phi[((t + (i0 >> 6u)) * cfg.d) + cfg.r_off + (wid.y * 64u) + (i0 & 63u)];
                }
            }
            wg_a[i0 & 2047u] = f0;
            wg_b[i0 & 2047u] = f1;
        }
        workgroupBarrier();
        f2 = 0.0f; f3 = 0.0f; f4 = 0.0f; f5 = 0.0f;
        f6 = 0.0f; f7 = 0.0f; f8 = 0.0f; f9 = 0.0f;
        f10 = 0.0f; f11 = 0.0f; f12 = 0.0f; f13 = 0.0f;
        f14 = 0.0f; f15 = 0.0f; f16r = 0.0f; f17 = 0.0f;
        for (k = 0u; k < 32u; k = k + 1u) {
            i1 = k * 64u;
            f18 = wg_b[(i1 + lid.y) & 2047u];
            f19 = wg_b[(i1 + lid.y + 16u) & 2047u];
            f20 = wg_b[(i1 + lid.y + 32u) & 2047u];
            f21 = wg_b[(i1 + lid.y + 48u) & 2047u];
            f22 = wg_a[(i1 + lid.x) & 2047u];
            f23 = wg_a[(i1 + lid.x + 16u) & 2047u];
            f24 = wg_a[(i1 + lid.x + 32u) & 2047u];
            f25 = wg_a[(i1 + lid.x + 48u) & 2047u];
            f2 = f2 + (f18 * f22);
            f3 = f3 + (f18 * f23);
            f4 = f4 + (f18 * f24);
            f5 = f5 + (f18 * f25);
            f6 = f6 + (f19 * f22);
            f7 = f7 + (f19 * f23);
            f8 = f8 + (f19 * f24);
            f9 = f9 + (f19 * f25);
            f10 = f10 + (f20 * f22);
            f11 = f11 + (f20 * f23);
            f12 = f12 + (f20 * f24);
            f13 = f13 + (f20 * f25);
            f14 = f14 + (f21 * f22);
            f15 = f15 + (f21 * f23);
            f16r = f16r + (f21 * f24);
            f17 = f17 + (f21 * f25);
        }
        i2 = ((t >> 5u) & 1u) * 16u;
        pa[(i2 + 0u) & 63u] = pa[(i2 + 0u) & 63u] + f2;
        pa[(i2 + 1u) & 63u] = pa[(i2 + 1u) & 63u] + f3;
        pa[(i2 + 2u) & 63u] = pa[(i2 + 2u) & 63u] + f4;
        pa[(i2 + 3u) & 63u] = pa[(i2 + 3u) & 63u] + f5;
        pa[(i2 + 4u) & 63u] = pa[(i2 + 4u) & 63u] + f6;
        pa[(i2 + 5u) & 63u] = pa[(i2 + 5u) & 63u] + f7;
        pa[(i2 + 6u) & 63u] = pa[(i2 + 6u) & 63u] + f8;
        pa[(i2 + 7u) & 63u] = pa[(i2 + 7u) & 63u] + f9;
        pa[(i2 + 8u) & 63u] = pa[(i2 + 8u) & 63u] + f10;
        pa[(i2 + 9u) & 63u] = pa[(i2 + 9u) & 63u] + f11;
        pa[(i2 + 10u) & 63u] = pa[(i2 + 10u) & 63u] + f12;
        pa[(i2 + 11u) & 63u] = pa[(i2 + 11u) & 63u] + f13;
        pa[(i2 + 12u) & 63u] = pa[(i2 + 12u) & 63u] + f14;
        pa[(i2 + 13u) & 63u] = pa[(i2 + 13u) & 63u] + f15;
        pa[(i2 + 14u) & 63u] = pa[(i2 + 14u) & 63u] + f16r;
        pa[(i2 + 15u) & 63u] = pa[(i2 + 15u) & 63u] + f17;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            i0 = (wid.y * 64u) + lid.y + (16u * k);
            i1 = (wid.x * 64u) + lid.x + (16u * k2);
            if ((i0 < cfg.r_rows) && (i1 < cfg.d) && ((cfg.r_off + i0) < cfg.d)) {
                h[(i0 * cfg.d) + i1] = h[(i0 * cfg.d) + i1] + (pa[((k * 4u) + k2) & 63u] + pa[(((k * 4u) + k2) + 16u) & 63u]);
            }
        }
    }
}
