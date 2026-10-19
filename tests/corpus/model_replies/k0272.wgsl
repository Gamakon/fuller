// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pa: array<f32, 64>;
    i0 = wid.x * 128u;
    i1 = wid.y * 64u;
    i2 = i0 + lid.x;
    i3 = i1 + lid.y;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            i4 = lidx + (256u * k);
            i5 = t + (i4 / 128u);
            i6 = i0 + (i4 % 128u);
            f0 = 0.0;
            if ((i5 < cfg.m) && (i6 < cfg.d)) {
                f0 = phi[(i5 * cfg.d) + i6];
            }
            wg_a[i4] = f0;
        }
        for (k = 0u; k < 4u; k = k + 1u) {
            i4 = lidx + (256u * k);
            i5 = t + (i4 / 64u);
            i7 = i1 + (i4 % 64u);
            f1 = 0.0;
            if ((i5 < cfg.m) && (i7 < cfg.r_rows) && ((cfg.r_off + i7) < cfg.d)) {
                f1 = phi[(i5 * cfg.d) + (cfg.r_off + i7)];
            }
            wg_b[i4] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            i4 = (k * 128u) + lid.x;
            i5 = (k * 64u) + lid.y;
            f0 = wg_a[i4];
            f1 = wg_a[i4 + 16u];
            f2 = wg_a[i4 + 32u];
            f3 = wg_a[i4 + 48u];
            f4 = wg_a[i4 + 64u];
            f5 = wg_a[i4 + 80u];
            f6 = wg_a[i4 + 96u];
            f7 = wg_a[i4 + 112u];
            f16r = wg_b[i5];
            f17 = wg_b[i5 + 16u];
            f18 = wg_b[i5 + 32u];
            f19 = wg_b[i5 + 48u];
            pa[0u] = pa[0u] + (f16r * f0); pa[1u] = pa[1u] + (f16r * f1); pa[2u] = pa[2u] + (f16r * f2); pa[3u] = pa[3u] + (f16r * f3);
            pa[4u] = pa[4u] + (f16r * f4); pa[5u] = pa[5u] + (f16r * f5); pa[6u] = pa[6u] + (f16r * f6); pa[7u] = pa[7u] + (f16r * f7);
            pa[8u] = pa[8u] + (f17 * f0); pa[9u] = pa[9u] + (f17 * f1); pa[10u] = pa[10u] + (f17 * f2); pa[11u] = pa[11u] + (f17 * f3);
            pa[12u] = pa[12u] + (f17 * f4); pa[13u] = pa[13u] + (f17 * f5); pa[14u] = pa[14u] + (f17 * f6); pa[15u] = pa[15u] + (f17 * f7);
            pa[16u] = pa[16u] + (f18 * f0); pa[17u] = pa[17u] + (f18 * f1); pa[18u] = pa[18u] + (f18 * f2); pa[19u] = pa[19u] + (f18 * f3);
            pa[20u] = pa[20u] + (f18 * f4); pa[21u] = pa[21u] + (f18 * f5); pa[22u] = pa[22u] + (f18 * f6); pa[23u] = pa[23u] + (f18 * f7);
            pa[24u] = pa[24u] + (f19 * f0); pa[25u] = pa[25u] + (f19 * f1); pa[26u] = pa[26u] + (f19 * f2); pa[27u] = pa[27u] + (f19 * f3);
            pa[28u] = pa[28u] + (f19 * f4); pa[29u] = pa[29u] + (f19 * f5); pa[30u] = pa[30u] + (f19 * f6); pa[31u] = pa[31u] + (f19 * f7);
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        i6 = i3 + (16u * k);
        if ((i6 < cfg.r_rows) && ((cfg.r_off + i6) < cfg.d)) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                i7 = i2 + (16u * k2);
                if (i7 < cfg.d) {
                    h[(i6 * cfg.d) + i7] = h[(i6 * cfg.d) + i7] + pa[(k * 8u) + k2];
                }
            }
        }
    }
}
