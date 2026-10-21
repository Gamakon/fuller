// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0; 
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; 
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; 
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u; var k4: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>;
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    }
    i2 = 0u;
    i3 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    i4 = 0u;
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.d)) {
    f0 = phi[((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + (wid.x * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    if (((((wid.y * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u))) < cfg.d))) {
    f1 = phi[(((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    }
    wg_a[(((lidx + (256u * k))) & 2047u)] = f0;
    wg_b[(((lidx + (256u * k))) & 2047u)] = f1;
    }
    workgroupBarrier();
    i5 = 0u;
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pb[((k) & 63u)] = 0.0f;
    }
    i6 = 0u;
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 32u; k = k + 2u) {
    f0 = wg_a[((((k * 64u) + (lid.x * 4u)) + 0u) & 2047u)];
    f1 = wg_a[((((k * 64u) + (lid.x * 4u)) + 1u) & 2047u)];
    f2 = wg_a[((((k * 64u) + (lid.x * 4u)) + 2u) & 2047u)];
    f3 = wg_a[((((k * 64u) + (lid.x * 4u)) + 3u) & 2047u)];
    f4 = wg_a[(((((k * 64u) + 64u) + (lid.x * 4u)) + 0u) & 2047u)];
    f5 = wg_a[(((((k * 64u) + 64u) + (lid.x * 4u)) + 1u) & 2047u)];
    f6 = wg_a[(((((k * 64u) + 64u) + (lid.x * 4u)) + 2u) & 2047u)];
    f7 = wg_a[(((((k * 64u) + 64u) + (lid.x * 4u)) + 3u) & 2047u)];
    i7 = 0u;
    i8 = 0u;
    i9 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    f8 = wg_b[((((k * 64u) + (lid.y * 4u)) + k2) & 2047u)];
    f9 = wg_b[(((((k * 64u) + 64u) + (lid.y * 4u)) + k2) & 2047u)];
    pb[((((k2 * 4u) + 0u)) & 63u)] = (pb[((((k2 * 4u) + 0u)) & 63u)] + ((f8 * f0) + (f9 * f4)));
    pb[((((k2 * 4u) + 1u)) & 63u)] = (pb[((((k2 * 4u) + 1u)) & 63u)] + ((f8 * f1) + (f9 * f5)));
    pb[((((k2 * 4u) + 2u)) & 63u)] = (pb[((((k2 * 4u) + 2u)) & 63u)] + ((f8 * f2) + (f9 * f6)));
    pb[((((k2 * 4u) + 3u)) & 63u)] = (pb[((((k2 * 4u) + 3u)) & 63u)] + ((f8 * f3) + (f9 * f7)));
    }
    }
    i13 = 0u;
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = (pa[((k) & 63u)] + pb[((k) & 63u)]);
    }
    workgroupBarrier();
    }
    i0 = 0u;
    i1 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + (lid.y * 4u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 4u)) + k)) < cfg.d))) {
    i14 = 0u;
    i8 = 0u;
    i9 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + (lid.x * 4u)) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pa[((((k * 4u) + k2)) & 63u)]; }
    }
    }
    }
    }
}
