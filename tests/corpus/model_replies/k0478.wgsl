// grid 16 16 2 2
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
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0; var f20: f32 = 0.0; var f21: f32 = 0.0; var f22: f32 = 0.0; var f23: f32 = 0.0;
    var f24: f32 = 0.0; var f25: f32 = 0.0; var f26: f32 = 0.0; var f27: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u;
    var b0: bool = false; var b1: bool = false;
    var t: u32 = 0u; var k: u32 = 0u;
    i0 = ((lid.y * 16u) + lid.x);
    i1 = (i0 / 32u);
    i2 = (i0 % 32u);
    i3 = (cfg.r_off + ((wid.y * 32u) + i2));
    i4 = ((wid.x * 32u) + i2);
    b0 = ((((wid.y * 32u) + i2) < cfg.r_rows) && (i3 < cfg.d));
    b1 = (i4 < cfg.d);
    i5 = lid.y;
    i6 = (lid.y + 16u);
    i7 = lid.x;
    i8 = (lid.x + 16u);
    for (t = 0u; t < cfg.m; t = t + 32u) {
    f16r = 0.0f;
    f17 = 0.0f;
    f18 = 0.0f;
    f19 = 0.0f;
    f20 = 0.0f;
    f21 = 0.0f;
    f22 = 0.0f;
    f23 = 0.0f;
    if ((t + i1) < cfg.m) {
    if (b0) {
    f16r = phi[((t + i1) * cfg.d) + i3];
    }
    if (b1) {
    f17 = phi[((t + i1) * cfg.d) + i4];
    }
    }
    if (((t + 8u) + i1) < cfg.m) {
    if (b0) {
    f18 = phi[(((t + 8u) + i1) * cfg.d) + i3];
    }
    if (b1) {
    f19 = phi[(((t + 8u) + i1) * cfg.d) + i4];
    }
    }
    if (((t + 16u) + i1) < cfg.m) {
    if (b0) {
    f20 = phi[(((t + 16u) + i1) * cfg.d) + i3];
    }
    if (b1) {
    f21 = phi[(((t + 16u) + i1) * cfg.d) + i4];
    }
    }
    if (((t + 24u) + i1) < cfg.m) {
    if (b0) {
    f22 = phi[(((t + 24u) + i1) * cfg.d) + i3];
    }
    if (b1) {
    f23 = phi[(((t + 24u) + i1) * cfg.d) + i4];
    }
    }
    wg_a[i0 & 2047u] = f16r;
    wg_b[i0 & 2047u] = f17;
    wg_a[(i0 + 256u) & 2047u] = f18;
    wg_b[(i0 + 256u) & 2047u] = f19;
    wg_a[(i0 + 512u) & 2047u] = f20;
    wg_b[(i0 + 512u) & 2047u] = f21;
    wg_a[(i0 + 768u) & 2047u] = f22;
    wg_b[(i0 + 768u) & 2047u] = f23;
    workgroupBarrier();
    for (k = 0u; k < 32u; k = k + 4u) {
    f24 = wg_a[((k * 32u) + i5) & 2047u];
    f25 = wg_a[((k * 32u) + i6) & 2047u];
    f26 = wg_b[((k * 32u) + i7) & 2047u];
    f27 = wg_b[((k * 32u) + i8) & 2047u];
    f0 = (f0 + (f24 * f26));
    f1 = (f1 + (f24 * f27));
    f2 = (f2 + (f25 * f26));
    f3 = (f3 + (f25 * f27));
    f24 = wg_a[(((k * 32u) + 32u) + i5) & 2047u];
    f25 = wg_a[(((k * 32u) + 32u) + i6) & 2047u];
    f26 = wg_b[(((k * 32u) + 32u) + i7) & 2047u];
    f27 = wg_b[(((k * 32u) + 32u) + i8) & 2047u];
    f4 = (f4 + (f24 * f26));
    f5 = (f5 + (f24 * f27));
    f6 = (f6 + (f25 * f26));
    f7 = (f7 + (f25 * f27));
    f24 = wg_a[(((k * 32u) + 64u) + i5) & 2047u];
    f25 = wg_a[(((k * 32u) + 64u) + i6) & 2047u];
    f26 = wg_b[(((k * 32u) + 64u) + i7) & 2047u];
    f27 = wg_b[(((k * 32u) + 64u) + i8) & 2047u];
    f8 = (f8 + (f24 * f26));
    f9 = (f9 + (f24 * f27));
    f10 = (f10 + (f25 * f26));
    f11 = (f11 + (f25 * f27));
    f24 = wg_a[(((k * 32u) + 96u) + i5) & 2047u];
    f25 = wg_a[(((k * 32u) + 96u) + i6) & 2047u];
    f26 = wg_b[(((k * 32u) + 96u) + i7) & 2047u];
    f27 = wg_b[(((k * 32u) + 96u) + i8) & 2047u];
    f12 = (f12 + (f24 * f26));
    f13 = (f13 + (f24 * f27));
    f14 = (f14 + (f25 * f26));
    f15 = (f15 + (f25 * f27));
    }
    workgroupBarrier();
    }
    if ((((((wid.y * 32u) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + lid.y)) < cfg.d)) && (((wid.x * 32u) + lid.x) < cfg.d))) {
    h[((wid.y * 32u + lid.y) * cfg.d) + (wid.x * 32u + lid.x)] = h[((wid.y * 32u + lid.y) * cfg.d) + (wid.x * 32u + lid.x)] + ((f0 + f4) + (f8 + f12));
    }
    if ((((((wid.y * 32u) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + lid.y)) < cfg.d)) && ((((wid.x * 32u) + lid.x) + 16u) < cfg.d))) {
    h[((wid.y * 32u + lid.y) * cfg.d) + (wid.x * 32u + lid.x + 16u)] = h[((wid.y * 32u + lid.y) * cfg.d) + (wid.x * 32u + lid.x + 16u)] + ((f1 + f5) + (f9 + f13));
    }
    if ((((((wid.y * 32u) + lid.y) + 16u) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 32u) + lid.y) + 16u)) < cfg.d)) && (((wid.x * 32u) + lid.x) < cfg.d)) {
    h[((wid.y * 32u + lid.y + 16u) * cfg.d) + (wid.x * 32u + lid.x)] = h[((wid.y * 32u + lid.y + 16u) * cfg.d) + (wid.x * 32u + lid.x)] + ((f2 + f6) + (f10 + f14));
    }
    if ((((((wid.y * 32u) + lid.y) + 16u) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 32u) + lid.y) + 16u)) < cfg.d)) && ((((wid.x * 32u) + lid.x) + 16u) < cfg.d)) {
    h[((wid.y * 32u + lid.y + 16u) * cfg.d) + (wid.x * 32u + lid.x + 16u)] = h[((wid.y * 32u + lid.y + 16u) * cfg.d) + (wid.x * 32u + lid.x + 16u)] + ((f3 + f7) + (f11 + f15));
    }
}
