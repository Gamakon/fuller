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
    var f24: f32 = 0.0; var f25: f32 = 0.0; var f26: f32 = 0.0; var f27: f32 = 0.0; var f28: f32 = 0.0; var f29: f32 = 0.0; var f30: f32 = 0.0; var f31: f32 = 0.0; 
    var f32r: f32 = 0.0; var f33: f32 = 0.0; var f34: f32 = 0.0; var f35: f32 = 0.0; var f36: f32 = 0.0; var f37: f32 = 0.0; var f38: f32 = 0.0; var f39: f32 = 0.0; 
    var f40: f32 = 0.0; var f41: f32 = 0.0; var f42: f32 = 0.0; var f43: f32 = 0.0; var f44: f32 = 0.0; var f45: f32 = 0.0; var f46: f32 = 0.0; var f47: f32 = 0.0; 
    var f48: f32 = 0.0; var f49: f32 = 0.0; var f50: f32 = 0.0; var f51: f32 = 0.0; var f52: f32 = 0.0; var f53: f32 = 0.0; var f54: f32 = 0.0; var f55: f32 = 0.0; 
    var f56: f32 = 0.0; var f57: f32 = 0.0; var f58: f32 = 0.0; var f59: f32 = 0.0; var f60: f32 = 0.0; var f61: f32 = 0.0; var f62: f32 = 0.0; var f63: f32 = 0.0; 
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); 
    var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0); var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0); var v22: vec4<f32> = vec4<f32>(0.0); var v23: vec4<f32> = vec4<f32>(0.0); 
    var v24: vec4<f32> = vec4<f32>(0.0); var v25: vec4<f32> = vec4<f32>(0.0); var v26: vec4<f32> = vec4<f32>(0.0); var v27: vec4<f32> = vec4<f32>(0.0); var v28: vec4<f32> = vec4<f32>(0.0); var v29: vec4<f32> = vec4<f32>(0.0); var v30: vec4<f32> = vec4<f32>(0.0); var v31: vec4<f32> = vec4<f32>(0.0); 
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; 
    var i8: u32 = 0u; var i9: u32 = 0u; var i10: u32 = 0u; var i11: u32 = 0u; var i12: u32 = 0u; var i13: u32 = 0u; var i14: u32 = 0u; var i15: u32 = 0u; 
    var i16: u32 = 0u; var i17: u32 = 0u; var i18: u32 = 0u; var i19: u32 = 0u; var i20: u32 = 0u; var i21: u32 = 0u; var i22: u32 = 0u; var i23: u32 = 0u; 
    var i24: u32 = 0u; var i25: u32 = 0u; var i26: u32 = 0u; var i27: u32 = 0u; var i28: u32 = 0u; var i29: u32 = 0u; var i30: u32 = 0u; var i31: u32 = 0u; 
    var i32r: u32 = 0u; var i33: u32 = 0u; var i34: u32 = 0u; var i35: u32 = 0u; var i36: u32 = 0u; var i37: u32 = 0u; var i38: u32 = 0u; var i39: u32 = 0u; 
    var i40: u32 = 0u; var i41: u32 = 0u; var i42: u32 = 0u; var i43: u32 = 0u; var i44: u32 = 0u; var i45: u32 = 0u; var i46: u32 = 0u; var i47: u32 = 0u; 
    var i48: u32 = 0u; var i49: u32 = 0u; var i50: u32 = 0u; var i51: u32 = 0u; var i52: u32 = 0u; var i53: u32 = 0u; var i54: u32 = 0u; var i55: u32 = 0u; 
    var i56: u32 = 0u; var i57: u32 = 0u; var i58: u32 = 0u; var i59: u32 = 0u; var i60: u32 = 0u; var i61: u32 = 0u; var i62: u32 = 0u; var i63: u32 = 0u; 
    var b0: bool = false; var b1: bool = false; var b2: bool = false; var b3: bool = false; var b4: bool = false; var b5: bool = false; var b6: bool = false; var b7: bool = false; 
    var b8: bool = false; var b9: bool = false; var b10: bool = false; var b11: bool = false; var b12: bool = false; var b13: bool = false; var b14: bool = false; var b15: bool = false; 
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u; var k4: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>; var pe: array<f32, 64>; var pf: array<f32, 64>;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>; var pi: array<u32, 16>; var pj: array<u32, 16>; var pk: array<u32, 16>; var pm: array<bool, 16>; var pn: array<bool, 16>;
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    f16r = 0.0f;
    f17 = 0.0f;
    f18 = 0.0f;
    f19 = 0.0f;
    f20 = 0.0f;
    f21 = 0.0f;
    f22 = 0.0f;
    f23 = 0.0f;
    i6 = ((lid.y * 16u) + lid.x);
    i2 = (i6 / 32u);
    i3 = (i6 % 32u);
    i4 = (cfg.r_off + ((wid.y * 32u) + i3));
    i5 = ((wid.x * 32u) + i3);
    b0 = ((((wid.y * 32u) + i3) < cfg.r_rows) && (i4 < cfg.d));
    b1 = (i5 < cfg.d);
    i7 = lid.y;
    i8 = (lid.y + 16u);
    i9 = lid.x;
    i10 = (lid.x + 16u);
    for (t = 0u; t < cfg.m; t = t + 32u) {
    f8 = 0.0f;
    f9 = 0.0f;
    f10 = 0.0f;
    f11 = 0.0f;
    f12 = 0.0f;
    f13 = 0.0f;
    f14 = 0.0f;
    f15 = 0.0f;
    if (((t + (((lid.y * 16u) + lid.x) / 32u)) < cfg.m)) {
    if (((((wid.y * 32u) + (((lid.y * 16u) + lid.x) % 32u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + (((lid.y * 16u) + lid.x) % 32u))) < cfg.d))) {
    f8 = phi[(((t + (((lid.y * 16u) + lid.x) / 32u)) * cfg.d) + (cfg.r_off + ((wid.y * 32u) + (((lid.y * 16u) + lid.x) % 32u))))];
    }
    if ((((wid.x * 32u) + (((lid.y * 16u) + lid.x) % 32u)) < cfg.d)) {
    f9 = phi[(((t + (((lid.y * 16u) + lid.x) / 32u)) * cfg.d) + ((wid.x * 32u) + (((lid.y * 16u) + lid.x) % 32u)))];
    }
    }
    if ((((t + 8u) + i2) < cfg.m)) {
    if (b0) {
    f10 = phi[((((t + 8u) + i2) * cfg.d) + i4)];
    }
    if (b1) {
    f11 = phi[((((t + 8u) + i2) * cfg.d) + i5)];
    }
    }
    if ((((t + 16u) + i2) < cfg.m)) {
    if (b0) {
    f12 = phi[((((t + 16u) + i2) * cfg.d) + i4)];
    }
    if (b1) {
    f13 = phi[((((t + 16u) + i2) * cfg.d) + i5)];
    }
    }
    if ((((t + 24u) + i2) < cfg.m)) {
    if (b0) {
    f14 = phi[((((t + 24u) + i2) * cfg.d) + i4)];
    }
    if (b1) {
    f15 = phi[((((t + 24u) + i2) * cfg.d) + i5)];
    }
    }
    wg_a[(i6 & 2047u)] = f8;
    wg_b[(i6 & 2047u)] = f9;
    wg_a[((i6 + 256u) & 2047u)] = f10;
    wg_b[((i6 + 256u) & 2047u)] = f11;
    wg_a[((i6 + 512u) & 2047u)] = f12;
    wg_b[((i6 + 512u) & 2047u)] = f13;
    wg_a[((i6 + 768u) & 2047u)] = f14;
    wg_b[((i6 + 768u) & 2047u)] = f15;
    workgroupBarrier();
    for (k = 0u; k < 32u; k = k + 4u) {
    f24 = wg_a[(((k * 32u) + i7) & 2047u)];
    f25 = wg_a[(((k * 32u) + i8) & 2047u)];
    f26 = wg_b[(((k * 32u) + i9) & 2047u)];
    f27 = wg_b[(((k * 32u) + i10) & 2047u)];
    f0 = (f0 + (f24 * f26));
    f1 = (f1 + (f24 * f27));
    f2 = (f2 + (f25 * f26));
    f3 = (f3 + (f25 * f27));
    f24 = wg_a[((((k * 32u) + 32u) + i7) & 2047u)];
    f25 = wg_a[((((k * 32u) + 32u) + i8) & 2047u)];
    f26 = wg_b[((((k * 32u) + 32u) + i9) & 2047u)];
    f27 = wg_b[((((k * 32u) + 32u) + i10) & 2047u)];
    f4 = (f4 + (f24 * f26));
    f5 = (f5 + (f24 * f27));
    f6 = (f6 + (f25 * f26));
    f7 = (f7 + (f25 * f27));
    f24 = wg_a[((((k * 32u) + 64u) + i7) & 2047u)];
    f25 = wg_a[((((k * 32u) + 64u) + i8) & 2047u)];
    f26 = wg_b[((((k * 32u) + 64u) + i9) & 2047u)];
    f27 = wg_b[((((k * 32u) + 64u) + i10) & 2047u)];
    f16r = (f16r + (f24 * f26));
    f17 = (f17 + (f24 * f27));
    f18 = (f18 + (f25 * f26));
    f19 = (f19 + (f25 * f27));
    f24 = wg_a[((((k * 32u) + 96u) + i7) & 2047u)];
    f25 = wg_a[((((k * 32u) + 96u) + i8) & 2047u)];
    f26 = wg_b[((((k * 32u) + 96u) + i9) & 2047u)];
    f27 = wg_b[((((k * 32u) + 96u) + i10) & 2047u)];
    f20 = (f20 + (f24 * f26));
    f21 = (f21 + (f24 * f27));
    f22 = (f22 + (f25 * f26));
    f23 = (f23 + (f25 * f27));
    }
    workgroupBarrier();
    }
    if ((((((wid.y * 32u) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + lid.y)) < cfg.d)) && (((wid.x * 32u) + lid.x) < cfg.d))) {
    if (((wid.y * 32u + lid.y)) < cfg.r_rows && ((wid.x * 32u + lid.x)) < cfg.d && (cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d) { h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] + ((f0 + f4) + (f16r + f20)); }
    }
    if ((((((wid.y * 32u) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 32u) + lid.y)) < cfg.d)) && ((((wid.x * 32u) + lid.x) + 16u) < cfg.d))) {
    if (((wid.y * 32u + lid.y)) < cfg.r_rows && (((wid.x * 32u + lid.x) + 16u)) < cfg.d && (cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d) { h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + ((f1 + f5) + (f17 + f21)); }
    }
    if (((((((wid.y * 32u) + lid.y) + 16u) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 32u) + lid.y) + 16u)) < cfg.d)) && (((wid.x * 32u) + lid.x) < cfg.d))) {
    if ((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows && ((wid.x * 32u + lid.x)) < cfg.d && (cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] + ((f2 + f6) + (f18 + f22)); }
    }
    if (((((((wid.y * 32u) + lid.y) + 16u) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 32u) + lid.y) + 16u)) < cfg.d)) && ((((wid.x * 32u) + lid.x) + 16u) < cfg.d))) {
    if ((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows && (((wid.x * 32u + lid.x) + 16u)) < cfg.d && (cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + ((f3 + f7) + (f19 + f23)); }
    }
}
