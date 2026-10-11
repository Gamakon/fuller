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
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>; var px: array<vec4<f32>, 16>; var py: array<vec4<f32>, 16>; var pi: array<u32, 16>; var pj: array<u32, 16>; var pk: array<u32, 16>; var pm: array<bool, 16>; var pn: array<bool, 16>;
    for (k = 0u; k < 16u; k = k + 1u) {
    pa[((k) & 63u)] = 0.0f;
    pb[((k) & 63u)] = 0.0f;
    }
    for (t = 0u; t < cfg.m; t = t + 16u) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) < cfg.m)) {
    if (((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d))) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))))];
    }
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 2047u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 2047u)] = f1;
    }
    workgroupBarrier();
    i1 = 0u;
    for (k = 0u; k < 16u; k = k + 2u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    pc[((k2) & 63u)] = wg_a[(((((k * 64u) + lid.y) + (16u * k2))) & 2047u)];
    pd[((k2) & 63u)] = wg_b[(((((k * 64u) + lid.x) + (16u * k2))) & 2047u)];
    pe[((k2) & 63u)] = wg_a[((((((k * 64u) + 64u) + lid.y) + (16u * k2))) & 2047u)];
    pf[((k2) & 63u)] = wg_b[((((((k * 64u) + 64u) + lid.x) + (16u * k2))) & 2047u)];
    }
    i3 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i4 = 0u;
    for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
    pa[((((k2 * 4u) + k3)) & 63u)] = (pa[((((k2 * 4u) + k3)) & 63u)] + (pc[((k2) & 63u)] * pd[((k3) & 63u)]));
    pb[((((k2 * 4u) + k3)) & 63u)] = (pb[((((k2 * 4u) + k3)) & 63u)] + (pe[((k2) & 63u)] * pf[((k3) & 63u)]));
    }
    }
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + lid.y) + (16u * k)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + lid.y) + (16u * k))) < cfg.d))) {
    i5 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + lid.x) + (16u * k2)) < cfg.d)) {
    if ((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows && (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + (pa[((((k * 4u) + k2)) & 63u)] + pb[((((k * 4u) + k2)) & 63u)]); }
    }
    }
    }
    }
}
