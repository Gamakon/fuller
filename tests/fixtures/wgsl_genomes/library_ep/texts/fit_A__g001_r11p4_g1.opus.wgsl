// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn lit_i32(x: i32) -> i32 { return x; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0; var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0; 
    var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0; var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0; 
    var f16r: f32 = 0.0; var f17: f32 = 0.0; var f18: f32 = 0.0; var f19: f32 = 0.0; var f20: f32 = 0.0; var f21: f32 = 0.0; var f22: f32 = 0.0; var f23: f32 = 0.0; 
    var f24: f32 = 0.0; var f25: f32 = 0.0; var f26: f32 = 0.0; var f27: f32 = 0.0; var f28: f32 = 0.0; var f29: f32 = 0.0; var f30: f32 = 0.0; var f31: f32 = 0.0; 
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); 
    var i0: u32 = 0u; var i1: u32 = 0u;
    var b0: bool = false; var b1: bool = false; var b2: bool = false; var b3: bool = false; var b4: bool = false; var b5: bool = false; var b6: bool = false; var b7: bool = false; 
    var t: u32 = 0u; var k: u32 = 0u;
    var pa: array<f32, 64>; var pb: array<f32, 64>; var pc: array<f32, 64>; var pd: array<f32, 64>; var pe: array<f32, 64>; var pf: array<f32, 64>;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>; var pi: array<u32, 16>; var pj: array<u32, 16>; var pk: array<u32, 16>; var pm: array<bool, 16>; var pn: array<bool, 16>;
    f0 = lit_f32(0.0f);
    f1 = lit_f32(0.0f);
    f2 = lit_f32(0.0f);
    {
    f3 = lit_f32(0.0f);
    f4 = lit_f32(0.0f);
    f5 = lit_f32(0.0f);
    {
    f6 = lit_f32(0.0f);
    f7 = lit_f32(0.0f);
    i0 = lit_u32(0u);
    {
    for (t = 0u; t < cfg.m; t = t + 8u) {
    f8 = lit_f32(0.0f);
    f9 = lit_f32(0.0f);
    if (((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(32u))) < cfg.m)) {
    if (((((wid.y * lit_u32(32u)) + (((lid.y * lit_u32(16u)) + lid.x) % lit_u32(32u))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * lit_u32(32u)) + (((lid.y * lit_u32(16u)) + lid.x) % lit_u32(32u)))) < cfg.d))) {
    f8 = phi[(((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(32u))) * cfg.d) + (cfg.r_off + ((wid.y * lit_u32(32u)) + (((lid.y * lit_u32(16u)) + lid.x) % lit_u32(32u)))))];
    }
    if ((((wid.x * lit_u32(32u)) + (((lid.y * lit_u32(16u)) + lid.x) % lit_u32(32u))) < cfg.d)) {
    f9 = phi[(((t + (((lid.y * lit_u32(16u)) + lid.x) / lit_u32(32u))) * cfg.d) + ((wid.x * lit_u32(32u)) + (((lid.y * lit_u32(16u)) + lid.x) % lit_u32(32u))))];
    }
    }
    {
    wg_a[((((lid.y * lit_u32(16u)) + lid.x)) & 2047u)] = f8;
    wg_b[((((lid.y * lit_u32(16u)) + lid.x)) & 2047u)] = f9;
    workgroupBarrier();
    {
    i1 = lit_u32(0u);
    for (k = 0u; k < 8u; k = k + 2u) {
    {
    f0 = (f0 + (wg_a[((((k * lit_u32(32u)) + lid.y)) & 2047u)] * wg_b[((((k * lit_u32(32u)) + lid.x)) & 2047u)]));
    f1 = (f1 + (wg_a[((((k * lit_u32(32u)) + lid.y)) & 2047u)] * wg_b[(((((k * lit_u32(32u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    f2 = (f2 + (wg_a[(((((k * lit_u32(32u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[((((k * lit_u32(32u)) + lid.x)) & 2047u)]));
    {
    f3 = (f3 + (wg_a[(((((k * lit_u32(32u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[(((((k * lit_u32(32u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    f4 = (f4 + (wg_a[(((((k * lit_u32(32u)) + lit_u32(32u)) + lid.y)) & 2047u)] * wg_b[(((((k * lit_u32(32u)) + lit_u32(32u)) + lid.x)) & 2047u)]));
    f5 = (f5 + (wg_a[(((((k * lit_u32(32u)) + lit_u32(32u)) + lid.y)) & 2047u)] * wg_b[((((((k * lit_u32(32u)) + lit_u32(32u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    {
    f6 = (f6 + (wg_a[((((((k * lit_u32(32u)) + lit_u32(32u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[(((((k * lit_u32(32u)) + lit_u32(32u)) + lid.x)) & 2047u)]));
    f7 = (f7 + (wg_a[((((((k * lit_u32(32u)) + lit_u32(32u)) + lid.y) + lit_u32(16u))) & 2047u)] * wg_b[((((((k * lit_u32(32u)) + lit_u32(32u)) + lid.x) + lit_u32(16u))) & 2047u)]));
    }
    }
    }
    }
    workgroupBarrier();
    }
    }
    }
    if ((((((wid.y * lit_u32(32u)) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * lit_u32(32u)) + lid.y)) < cfg.d)) && (((wid.x * lit_u32(32u)) + lid.x) < cfg.d))) {
    if (((wid.y * 32u + lid.y)) < cfg.r_rows && ((wid.x * 32u + lid.x)) < cfg.d && (cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d) { h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + ((wid.x * 32u + lid.x)))] + (f0 + f4); }
    }
    if ((((((wid.y * lit_u32(32u)) + lid.y) < cfg.r_rows) && ((cfg.r_off + ((wid.y * lit_u32(32u)) + lid.y)) < cfg.d)) && ((((wid.x * lit_u32(32u)) + lid.x) + lit_u32(16u)) < cfg.d))) {
    if (((wid.y * 32u + lid.y)) < cfg.r_rows && (((wid.x * 32u + lid.x) + 16u)) < cfg.d && (cfg.r_off + ((wid.y * 32u + lid.y))) < cfg.d) { h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[(((wid.y * 32u + lid.y)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + (f1 + f5); }
    }
    {
    if (((((((wid.y * lit_u32(32u)) + lid.y) + lit_u32(16u)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * lit_u32(32u)) + lid.y) + lit_u32(16u))) < cfg.d)) && (((wid.x * lit_u32(32u)) + lid.x) < cfg.d))) {
    if ((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows && ((wid.x * 32u + lid.x)) < cfg.d && (cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + ((wid.x * 32u + lid.x)))] + (f2 + f6); }
    }
    if (((((((wid.y * lit_u32(32u)) + lid.y) + lit_u32(16u)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * lit_u32(32u)) + lid.y) + lit_u32(16u))) < cfg.d)) && ((((wid.x * lit_u32(32u)) + lid.x) + lit_u32(16u)) < cfg.d))) {
    if ((((wid.y * 32u + lid.y) + 16u)) < cfg.r_rows && (((wid.x * 32u + lid.x) + 16u)) < cfg.d && (cfg.r_off + (((wid.y * 32u + lid.y) + 16u))) < cfg.d) { h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] = h[((((wid.y * 32u + lid.y) + 16u)) * cfg.d + (((wid.x * 32u + lid.x) + 16u)))] + (f3 + f7); }
    }
    }
    }
    }
    }
}
