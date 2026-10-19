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
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0); 
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); 
    var i4: u32 = 0u; var i5: u32 = 0u; var i6: u32 = 0u; var i7: u32 = 0u; var i8: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    i4 = (lidx + (256u * k));
    i5 = (t + ((i4) >> ((4u) & 31u)));
    i6 = ((wid.x * 64u) + (((i4) & (15u)) * 4u));
    i7 = ((wid.y * 64u) + (((i4) & (15u)) * 4u));
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if ((i5 < cfg.m)) {
    i8 = (i5 * cfg.d);
    if ((i6 < cfg.d)) {
    f0 = phi[(i8 + i6)];
    }
    if (((i6 + 1u) < cfg.d)) {
    f1 = phi[((i8 + i6) + 1u)];
    }
    if (((i6 + 2u) < cfg.d)) {
    f2 = phi[((i8 + i6) + 2u)];
    }
    if (((i6 + 3u) < cfg.d)) {
    f3 = phi[((i8 + i6) + 3u)];
    }
    if (((i7 < cfg.r_rows) && ((cfg.r_off + i7) < cfg.d))) {
    f4 = phi[((i8 + cfg.r_off) + i7)];
    }
    if ((((i7 + 1u) < cfg.r_rows) && (((cfg.r_off + i7) + 1u) < cfg.d))) {
    f5 = phi[(((i8 + cfg.r_off) + i7) + 1u)];
    }
    if ((((i7 + 2u) < cfg.r_rows) && (((cfg.r_off + i7) + 2u) < cfg.d))) {
    f6 = phi[(((i8 + cfg.r_off) + i7) + 2u)];
    }
    if ((((i7 + 3u) < cfg.r_rows) && (((cfg.r_off + i7) + 3u) < cfg.d))) {
    f7 = phi[(((i8 + cfg.r_off) + i7) + 3u)];
    }
    }
    wg_va[((i4) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[((i4) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    for (k = 0u; k < 32u; k = k + 1u) {
    v4 = wg_va[((((k * 16u) + lid.x)) & 511u)];
    v5 = wg_vb[((((k * 16u) + lid.y)) & 511u)];
    v0 = (v0 + (vec4<f32>(v5[(0u) & 3u]) * v4));
    v1 = (v1 + (vec4<f32>(v5[(1u) & 3u]) * v4));
    v2 = (v2 + (vec4<f32>(v5[(2u) & 3u]) * v4));
    v3 = (v3 + (vec4<f32>(v5[(3u) & 3u]) * v4));
    }
    v6 = (v6 + v0);
    v7 = (v7 + v1);
    v8 = (v8 + v2);
    v9 = (v9 + v3);
    if ((((((t) >> ((5u) & 31u))) & (15u)) == 15u)) {
    v10 = (v10 + v6);
    v11 = (v11 + v7);
    v12 = (v12 + v8);
    v13 = (v13 + v9);
    v6 = vec4<f32>(0.0f);
    v7 = vec4<f32>(0.0f);
    v8 = vec4<f32>(0.0f);
    v9 = vec4<f32>(0.0f);
    }
    workgroupBarrier();
    }
    pv[0u] = (v10 + v6);
    pv[1u] = (v11 + v7);
    pv[2u] = (v12 + v8);
    pv[3u] = (v13 + v9);
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    h[(((gid.y * 4u) + k) * cfg.d + ((gid.x * 4u) + k2))] = h[(((gid.y * 4u) + k) * cfg.d + ((gid.x * 4u) + k2))] + pv[((k) & 15u)][(k2) & 3u];
    }
    }
    }
    }
}
