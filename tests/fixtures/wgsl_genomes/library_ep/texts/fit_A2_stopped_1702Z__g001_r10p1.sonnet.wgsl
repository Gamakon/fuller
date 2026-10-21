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
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var k3: u32 = 0u; var k4: u32 = 0u;
    i0 = 0u;
    for (t = 0u; t < cfg.m; t = t + 128u) {
    i1 = 0u;
    i2 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    i3 = 0u;
    i4 = 0u;
    for (k2 = 0u; k2 < 2u; k2 = k2 + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    f2 = 0.0f;
    f3 = 0.0f;
    f4 = 0.0f;
    f5 = 0.0f;
    f6 = 0.0f;
    f7 = 0.0f;
    if ((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) < cfg.d)) {
    f0 = phi[(((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + (wid.x * 64u)) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))];
    }
    if (((((wid.x * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 1u) < cfg.d)) {
    f1 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + (wid.x * 64u)) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 1u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 2u) < cfg.d)) {
    f2 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + (wid.x * 64u)) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 2u)];
    }
    if (((((wid.x * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 3u) < cfg.d)) {
    f3 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + (wid.x * 64u)) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 3u)];
    }
    if (((((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) < cfg.d))) {
    f4 = phi[(((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)))];
    }
    if ((((((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 1u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 1u) < cfg.d))) {
    f5 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 1u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 2u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 2u) < cfg.d))) {
    f6 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 2u)];
    }
    if ((((((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u)) + 3u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 3u) < cfg.d))) {
    f7 = phi[((((((t + (k * 32u)) + (((lidx + (k2 * 256u))) >> ((4u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + ((((lidx + (k2 * 256u))) & (15u)) * 4u))) + 3u)];
    }
    }
    wg_va[(((lidx + (k2 * 256u))) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f0, f1), f2), f3);
    wg_vb[(((lidx + (k2 * 256u))) & 511u)] = vec4<f32>(vec3<f32>(vec2<f32>(f4, f5), f6), f7);
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    i5 = 0u;
    i4 = 0u;
    for (k2 = 0u; k2 < 16u; k2 = k2 + 1u) {
    v0 = ((v0 + (wg_va[((((k2 * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[((((k2 * 16u) + lid.y)) & 511u)].x))) + (wg_va[(((((k2 + 16u) * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[(((((k2 + 16u) * 16u) + lid.y)) & 511u)].x)));
    v1 = ((v1 + (wg_va[((((k2 * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[((((k2 * 16u) + lid.y)) & 511u)].y))) + (wg_va[(((((k2 + 16u) * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[(((((k2 + 16u) * 16u) + lid.y)) & 511u)].y)));
    v2 = ((v2 + (wg_va[((((k2 * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[((((k2 * 16u) + lid.y)) & 511u)].z))) + (wg_va[(((((k2 + 16u) * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[(((((k2 + 16u) * 16u) + lid.y)) & 511u)].z)));
    v3 = ((v3 + (wg_va[((((k2 * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[((((k2 * 16u) + lid.y)) & 511u)].w))) + (wg_va[(((((k2 + 16u) * 16u) + lid.x)) & 511u)] * vec4<f32>(wg_vb[(((((k2 + 16u) * 16u) + lid.y)) & 511u)].w)));
    }
    v4 = (v4 + v0);
    v5 = (v5 + v1);
    v6 = (v6 + v2);
    v7 = (v7 + v3);
    workgroupBarrier();
    }
    }
    if (((((wid.y * 64u) + (lid.y * 4u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + (lid.y * 4u))) < cfg.d))) {
    if ((((wid.x * 64u) + (lid.x * 4u)) < cfg.d)) {
    if (((gid.y * 4u)) < cfg.r_rows && ((gid.x * 4u)) < cfg.d && (cfg.r_off + ((gid.y * 4u))) < cfg.d) { h[(((gid.y * 4u)) * cfg.d + ((gid.x * 4u)))] = h[(((gid.y * 4u)) * cfg.d + ((gid.x * 4u)))] + v4[(0u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 1u) < cfg.d)) {
    if (((gid.y * 4u)) < cfg.r_rows && (((gid.x * 4u) + 1u)) < cfg.d && (cfg.r_off + ((gid.y * 4u))) < cfg.d) { h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 1u)))] = h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 1u)))] + v4[(1u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 2u) < cfg.d)) {
    if (((gid.y * 4u)) < cfg.r_rows && (((gid.x * 4u) + 2u)) < cfg.d && (cfg.r_off + ((gid.y * 4u))) < cfg.d) { h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 2u)))] = h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 2u)))] + v4[(2u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 3u) < cfg.d)) {
    if (((gid.y * 4u)) < cfg.r_rows && (((gid.x * 4u) + 3u)) < cfg.d && (cfg.r_off + ((gid.y * 4u))) < cfg.d) { h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 3u)))] = h[(((gid.y * 4u)) * cfg.d + (((gid.x * 4u) + 3u)))] + v4[(3u) & 3u]; }
    }
    }
    if ((((((wid.y * 64u) + (lid.y * 4u)) + 1u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.y * 4u))) + 1u) < cfg.d))) {
    if ((((wid.x * 64u) + (lid.x * 4u)) < cfg.d)) {
    if ((((gid.y * 4u) + 1u)) < cfg.r_rows && ((gid.x * 4u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 1u))) < cfg.d) { h[((((gid.y * 4u) + 1u)) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + 1u)) * cfg.d + ((gid.x * 4u)))] + v5[(0u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 1u) < cfg.d)) {
    if ((((gid.y * 4u) + 1u)) < cfg.r_rows && (((gid.x * 4u) + 1u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 1u))) < cfg.d) { h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 1u)))] + v5[(1u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 2u) < cfg.d)) {
    if ((((gid.y * 4u) + 1u)) < cfg.r_rows && (((gid.x * 4u) + 2u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 1u))) < cfg.d) { h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 2u)))] + v5[(2u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 3u) < cfg.d)) {
    if ((((gid.y * 4u) + 1u)) < cfg.r_rows && (((gid.x * 4u) + 3u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 1u))) < cfg.d) { h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + 1u)) * cfg.d + (((gid.x * 4u) + 3u)))] + v5[(3u) & 3u]; }
    }
    }
    if ((((((wid.y * 64u) + (lid.y * 4u)) + 2u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.y * 4u))) + 2u) < cfg.d))) {
    if ((((wid.x * 64u) + (lid.x * 4u)) < cfg.d)) {
    if ((((gid.y * 4u) + 2u)) < cfg.r_rows && ((gid.x * 4u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 2u))) < cfg.d) { h[((((gid.y * 4u) + 2u)) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + 2u)) * cfg.d + ((gid.x * 4u)))] + v6[(0u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 1u) < cfg.d)) {
    if ((((gid.y * 4u) + 2u)) < cfg.r_rows && (((gid.x * 4u) + 1u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 2u))) < cfg.d) { h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 1u)))] + v6[(1u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 2u) < cfg.d)) {
    if ((((gid.y * 4u) + 2u)) < cfg.r_rows && (((gid.x * 4u) + 2u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 2u))) < cfg.d) { h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 2u)))] + v6[(2u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 3u) < cfg.d)) {
    if ((((gid.y * 4u) + 2u)) < cfg.r_rows && (((gid.x * 4u) + 3u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 2u))) < cfg.d) { h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + 2u)) * cfg.d + (((gid.x * 4u) + 3u)))] + v6[(3u) & 3u]; }
    }
    }
    if ((((((wid.y * 64u) + (lid.y * 4u)) + 3u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.y * 4u))) + 3u) < cfg.d))) {
    if ((((wid.x * 64u) + (lid.x * 4u)) < cfg.d)) {
    if ((((gid.y * 4u) + 3u)) < cfg.r_rows && ((gid.x * 4u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 3u))) < cfg.d) { h[((((gid.y * 4u) + 3u)) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + 3u)) * cfg.d + ((gid.x * 4u)))] + v7[(0u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 1u) < cfg.d)) {
    if ((((gid.y * 4u) + 3u)) < cfg.r_rows && (((gid.x * 4u) + 1u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 3u))) < cfg.d) { h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 1u)))] + v7[(1u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 2u) < cfg.d)) {
    if ((((gid.y * 4u) + 3u)) < cfg.r_rows && (((gid.x * 4u) + 2u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 3u))) < cfg.d) { h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 2u)))] + v7[(2u) & 3u]; }
    }
    if (((((wid.x * 64u) + (lid.x * 4u)) + 3u) < cfg.d)) {
    if ((((gid.y * 4u) + 3u)) < cfg.r_rows && (((gid.x * 4u) + 3u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + 3u))) < cfg.d) { h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + 3u)) * cfg.d + (((gid.x * 4u) + 3u)))] + v7[(3u) & 3u]; }
    }
    }
}
