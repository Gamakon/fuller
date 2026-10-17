// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    if (((t + lid.y) < cfg.m)) {
    if ((((gid.x * 4u) + 3u) < cfg.d)) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(phi[(((t + lid.y) * cfg.d) + (gid.x * 4u))], phi[((((t + lid.y) * cfg.d) + (gid.x * 4u)) + 1u)]), phi[((((t + lid.y) * cfg.d) + (gid.x * 4u)) + 2u)]), phi[((((t + lid.y) * cfg.d) + (gid.x * 4u)) + 3u)]);
    } else {
    if (((gid.x * 4u) < cfg.d)) {
    v0[(0u) & 3u] = phi[(((t + lid.y) * cfg.d) + (gid.x * 4u))];
    }
    if ((((gid.x * 4u) + 1u) < cfg.d)) {
    v0[(1u) & 3u] = phi[((((t + lid.y) * cfg.d) + (gid.x * 4u)) + 1u)];
    }
    if ((((gid.x * 4u) + 2u) < cfg.d)) {
    v0[(2u) & 3u] = phi[((((t + lid.y) * cfg.d) + (gid.x * 4u)) + 2u)];
    }
    }
    if (((((wid.y * 64u) + (lid.x * 4u)) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + (lid.x * 4u))) < cfg.d))) {
    v1[(0u) & 3u] = phi[(((t + lid.y) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (lid.x * 4u))))];
    }
    if ((((((wid.y * 64u) + (lid.x * 4u)) + 1u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.x * 4u))) + 1u) < cfg.d))) {
    v1[(1u) & 3u] = phi[((((t + lid.y) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (lid.x * 4u)))) + 1u)];
    }
    if ((((((wid.y * 64u) + (lid.x * 4u)) + 2u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.x * 4u))) + 2u) < cfg.d))) {
    v1[(2u) & 3u] = phi[((((t + lid.y) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (lid.x * 4u)))) + 2u)];
    }
    if ((((((wid.y * 64u) + (lid.x * 4u)) + 3u) < cfg.r_rows) && (((cfg.r_off + ((wid.y * 64u) + (lid.x * 4u))) + 3u) < cfg.d))) {
    v1[(3u) & 3u] = phi[((((t + lid.y) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (lid.x * 4u)))) + 3u)];
    }
    }
    wg_va[((((lid.y * 16u) + lid.x)) & 255u)] = v0;
    wg_vb[((((lid.y * 16u) + lid.x)) & 255u)] = v1;
    workgroupBarrier();
    for (k = 0u; k < 16u; k = k + 1u) {
    v2 = wg_va[((((k * 16u) + lid.x)) & 255u)];
    v3 = wg_vb[((((k * 16u) + lid.y)) & 255u)];
    v4 = (v4 + (vec4<f32>(v3.x) * v2));
    v5 = (v5 + (vec4<f32>(v3.y) * v2));
    v6 = (v6 + (vec4<f32>(v3.z) * v2));
    v7 = (v7 + (vec4<f32>(v3.w) * v2));
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) && ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d))) {
    v8 = v4;
    if ((k == 1u)) {
    v8 = v5;
    }
    if ((k == 2u)) {
    v8 = v6;
    }
    if ((k == 3u)) {
    v8 = v7;
    }
    if (((gid.x * 4u) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && ((gid.x * 4u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] + v8[(0u) & 3u]; }
    }
    if ((((gid.x * 4u) + 1u) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + 1u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] + v8[(1u) & 3u]; }
    }
    if ((((gid.x * 4u) + 2u) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + 2u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] + v8[(2u) & 3u]; }
    }
    if ((((gid.x * 4u) + 3u) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + 3u)) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] + v8[(3u) & 3u]; }
    }
    }
    }
}
