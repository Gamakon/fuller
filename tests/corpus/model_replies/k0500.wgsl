// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
fn lit_u32(x: u32) -> u32 { return x; }
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { var t = v; return t[i & 3u]; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 8u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    if ((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows)) {
    if (((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d)) {
    f1 = phi[((((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 2047u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 2047u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 256u) + (lid.x * 4u))) & 2047u)], wg_a[(((((k * 256u) + (lid.x * 4u)) + 1u)) & 2047u)]), wg_a[(((((k * 256u) + (lid.x * 4u)) + 2u)) & 2047u)]), wg_a[(((((k * 256u) + (lid.x * 4u)) + 3u)) & 2047u)]);
    v1 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(((((k * 256u) + 64u) + (lid.x * 4u))) & 2047u)], wg_a[((((((k * 256u) + 64u) + (lid.x * 4u)) + 1u)) & 2047u)]), wg_a[((((((k * 256u) + 64u) + (lid.x * 4u)) + 2u)) & 2047u)]), wg_a[((((((k * 256u) + 64u) + (lid.x * 4u)) + 3u)) & 2047u)]);
    v2 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(((((k * 256u) + 128u) + (lid.x * 4u))) & 2047u)], wg_a[((((((k * 256u) + 128u) + (lid.x * 4u)) + 1u)) & 2047u)]), wg_a[((((((k * 256u) + 128u) + (lid.x * 4u)) + 2u)) & 2047u)]), wg_a[((((((k * 256u) + 128u) + (lid.x * 4u)) + 3u)) & 2047u)]);
    v3 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(((((k * 256u) + 192u) + (lid.x * 4u))) & 2047u)], wg_a[((((((k * 256u) + 192u) + (lid.x * 4u)) + 1u)) & 2047u)]), wg_a[((((((k * 256u) + 192u) + (lid.x * 4u)) + 2u)) & 2047u)]), wg_a[((((((k * 256u) + 192u) + (lid.x * 4u)) + 3u)) & 2047u)]);
    v4 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[((((k * 256u) + (lid.y * 4u))) & 2047u)], wg_b[(((((k * 256u) + (lid.y * 4u)) + 1u)) & 2047u)]), wg_b[(((((k * 256u) + (lid.y * 4u)) + 2u)) & 2047u)]), wg_b[(((((k * 256u) + (lid.y * 4u)) + 3u)) & 2047u)]);
    v5 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 256u) + 64u) + (lid.y * 4u))) & 2047u)], wg_b[((((((k * 256u) + 64u) + (lid.y * 4u)) + 1u)) & 2047u)]), wg_b[((((((k * 256u) + 64u) + (lid.y * 4u)) + 2u)) & 2047u)]), wg_b[((((((k * 256u) + 64u) + (lid.y * 4u)) + 3u)) & 2047u)]);
    v6 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 256u) + 128u) + (lid.y * 4u))) & 2047u)], wg_b[((((((k * 256u) + 128u) + (lid.y * 4u)) + 1u)) & 2047u)]), wg_b[((((((k * 256u) + 128u) + (lid.y * 4u)) + 2u)) & 2047u)]), wg_b[((((((k * 256u) + 128u) + (lid.y * 4u)) + 3u)) & 2047u)]);
    v7 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 256u) + 192u) + (lid.y * 4u))) & 2047u)], wg_b[((((((k * 256u) + 192u) + (lid.y * 4u)) + 1u)) & 2047u)]), wg_b[((((((k * 256u) + 192u) + (lid.y * 4u)) + 2u)) & 2047u)]), wg_b[((((((k * 256u) + 192u) + (lid.y * 4u)) + 3u)) & 2047u)]);
    if ((k == 0u)) {
    v8 = (v8 + (((v0 * vec4<f32>(v4[(0u) & 3u])) + (v1 * vec4<f32>(v5[(0u) & 3u]))) + ((v2 * vec4<f32>(v6[(0u) & 3u])) + (v3 * vec4<f32>(v7[(0u) & 3u])))));
    v9 = (v9 + (((v0 * vec4<f32>(v4[(1u) & 3u])) + (v1 * vec4<f32>(v5[(1u) & 3u]))) + ((v2 * vec4<f32>(v6[(1u) & 3u])) + (v3 * vec4<f32>(v7[(1u) & 3u])))));
    v10 = (v10 + (((v0 * vec4<f32>(v4[(2u) & 3u])) + (v1 * vec4<f32>(v5[(2u) & 3u]))) + ((v2 * vec4<f32>(v6[(2u) & 3u])) + (v3 * vec4<f32>(v7[(2u) & 3u])))));
    v11 = (v11 + (((v0 * vec4<f32>(v4[(3u) & 3u])) + (v1 * vec4<f32>(v5[(3u) & 3u]))) + ((v2 * vec4<f32>(v6[(3u) & 3u])) + (v3 * vec4<f32>(v7[(3u) & 3u])))));
    } else {
    v12 = (v12 + (((v0 * vec4<f32>(v4[(0u) & 3u])) + (v1 * vec4<f32>(v5[(0u) & 3u]))) + ((v2 * vec4<f32>(v6[(0u) & 3u])) + (v3 * vec4<f32>(v7[(0u) & 3u])))));
    v13 = (v13 + (((v0 * vec4<f32>(v4[(1u) & 3u])) + (v1 * vec4<f32>(v5[(1u) & 3u]))) + ((v2 * vec4<f32>(v6[(1u) & 3u])) + (v3 * vec4<f32>(v7[(1u) & 3u])))));
    v14 = (v14 + (((v0 * vec4<f32>(v4[(2u) & 3u])) + (v1 * vec4<f32>(v5[(2u) & 3u]))) + ((v2 * vec4<f32>(v6[(2u) & 3u])) + (v3 * vec4<f32>(v7[(2u) & 3u])))));
    v15 = (v15 + (((v0 * vec4<f32>(v4[(3u) & 3u])) + (v1 * vec4<f32>(v5[(3u) & 3u]))) + ((v2 * vec4<f32>(v6[(3u) & 3u])) + (v3 * vec4<f32>(v7[(3u) & 3u])))));
    }
    }
    workgroupBarrier();
    }
    pv[((lit_u32(0u)) & 15u)] = (v8 + v12);
    pv[((lit_u32(1u)) & 15u)] = (v9 + v13);
    pv[((lit_u32(2u)) & 15u)] = (v10 + v14);
    pv[((lit_u32(3u)) & 15u)] = (v11 + v15);
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((gid.y * 4u) + k) < cfg.r_rows)) {
    if (((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d)) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    if ((((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) && (((gid.x * 4u) + min(k2, 3u)) < cfg.d)) && ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d))) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + at_vec4f(pv[((k) & 15u)], ((k2) & (3u))); }
    }
    }
    }
    }
    }
    }
}
