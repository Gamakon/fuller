// grid 32 8 2 8
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
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m)) {
    f0 = phi[(((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + min(((wid.x * 64u) + (((lidx + (256u * k))) & (63u))), (cfg.d - 1u)))];
    f1 = phi[(((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + min(((cfg.r_off + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u))), (cfg.d - 1u)))];
    }
    wg_a[(((lidx + (256u * k))) & 2047u)] = f0;
    wg_b[(((lidx + (256u * k))) & 2047u)] = f1;
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    for (k = 0u; k < 32u; k = k + 1u) {
    v0 = (v0 + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[((((k * 64u) + lid.x)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_b[((((k * 64u) + (lid.y * 8u))) & 2047u)], wg_b[((((k * 64u) + (lid.y * 8u))) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 1u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 1u)) & 2047u)])));
    v1 = (v1 + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[((((k * 64u) + lid.x)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 64u) + (lid.y * 8u)) + 2u)) & 2047u)], wg_b[(((((k * 64u) + (lid.y * 8u)) + 2u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 3u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 3u)) & 2047u)])));
    v2 = (v2 + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[((((k * 64u) + lid.x)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 64u) + (lid.y * 8u)) + 4u)) & 2047u)], wg_b[(((((k * 64u) + (lid.y * 8u)) + 4u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 5u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 5u)) & 2047u)])));
    v3 = (v3 + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[((((k * 64u) + lid.x)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(((((k * 64u) + (lid.y * 8u)) + 6u)) & 2047u)], wg_b[(((((k * 64u) + (lid.y * 8u)) + 6u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 7u)) & 2047u)]), wg_b[(((((k * 64u) + (lid.y * 8u)) + 7u)) & 2047u)])));
    }
    if ((((((t) >> ((5u) & 31u))) & (1u)) == 0u)) {
    v4 = (v4 + v0);
    v5 = (v5 + v1);
    v6 = (v6 + v2);
    v7 = (v7 + v3);
    } else {
    v8 = (v8 + v0);
    v9 = (v9 + v1);
    v10 = (v10 + v2);
    v11 = (v11 + v3);
    }
    workgroupBarrier();
    }
    pv[((lit_u32(0u)) & 15u)] = (v4 + v8);
    pv[((lit_u32(1u)) & 15u)] = (v5 + v9);
    pv[((lit_u32(2u)) & 15u)] = (v6 + v10);
    pv[((lit_u32(3u)) & 15u)] = (v7 + v11);
    for (k = 0u; k < 8u; k = k + 1u) {
    if ((((((wid.y * 64u) + (lid.y * 8u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 8u)) + k)) < cfg.d))) {
    if ((((wid.x * 64u) + lid.x) < cfg.d)) {
    if ((((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) && (((wid.x * 64u) + lid.x) < cfg.d)) && ((cfg.r_off + ((gid.y * 8u) + min(k, 7u))) < cfg.d))) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && ((wid.x * 64u + lid.x)) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + ((wid.x * 64u + lid.x)))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + ((wid.x * 64u + lid.x)))] + at_vec4f(pv[((((k) >> ((1u) & 31u))) & 15u)], (((k) & (1u)) * 2u)); }
    }
    }
    if (((((wid.x * 64u) + lid.x) + 32u) < cfg.d)) {
    if ((((((gid.y * 8u) + min(k, 7u)) < cfg.r_rows) && ((((wid.x * 64u) + lid.x) + 32u) < cfg.d)) && ((cfg.r_off + ((gid.y * 8u) + min(k, 7u))) < cfg.d))) {
    if ((((gid.y * 8u) + min(k, 7u))) < cfg.r_rows && (((wid.x * 64u + lid.x) + 32u)) < cfg.d && (cfg.r_off + (((gid.y * 8u) + min(k, 7u)))) < cfg.d) { h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((wid.x * 64u + lid.x) + 32u)))] = h[((((gid.y * 8u) + min(k, 7u))) * cfg.d + (((wid.x * 64u + lid.x) + 32u)))] + at_vec4f(pv[((((k) >> ((1u) & 31u))) & 15u)], ((((k) & (1u)) * 2u) + 1u)); }
    }
    }
    }
    }
}
