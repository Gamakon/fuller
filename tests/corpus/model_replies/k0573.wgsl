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
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>; var pw: array<vec4<f32>, 16>;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    pw[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.d)) {
    f0 = phi[(((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((lidx + (256u * k))) & (63u))))];
    }
    if (((((wid.y * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * 64u) + (((lidx + (256u * k))) & (63u)))) < cfg.d))) {
    f1 = phi[((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + (((lidx + (256u * k))) & (63u))))];
    }
    }
    wg_a[(((lidx + (256u * k))) & 2047u)] = f0;
    wg_b[(((lidx + (256u * k))) & 2047u)] = f1;
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    i1 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    v0 = (v0 + (vec4<f32>(wg_b[((((k * 64u) + lid.y)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)])));
    v1 = (v1 + (vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)])));
    v2 = (v2 + (vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)])));
    v3 = (v3 + (vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)]) * vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)])));
    }
    if (((((t) >> ((5u) & 31u)) & (1u)) == 0u)) {
    pv[0] = pv[0] + v0;
    pv[1] = pv[1] + v1;
    pv[2] = pv[2] + v2;
    pv[3] = pv[3] + v3;
    } else {
    pw[0] = pw[0] + v0;
    pw[1] = pw[1] + v1;
    pw[2] = pw[2] + v2;
    pw[3] = pw[3] + v3;
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + lid.y) + (16u * k)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + lid.y) + (16u * k))) < cfg.d))) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + lid.x) + (16u * k2)) < cfg.d)) {
    if ((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows && (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + at_vec4f((pv[((k) & 15u)] + pw[((k) & 15u)]), k2); }
    }
    }
    }
    }
}
