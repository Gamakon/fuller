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
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pw: array<vec4<f32>, 16>; var px: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    i0 = 0u;
    for (k = 0u; k < 8u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m)) {
    if ((((wid.x * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.d)) {
    f0 = phi[((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + (wid.x * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    if (((((wid.y * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u))) < cfg.d))) {
    f1 = phi[(((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    }
    wg_a[(((lidx + (256u * k))) & 2047u)] = f0;
    wg_b[(((lidx + (256u * k))) & 2047u)] = f1;
    }
    workgroupBarrier();
    pw[0u] = vec4<f32>(0.0f);
    pw[1u] = vec4<f32>(0.0f);
    pw[2u] = vec4<f32>(0.0f);
    pw[3u] = vec4<f32>(0.0f);
    i1 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pw[0u] = (pw[0u] + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]) * vec4<f32>(wg_b[((((k * 64u) + lid.y)) & 2047u)])));
    pw[1u] = (pw[1u] + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]) * vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 16u)) & 2047u)])));
    pw[2u] = (pw[2u] + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]) * vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 32u)) & 2047u)])));
    pw[3u] = (pw[3u] + (vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + lid.x)) & 2047u)], wg_a[(((((k * 64u) + lid.x) + 16u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 32u)) & 2047u)]), wg_a[(((((k * 64u) + lid.x) + 48u)) & 2047u)]) * vec4<f32>(wg_b[(((((k * 64u) + lid.y) + 48u)) & 2047u)])));
    }
    px[0u] = (px[0u] + pw[0u]);
    px[1u] = (px[1u] + pw[1u]);
    px[2u] = (px[2u] + pw[2u]);
    px[3u] = (px[3u] + pw[3u]);
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((((wid.y * 64u) + lid.y) + (16u * k)) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + lid.y) + (16u * k))) < cfg.d))) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + lid.x) + (16u * k2)) < cfg.d)) {
    if ((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) < cfg.r_rows && (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)) < cfg.d && (cfg.r_off + (((wid.y * 64u + lid.y) + min(k, 3u) * 16u))) < cfg.d) { h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] = h[((((wid.y * 64u + lid.y) + min(k, 3u) * 16u)) * cfg.d + (((wid.x * 64u + lid.x) + min(k2, 3u) * 16u)))] + px[((k) & 15u)][(k2) & 3u]; }
    }
    }
    }
    }
}
