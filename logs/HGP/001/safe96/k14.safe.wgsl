// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_va: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    v0 = vec4<f32>(0.0f);
    if (((t + (((lid.y * 16u) + lid.x) / 16u)) < cfg.m)) {
    if ((((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u)) < cfg.d)) {
    v0[(0u) & 3u] = phi[(((t + (((lid.y * 16u) + lid.x) / 16u)) * cfg.d) + ((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u)))];
    }
    if (((((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u)) + 1u) < cfg.d)) {
    v0[(1u) & 3u] = phi[((((t + (((lid.y * 16u) + lid.x) / 16u)) * cfg.d) + ((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u))) + 1u)];
    }
    if (((((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u)) + 2u) < cfg.d)) {
    v0[(2u) & 3u] = phi[((((t + (((lid.y * 16u) + lid.x) / 16u)) * cfg.d) + ((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u))) + 2u)];
    }
    if (((((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u)) + 3u) < cfg.d)) {
    v0[(3u) & 3u] = phi[((((t + (((lid.y * 16u) + lid.x) / 16u)) * cfg.d) + ((wid.x * 64u) + ((((lid.y * 16u) + lid.x) % 16u) * 4u))) + 3u)];
    }
    }
    wg_va[((((lid.y * 16u) + lid.x)) & 255u)] = v0;
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    if (((((t + ((((lid.y * 16u) + lid.x) + (k * 256u)) / 64u)) < cfg.m) & (((wid.y * 64u) + ((((lid.y * 16u) + lid.x) + (k * 256u)) % 64u)) < cfg.r_rows)) & ((cfg.r_off + ((wid.y * 64u) + ((((lid.y * 16u) + lid.x) + (k * 256u)) % 64u))) < cfg.d))) {
    f0 = phi[(((t + ((((lid.y * 16u) + lid.x) + (k * 256u)) / 64u)) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + ((((lid.y * 16u) + lid.x) + (k * 256u)) % 64u))))];
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 1023u)] = f0;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    v1 = wg_va[((((k * 16u) + lid.x)) & 255u)];
    v2 = (v2 + (vec4<f32>(wg_a[((((k * 64u) + (lid.y * 4u))) & 1023u)]) * v1));
    v3 = (v3 + (vec4<f32>(wg_a[(((((k * 64u) + (lid.y * 4u)) + 1u)) & 1023u)]) * v1));
    v4 = (v4 + (vec4<f32>(wg_a[(((((k * 64u) + (lid.y * 4u)) + 2u)) & 1023u)]) * v1));
    v5 = (v5 + (vec4<f32>(wg_a[(((((k * 64u) + (lid.y * 4u)) + 3u)) & 1023u)]) * v1));
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows)) {
    v6 = v2;
    if ((k == 1u)) {
    v6 = v3;
    }
    if ((k == 2u)) {
    v6 = v4;
    }
    if ((k == 3u)) {
    v6 = v5;
    }
    if (((gid.x * 4u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & (((gid.x * 4u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + ((gid.x * 4u)))] + v6[(0u) & 3u]; }
    }
    if ((((gid.x * 4u) + 1u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 1u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 1u)))] + v6[(1u) & 3u]; }
    }
    if ((((gid.x * 4u) + 2u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 2u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 2u)))] + v6[(2u) & 3u]; }
    }
    if ((((gid.x * 4u) + 3u) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + 3u)) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + 3u)))] + v6[(3u) & 3u]; }
    }
    }
    }
}
