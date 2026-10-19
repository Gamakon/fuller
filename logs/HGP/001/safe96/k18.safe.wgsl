// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 8u) {
    i0 = 0u;
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) < cfg.m)) {
    if (((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows) & ((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d))) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))))];
    }
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    wg_a[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 511u)] = f0;
    wg_b[(((((lid.y * 16u) + lid.x) + (k * 256u))) & 511u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 8u; k = k + 1u) {
    v0 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[((((k * 64u) + (lid.y * 4u))) & 511u)], wg_a[(((((k * 64u) + (lid.y * 4u)) + 1u)) & 511u)]), wg_a[(((((k * 64u) + (lid.y * 4u)) + 2u)) & 511u)]), wg_a[(((((k * 64u) + (lid.y * 4u)) + 3u)) & 511u)]);
    v1 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[((((k * 64u) + (lid.x * 4u))) & 511u)], wg_b[(((((k * 64u) + (lid.x * 4u)) + 1u)) & 511u)]), wg_b[(((((k * 64u) + (lid.x * 4u)) + 2u)) & 511u)]), wg_b[(((((k * 64u) + (lid.x * 4u)) + 3u)) & 511u)]);
    v2 = (v2 + (vec4<f32>(v0.x) * v1));
    v3 = (v3 + (vec4<f32>(v0.y) * v1));
    v4 = (v4 + (vec4<f32>(v0.z) * v1));
    v5 = (v5 + (vec4<f32>(v0.w) * v1));
    }
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows) & ((cfg.r_off + ((gid.y * 4u) + min(k, 3u))) < cfg.d))) {
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
    i1 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + min(k2, 3u)) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + v6[(k2) & 3u]; }
    }
    }
    }
    }
}
