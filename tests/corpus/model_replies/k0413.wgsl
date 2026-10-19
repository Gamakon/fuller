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
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
    i0 = 0u;
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    if (((t + ((lidx + (k * 256u)) / 16u)) < cfg.m)) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2)) < cfg.d))) {
    v0[(k2) & 3u] = phi[((((t + ((lidx + (k * 256u)) / 16u)) * cfg.d) + cfg.r_off) + (((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2))];
    }
    if (((((wid.x * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2) < cfg.d)) {
    v1[(k2) & 3u] = phi[(((t + ((lidx + (k * 256u)) / 16u)) * cfg.d) + (((wid.x * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2))];
    }
    }
    }
    wg_va[(((lidx + (k * 256u))) & 511u)] = v0;
    wg_vb[(((lidx + (k * 256u))) & 511u)] = v1;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[((k) & 15u)] = vec4<f32>(0.0f);
    }
    i0 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
    pv[0u] = (pv[0u] + (vec4<f32>(wg_va[((((k * 16u) + lid.y)) & 511u)].x) * wg_vb[((((k * 16u) + lid.x)) & 511u)]));
    pv[1u] = (pv[1u] + (vec4<f32>(wg_va[((((k * 16u) + lid.y)) & 511u)].y) * wg_vb[((((k * 16u) + lid.x)) & 511u)]));
    pv[2u] = (pv[2u] + (vec4<f32>(wg_va[((((k * 16u) + lid.y)) & 511u)].z) * wg_vb[((((k * 16u) + lid.x)) & 511u)]));
    pv[3u] = (pv[3u] + (vec4<f32>(wg_va[((((k * 16u) + lid.y)) & 511u)].w) * wg_vb[((((k * 16u) + lid.x)) & 511u)]));
    }
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[(((k + 4u)) & 15u)] = (pv[(((k + 4u)) & 15u)] + pv[((k) & 15u)]);
    }
    i3 = (i3 + 1u);
    if ((i3 == 16u)) {
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[(((k + 8u)) & 15u)] = (pv[(((k + 8u)) & 15u)] + pv[(((k + 4u)) & 15u)]);
    pv[(((k + 4u)) & 15u)] = vec4<f32>(0.0f);
    }
    i3 = 0u;
    }
    workgroupBarrier();
    }
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    pv[(((k + 8u)) & 15u)] = (pv[(((k + 8u)) & 15u)] + pv[(((k + 4u)) & 15u)]);
    }
    i0 = 0u;
    for (k = 0u; k < 4u; k = k + 1u) {
    i2 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[(((k + 8u)) & 15u)][(k2) & 3u]; }
    }
    }
}
