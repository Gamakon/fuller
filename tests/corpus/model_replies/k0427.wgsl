// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var e2: vec4<f32> = vec4<f32>(0.0); var e3: vec4<f32> = vec4<f32>(0.0); var e4: vec4<f32> = vec4<f32>(0.0); var e5: vec4<f32> = vec4<f32>(0.0);
    var o2: vec4<f32> = vec4<f32>(0.0); var o3: vec4<f32> = vec4<f32>(0.0); var o4: vec4<f32> = vec4<f32>(0.0); var o5: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    for (t = 0u; t < (((cfg.m + 31u) / 32u)); t = t + 1u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    if ((((t * 32u) + ((lidx + (k * 256u)) / 16u)) < cfg.m)) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if ((((((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2)) < cfg.d))) {
    f0 = phi[(((((t * 32u) + ((lidx + (k * 256u)) / 16u)) * cfg.d) + cfg.r_off) + (((wid.y * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2))];
    }
    if (((((wid.x * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2) < cfg.d)) {
    f1 = phi[((((t * 32u) + ((lidx + (k * 256u)) / 16u)) * cfg.d) + (((wid.x * 64u) + (((lidx + (k * 256u)) % 16u) * 4u)) + k2))];
    }
    v0[(k2) & 3u] = f0;
    v1[(k2) & 3u] = f1;
    }
    }
    wg_va[(((lidx + (k * 256u))) & 511u)] = v0;
    wg_vb[(((lidx + (k * 256u))) & 511u)] = v1;
    }
    workgroupBarrier();
    e2 = vec4<f32>(0.0); e3 = vec4<f32>(0.0); e4 = vec4<f32>(0.0); e5 = vec4<f32>(0.0);
    o2 = vec4<f32>(0.0); o3 = vec4<f32>(0.0); o4 = vec4<f32>(0.0); o5 = vec4<f32>(0.0);
    for (k = 0u; k < 32u; k = k + 2u) {
    e2 = e2 + vec4<f32>(wg_va[((k * 16u) + lid.y) & 511u].x) * wg_vb[((k * 16u) + lid.x) & 511u];
    e3 = e3 + vec4<f32>(wg_va[((k * 16u) + lid.y) & 511u].y) * wg_vb[((k * 16u) + lid.x) & 511u];
    e4 = e4 + vec4<f32>(wg_va[((k * 16u) + lid.y) & 511u].z) * wg_vb[((k * 16u) + lid.x) & 511u];
    e5 = e5 + vec4<f32>(wg_va[((k * 16u) + lid.y) & 511u].w) * wg_vb[((k * 16u) + lid.x) & 511u];
    o2 = o2 + vec4<f32>(wg_va[((k * 16u) + 16u + lid.y) & 511u].x) * wg_vb[((k * 16u) + 16u + lid.x) & 511u];
    o3 = o3 + vec4<f32>(wg_va[((k * 16u) + 16u + lid.y) & 511u].y) * wg_vb[((k * 16u) + 16u + lid.x) & 511u];
    o4 = o4 + vec4<f32>(wg_va[((k * 16u) + 16u + lid.y) & 511u].z) * wg_vb[((k * 16u) + 16u + lid.x) & 511u];
    o5 = o5 + vec4<f32>(wg_va[((k * 16u) + 16u + lid.y) & 511u].w) * wg_vb[((k * 16u) + 16u + lid.x) & 511u];
    }
    v2 = v2 + (e2 + o2);
    v3 = v3 + (e3 + o3);
    v4 = v4 + (e4 + o4);
    v5 = v5 + (e5 + o5);
    workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
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
    if ((((((wid.y * 64u) + (lid.y * 4u)) + k) < cfg.r_rows) && ((cfg.r_off + (((wid.y * 64u) + (lid.y * 4u)) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if (((((wid.x * 64u) + (lid.x * 4u)) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + v6[(k2) & 3u]; }
    }
    }
    }
    }
}
