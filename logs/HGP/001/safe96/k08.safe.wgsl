// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { var t = v; return t[i & 3u]; }
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
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u;
    var i1: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) < cfg.m)) {
    if ((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows)) {
    if (((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d)) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + (cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))))];
    }
    }
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f1 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u))) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    v0 = wg_va[(((((((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u)) * 16u) + (((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) >> ((2u) & 31u)))) & 255u)];
    v0[((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) & 3u] = f0;
    wg_va[(((((((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u)) * 16u) + (((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) >> ((2u) & 31u)))) & 255u)] = v0;
    v1 = wg_vb[(((((((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u)) * 16u) + (((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) >> ((2u) & 31u)))) & 255u)];
    v1[((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) & 3u] = f1;
    wg_vb[(((((((((lid.y * 16u) + lid.x) + (k * 256u))) >> ((6u) & 31u)) * 16u) + (((((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) >> ((2u) & 31u)))) & 255u)] = v1;
    }
    workgroupBarrier();
    i0 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
    v2 = wg_va[((((k * 16u) + lid.y)) & 255u)];
    v3 = wg_vb[((((k * 16u) + lid.x)) & 255u)];
    v4 = (v4 + (vec4<f32>(v2.x) * v3));
    v5 = (v5 + (vec4<f32>(v2.y) * v3));
    v6 = (v6 + (vec4<f32>(v2.z) * v3));
    v7 = (v7 + (vec4<f32>(v2.w) * v3));
    }
    workgroupBarrier();
    }
    pv[0u] = v4;
    pv[1u] = v5;
    pv[2u] = v6;
    pv[3u] = v7;
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((gid.y * 4u) + min(k, 3u)) < cfg.r_rows)) {
    v8 = pv[((k) & 15u)];
    i1 = 0u;
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + min(k2, 3u)) < cfg.d)) {
    if (((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows) & ((((gid.x * 4u) + min(k2, 3u))) < cfg.d) & ((cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d)) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + at_vec4f(v8, k2); }
    }
    }
    }
    }
}
