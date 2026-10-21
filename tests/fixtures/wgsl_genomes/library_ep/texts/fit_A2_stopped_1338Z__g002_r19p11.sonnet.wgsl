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
    var i1: u32 = 0u; var i2: u32 = 0u; var i4: u32 = 0u; var i5: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 8u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = 0.0f;
    f1 = 0.0f;
    if (((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> 6u)) < cfg.m)) {
    if ((((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.d)) {
    f0 = phi[(((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> 6u)) * cfg.d) + ((wid.x * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    if ((((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))) < cfg.r_rows)) {
    if (((cfg.r_off + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u)))) < cfg.d)) {
    f1 = phi[((((t + (((((lid.y * 16u) + lid.x) + (k * 256u))) >> 6u)) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + (((((lid.y * 16u) + lid.x) + (k * 256u))) & (63u))))];
    }
    }
    }
    wg_a[(((lid.y * 16u) + lid.x) + (k * 256u)) & 2047u] = f0;
    wg_b[(((lid.y * 16u) + lid.x) + (k * 256u)) & 2047u] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 2u; k = k + 1u) {
    v8 = vec4<f32>(wg_a[(k * 256u + lid.x * 4u) & 2047u], wg_a[(k * 256u + lid.x * 4u + 1u) & 2047u], wg_a[(k * 256u + lid.x * 4u + 2u) & 2047u], wg_a[(k * 256u + lid.x * 4u + 3u) & 2047u]);
    v9 = vec4<f32>(wg_a[(k * 256u + 64u + lid.x * 4u) & 2047u], wg_a[(k * 256u + 64u + lid.x * 4u + 1u) & 2047u], wg_a[(k * 256u + 64u + lid.x * 4u + 2u) & 2047u], wg_a[(k * 256u + 64u + lid.x * 4u + 3u) & 2047u]);
    v10 = vec4<f32>(wg_a[(k * 256u + 128u + lid.x * 4u) & 2047u], wg_a[(k * 256u + 128u + lid.x * 4u + 1u) & 2047u], wg_a[(k * 256u + 128u + lid.x * 4u + 2u) & 2047u], wg_a[(k * 256u + 128u + lid.x * 4u + 3u) & 2047u]);
    v11 = vec4<f32>(wg_a[(k * 256u + 192u + lid.x * 4u) & 2047u], wg_a[(k * 256u + 192u + lid.x * 4u + 1u) & 2047u], wg_a[(k * 256u + 192u + lid.x * 4u + 2u) & 2047u], wg_a[(k * 256u + 192u + lid.x * 4u + 3u) & 2047u]);
    v12 = vec4<f32>(wg_b[(k * 256u + lid.y * 4u) & 2047u], wg_b[(k * 256u + lid.y * 4u + 1u) & 2047u], wg_b[(k * 256u + lid.y * 4u + 2u) & 2047u], wg_b[(k * 256u + lid.y * 4u + 3u) & 2047u]);
    v13 = vec4<f32>(wg_b[(k * 256u + 64u + lid.y * 4u) & 2047u], wg_b[(k * 256u + 64u + lid.y * 4u + 1u) & 2047u], wg_b[(k * 256u + 64u + lid.y * 4u + 2u) & 2047u], wg_b[(k * 256u + 64u + lid.y * 4u + 3u) & 2047u]);
    v14 = vec4<f32>(wg_b[(k * 256u + 128u + lid.y * 4u) & 2047u], wg_b[(k * 256u + 128u + lid.y * 4u + 1u) & 2047u], wg_b[(k * 256u + 128u + lid.y * 4u + 2u) & 2047u], wg_b[(k * 256u + 128u + lid.y * 4u + 3u) & 2047u]);
    v15 = vec4<f32>(wg_b[(k * 256u + 192u + lid.y * 4u) & 2047u], wg_b[(k * 256u + 192u + lid.y * 4u + 1u) & 2047u], wg_b[(k * 256u + 192u + lid.y * 4u + 2u) & 2047u], wg_b[(k * 256u + 192u + lid.y * 4u + 3u) & 2047u]);
    if ((k == 0u)) {
    v0 = (v0 + (((v8 * vec4<f32>(v12[0u])) + (v9 * vec4<f32>(v13[0u]))) + ((v10 * vec4<f32>(v14[0u])) + (v11 * vec4<f32>(v15[0u])))));
    v1 = (v1 + (((v8 * vec4<f32>(v12[1u])) + (v9 * vec4<f32>(v13[1u]))) + ((v10 * vec4<f32>(v14[1u])) + (v11 * vec4<f32>(v15[1u])))));
    v2 = (v2 + (((v8 * vec4<f32>(v12[2u])) + (v9 * vec4<f32>(v13[2u]))) + ((v10 * vec4<f32>(v14[2u])) + (v11 * vec4<f32>(v15[2u])))));
    v3 = (v3 + (((v8 * vec4<f32>(v12[3u])) + (v9 * vec4<f32>(v13[3u]))) + ((v10 * vec4<f32>(v14[3u])) + (v11 * vec4<f32>(v15[3u])))));
    } else {
    v4 = (v4 + (((v8 * vec4<f32>(v12[0u])) + (v9 * vec4<f32>(v13[0u]))) + ((v10 * vec4<f32>(v14[0u])) + (v11 * vec4<f32>(v15[0u])))));
    v5 = (v5 + (((v8 * vec4<f32>(v12[1u])) + (v9 * vec4<f32>(v13[1u]))) + ((v10 * vec4<f32>(v14[1u])) + (v11 * vec4<f32>(v15[1u])))));
    v6 = (v6 + (((v8 * vec4<f32>(v12[2u])) + (v9 * vec4<f32>(v13[2u]))) + ((v10 * vec4<f32>(v14[2u])) + (v11 * vec4<f32>(v15[2u])))));
    v7 = (v7 + (((v8 * vec4<f32>(v12[3u])) + (v9 * vec4<f32>(v13[3u]))) + ((v10 * vec4<f32>(v14[3u])) + (v11 * vec4<f32>(v15[3u])))));
    }
    }
    workgroupBarrier();
    }
    pv[0u] = (v0 + v4);
    pv[1u] = (v1 + v5);
    pv[2u] = (v2 + v6);
    pv[3u] = (v3 + v7);
    for (k = 0u; k < 4u; k = k + 1u) {
    if ((((gid.y * 4u) + k) < cfg.r_rows)) {
    if (((cfg.r_off + ((gid.y * 4u) + k)) < cfg.d)) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * 4u) + k2) < cfg.d)) {
    h[((gid.y * 4u + k) * cfg.d + (gid.x * 4u + k2))] = h[((gid.y * 4u + k) * cfg.d + (gid.x * 4u + k2))] + at_vec4f(pv[k & 15u], k2 & 3u);
    }
    }
    }
    }
    }
}
