// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
fn lit_f32(x: f32) -> f32 { return x; }
fn lit_u32(x: u32) -> u32 { return x; }
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { return v[i & 3u]; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0); var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    v0 = vec4<f32>(lit_f32(0.0f));
    v1 = vec4<f32>(lit_f32(0.0f));
    v2 = vec4<f32>(lit_f32(0.0f));
    v3 = vec4<f32>(lit_f32(0.0f));
    v4 = vec4<f32>(lit_f32(0.0f));
    v5 = vec4<f32>(lit_f32(0.0f));
    v6 = vec4<f32>(lit_f32(0.0f));
    v7 = vec4<f32>(lit_f32(0.0f));
    for (t = 0u; t < cfg.m; t = t + 8u) {
    for (k = 0u; k < 2u; k = k + 1u) {
    f0 = lit_f32(0.0f);
    f1 = lit_f32(0.0f);
    if (((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) < cfg.m)) {
    if ((((wid.x * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))) < cfg.d)) {
    f0 = phi[(((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) * cfg.d) + ((wid.x * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))))];
    }
    if (((((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))) < cfg.r_rows) && ((cfg.r_off + ((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u))))) < cfg.d))) {
    f1 = phi[((((t + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) >> ((lit_u32(6u)) & 31u))) * cfg.d) + cfg.r_off) + ((wid.y * lit_u32(64u)) + (((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & (lit_u32(63u)))))];
    }
    }
    wg_a[(((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & 2047u)] = f0;
    wg_b[(((((lid.y * lit_u32(16u)) + lid.x) + (k * lit_u32(256u)))) & 2047u)] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 2u; k = k + 1u) {
    v8 = vec4<f32>(wg_a[(((k * lit_u32(256u)) + (lid.x * lit_u32(4u))) & 2047u)], wg_a[((((k * lit_u32(256u)) + (lid.x * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_a[((((k * lit_u32(256u)) + (lid.x * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_a[((((k * lit_u32(256u)) + (lid.x * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v9 = vec4<f32>(wg_a[((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.x * lit_u32(4u))) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.x * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v10 = vec4<f32>(wg_a[((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.x * lit_u32(4u))) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.x * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.x * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.x * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v11 = vec4<f32>(wg_a[((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.x * lit_u32(4u))) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.x * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.x * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_a[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.x * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v12 = vec4<f32>(wg_b[(((k * lit_u32(256u)) + (lid.y * lit_u32(4u))) & 2047u)], wg_b[((((k * lit_u32(256u)) + (lid.y * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_b[((((k * lit_u32(256u)) + (lid.y * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_b[((((k * lit_u32(256u)) + (lid.y * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v13 = vec4<f32>(wg_b[((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.y * lit_u32(4u))) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(64u)) + (lid.y * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v14 = vec4<f32>(wg_b[((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.y * lit_u32(4u))) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.y * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.y * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(128u)) + (lid.y * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    v15 = vec4<f32>(wg_b[((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.y * lit_u32(4u))) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.y * lit_u32(4u))) + lit_u32(1u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.y * lit_u32(4u))) + lit_u32(2u)) & 2047u)], wg_b[(((((k * lit_u32(256u)) + lit_u32(192u)) + (lid.y * lit_u32(4u))) + lit_u32(3u)) & 2047u)]);
    if ((k == lit_u32(0u))) {
    v0 = (v0 + (((v8 * v12.x) + (v9 * v13.x)) + ((v10 * v14.x) + (v11 * v15.x))));
    v1 = (v1 + (((v8 * v12.y) + (v9 * v13.y)) + ((v10 * v14.y) + (v11 * v15.y))));
    v2 = (v2 + (((v8 * v12.z) + (v9 * v13.z)) + ((v10 * v14.z) + (v11 * v15.z))));
    v3 = (v3 + (((v8 * v12.w) + (v9 * v13.w)) + ((v10 * v14.w) + (v11 * v15.w))));
    } else {
    v4 = (v4 + (((v8 * v12.x) + (v9 * v13.x)) + ((v10 * v14.x) + (v11 * v15.x))));
    v5 = (v5 + (((v8 * v12.y) + (v9 * v13.y)) + ((v10 * v14.y) + (v11 * v15.y))));
    v6 = (v6 + (((v8 * v12.z) + (v9 * v13.z)) + ((v10 * v14.z) + (v11 * v15.z))));
    v7 = (v7 + (((v8 * v12.w) + (v9 * v13.w)) + ((v10 * v14.w) + (v11 * v15.w))));
    }
    }
    workgroupBarrier();
    }
    pv[((lit_u32(0u)) & 15u)] = (v0 + v4);
    pv[((lit_u32(1u)) & 15u)] = (v1 + v5);
    pv[((lit_u32(2u)) & 15u)] = (v2 + v6);
    pv[((lit_u32(3u)) & 15u)] = (v3 + v7);
    for (k = 0u; k < 4u; k = k + 1u) {
    if (((((gid.y * lit_u32(4u)) + k) < cfg.r_rows) && ((cfg.r_off + ((gid.y * lit_u32(4u)) + k)) < cfg.d))) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    if ((((gid.x * lit_u32(4u)) + k2) < cfg.d)) {
    if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + at_vec4f(pv[((k) & 15u)], k2); }
    }
    }
    }
    }
}
