// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
fn at_vec4f(v: vec4<f32>, i: u32) -> f32 { return v[i & 3u]; }
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var f0: f32 = 0.0; var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var i0: u32 = 0u; var i1: u32 = 0u; var i2: u32 = 0u; var i3: u32 = 0u; var i4: u32 = 0u;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 16u) {
    for (k = 0u; k < 4u; k = k + 1u) {
    i0 = ((lid.y * 16u) + lid.x) + (k * 256u);
    i1 = i0 >> 6u;
    i2 = i0 & 63u;
    f0 = 0.0;
    f1 = 0.0;
    if ((t + i1) < cfg.m) {
    if (((wid.x * 64u) + i2) < cfg.d) {
    f0 = phi[((t + i1) * cfg.d) + ((wid.x * 64u) + i2)];
    }
    if (((wid.y * 64u) + i2) < cfg.r_rows) {
    if ((cfg.r_off + ((wid.y * 64u) + i2)) < cfg.d) {
    f1 = phi[(((t + i1) * cfg.d) + cfg.r_off) + ((wid.y * 64u) + i2)];
    }
    }
    }
    wg_a[i0 & 2047u] = f0;
    wg_b[i0 & 2047u] = f1;
    }
    workgroupBarrier();
    for (k = 0u; k < 2u; k = k + 1u) {
    i3 = (k * 512u) + (lid.x * 4u);
    i4 = (k * 512u) + (lid.y * 4u);
    v8 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[i3 & 2047u], wg_a[(i3 + 1u) & 2047u]), wg_a[(i3 + 2u) & 2047u]), wg_a[(i3 + 3u) & 2047u]);
    v9 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 64u) & 2047u], wg_a[(i3 + 65u) & 2047u]), wg_a[(i3 + 66u) & 2047u]), wg_a[(i3 + 67u) & 2047u]);
    v10 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 128u) & 2047u], wg_a[(i3 + 129u) & 2047u]), wg_a[(i3 + 130u) & 2047u]), wg_a[(i3 + 131u) & 2047u]);
    v11 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 192u) & 2047u], wg_a[(i3 + 193u) & 2047u]), wg_a[(i3 + 194u) & 2047u]), wg_a[(i3 + 195u) & 2047u]);
    v12 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[i4 & 2047u], wg_b[(i4 + 1u) & 2047u]), wg_b[(i4 + 2u) & 2047u]), wg_b[(i4 + 3u) & 2047u]);
    v13 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 64u) & 2047u], wg_b[(i4 + 65u) & 2047u]), wg_b[(i4 + 66u) & 2047u]), wg_b[(i4 + 67u) & 2047u]);
    v14 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 128u) & 2047u], wg_b[(i4 + 129u) & 2047u]), wg_b[(i4 + 130u) & 2047u]), wg_b[(i4 + 131u) & 2047u]);
    v15 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 192u) & 2047u], wg_b[(i4 + 193u) & 2047u]), wg_b[(i4 + 194u) & 2047u]), wg_b[(i4 + 195u) & 2047u]);
    v0 = v0 + (((v8 * vec4<f32>(v12[0u & 3u])) + (v9 * vec4<f32>(v13[0u & 3u]))) + ((v10 * vec4<f32>(v14[0u & 3u])) + (v11 * vec4<f32>(v15[0u & 3u]))));
    v1 = v1 + (((v8 * vec4<f32>(v12[1u & 3u])) + (v9 * vec4<f32>(v13[1u & 3u]))) + ((v10 * vec4<f32>(v14[1u & 3u])) + (v11 * vec4<f32>(v15[1u & 3u]))));
    v2 = v2 + (((v8 * vec4<f32>(v12[2u & 3u])) + (v9 * vec4<f32>(v13[2u & 3u]))) + ((v10 * vec4<f32>(v14[2u & 3u])) + (v11 * vec4<f32>(v15[2u & 3u]))));
    v3 = v3 + (((v8 * vec4<f32>(v12[3u & 3u])) + (v9 * vec4<f32>(v13[3u & 3u]))) + ((v10 * vec4<f32>(v14[3u & 3u])) + (v11 * vec4<f32>(v15[3u & 3u]))));
    v8 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 256u) & 2047u], wg_a[(i3 + 257u) & 2047u]), wg_a[(i3 + 258u) & 2047u]), wg_a[(i3 + 259u) & 2047u]);
    v9 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 320u) & 2047u], wg_a[(i3 + 321u) & 2047u]), wg_a[(i3 + 322u) & 2047u]), wg_a[(i3 + 323u) & 2047u]);
    v10 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 384u) & 2047u], wg_a[(i3 + 385u) & 2047u]), wg_a[(i3 + 386u) & 2047u]), wg_a[(i3 + 387u) & 2047u]);
    v11 = vec4<f32>(vec3<f32>(vec2<f32>(wg_a[(i3 + 448u) & 2047u], wg_a[(i3 + 449u) & 2047u]), wg_a[(i3 + 450u) & 2047u]), wg_a[(i3 + 451u) & 2047u]);
    v12 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 256u) & 2047u], wg_b[(i4 + 257u) & 2047u]), wg_b[(i4 + 258u) & 2047u]), wg_b[(i4 + 259u) & 2047u]);
    v13 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 320u) & 2047u], wg_b[(i4 + 321u) & 2047u]), wg_b[(i4 + 322u) & 2047u]), wg_b[(i4 + 323u) & 2047u]);
    v14 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 384u) & 2047u], wg_b[(i4 + 385u) & 2047u]), wg_b[(i4 + 386u) & 2047u]), wg_b[(i4 + 387u) & 2047u]);
    v15 = vec4<f32>(vec3<f32>(vec2<f32>(wg_b[(i4 + 448u) & 2047u], wg_b[(i4 + 449u) & 2047u]), wg_b[(i4 + 450u) & 2047u]), wg_b[(i4 + 451u) & 2047u]);
    v4 = v4 + (((v8 * vec4<f32>(v12[0u & 3u])) + (v9 * vec4<f32>(v13[0u & 3u]))) + ((v10 * vec4<f32>(v14[0u & 3u])) + (v11 * vec4<f32>(v15[0u & 3u]))));
    v5 = v5 + (((v8 * vec4<f32>(v12[1u & 3u])) + (v9 * vec4<f32>(v13[1u & 3u]))) + ((v10 * vec4<f32>(v14[1u & 3u])) + (v11 * vec4<f32>(v15[1u & 3u]))));
    v6 = v6 + (((v8 * vec4<f32>(v12[2u & 3u])) + (v9 * vec4<f32>(v13[2u & 3u]))) + ((v10 * vec4<f32>(v14[2u & 3u])) + (v11 * vec4<f32>(v15[2u & 3u]))));
    v7 = v7 + (((v8 * vec4<f32>(v12[3u & 3u])) + (v9 * vec4<f32>(v13[3u & 3u]))) + ((v10 * vec4<f32>(v14[3u & 3u])) + (v11 * vec4<f32>(v15[3u & 3u]))));
    }
    workgroupBarrier();
    }
    pv[0u & 15u] = (v0 + v4);
    pv[1u & 15u] = (v1 + v5);
    pv[2u & 15u] = (v2 + v6);
    pv[3u & 15u] = (v3 + v7);
    for (k = 0u; k < 4u; k = k + 1u) {
    i0 = (gid.y * 4u) + k;
    if (i0 < cfg.r_rows) {
    if ((cfg.r_off + i0) < cfg.d) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    i1 = (gid.x * 4u) + k2;
    if (i1 < cfg.d) {
    h[(i0 * cfg.d) + i1] = h[(i0 * cfg.d) + i1] + at_vec4f(pv[k & 15u], k2 & 3u);
    }
    }
    }
    }
    }
}
