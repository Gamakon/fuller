// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
    for (k = 0u; k < 8u; k = k + 1u) {
    wg_a[(((lidx + (256u * k))) & 2047u)] = 0.0f;
    wg_b[(((lidx + (256u * k))) & 2047u)] = 0.0f;
    if ((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m) && (((wid.x * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.d))) {
    wg_a[(((lidx + (256u * k))) & 2047u)] = phi[((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + (wid.x * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    if (((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) < cfg.m) && (((wid.y * 64u) + (((lidx + (256u * k))) & (63u))) < cfg.r_rows)) && (((cfg.r_off + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u))) < cfg.d))) {
    wg_b[(((lidx + (256u * k))) & 2047u)] = phi[(((((t + (((lidx + (256u * k))) >> ((6u) & 31u))) * cfg.d) + cfg.r_off) + (wid.y * 64u)) + (((lidx + (256u * k))) & (63u)))];
    }
    }
    workgroupBarrier();
    v0 = vec4<f32>(0.0f);
    v1 = vec4<f32>(0.0f);
    v2 = vec4<f32>(0.0f);
    v3 = vec4<f32>(0.0f);
    for (k = 0u; k < 32u; k = k + 1u) {
    let av = vec4<f32>(wg_a[((k * 64u) + (lid.x * 4u)) & 2047u], wg_a[(((k * 64u) + (lid.x * 4u)) + 1u) & 2047u], wg_a[(((k * 64u) + (lid.x * 4u)) + 2u) & 2047u], wg_a[(((k * 64u) + (lid.x * 4u)) + 3u) & 2047u]);
    v0 = v0 + vec4<f32>(wg_b[((k * 64u) + (lid.y * 4u)) & 2047u]) * av;
    v1 = v1 + vec4<f32>(wg_b[(((k * 64u) + (lid.y * 4u)) + 1u) & 2047u]) * av;
    v2 = v2 + vec4<f32>(wg_b[(((k * 64u) + (lid.y * 4u)) + 2u) & 2047u]) * av;
    v3 = v3 + vec4<f32>(wg_b[(((k * 64u) + (lid.y * 4u)) + 3u) & 2047u]) * av;
    }
    v4 = (v4 + v0);
    v5 = (v5 + v1);
    v6 = (v6 + v2);
    v7 = (v7 + v3);
    workgroupBarrier();
    }
    pv[0u] = v4;
    pv[1u] = v5;
    pv[2u] = v6;
    pv[3u] = v7;
    for (k = 0u; k < 4u; k = k + 1u) {
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
    let lr = (wid.y * 64u) + (lid.y * 4u) + k;
    let cc = (wid.x * 64u) + (lid.x * 4u) + k2;
    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && cc < cfg.d) {
    h[lr * cfg.d + cc] = h[lr * cfg.d + cc] + pv[k & 15u][k2 & 3u];
    }
    }
    }
}
