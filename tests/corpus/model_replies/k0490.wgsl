// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var c0: vec4<f32> = vec4<f32>(0.0);
    var c1: vec4<f32> = vec4<f32>(0.0);
    var c2: vec4<f32> = vec4<f32>(0.0);
    var c3: vec4<f32> = vec4<f32>(0.0);
    var p0: vec4<f32> = vec4<f32>(0.0);
    var p1: vec4<f32> = vec4<f32>(0.0);
    var p2: vec4<f32> = vec4<f32>(0.0);
    var p3: vec4<f32> = vec4<f32>(0.0);
    var av: vec4<f32> = vec4<f32>(0.0);
    var bv: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var idx: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            idx = lidx + 256u * k;
            f0 = 0.0f;
            f1 = 0.0f;
            if ((t + (idx >> 6u)) < cfg.m) {
                if (((wid.x * 64u) + (idx & 63u)) < cfg.d) {
                    f0 = phi[((t + (idx >> 6u)) * cfg.d) + (wid.x * 64u) + (idx & 63u)];
                }
                if ((((wid.y * 64u) + (idx & 63u)) < cfg.r_rows) && (((cfg.r_off + (wid.y * 64u)) + (idx & 63u)) < cfg.d)) {
                    f1 = phi[((t + (idx >> 6u)) * cfg.d) + cfg.r_off + (wid.y * 64u) + (idx & 63u)];
                }
            }
            wg_va[(idx >> 2u) & 511u][idx & 3u] = f0;
            wg_vb[(idx >> 2u) & 511u][idx & 3u] = f1;
        }
        workgroupBarrier();
        c0 = vec4<f32>(0.0);
        c1 = vec4<f32>(0.0);
        c2 = vec4<f32>(0.0);
        c3 = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 1u) {
            av = wg_va[((k * 16u) + lid.x) & 511u];
            bv = wg_vb[((k * 16u) + lid.y) & 511u];
            c0 = c0 + bv.x * av;
            c1 = c1 + bv.y * av;
            c2 = c2 + bv.z * av;
            c3 = c3 + bv.w * av;
        }
        p0 = p0 + c0;
        p1 = p1 + c1;
        p2 = p2 + c2;
        p3 = p3 + c3;
        workgroupBarrier();
    }
    pv[0] = p0;
    pv[1] = p1;
    pv[2] = p2;
    pv[3] = p3;
    for (k = 0u; k < 4u; k = k + 1u) {
        if ((((wid.y * 64u) + (lid.y * 4u) + k) < cfg.r_rows) && ((cfg.r_off + (wid.y * 64u) + (lid.y * 4u) + k) < cfg.d)) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((wid.x * 64u) + (lid.x * 4u) + k2) < cfg.d) {
                    idx = (((wid.y * 64u) + (lid.y * 4u) + k) * cfg.d) + (wid.x * 64u) + (lid.x * 4u) + k2;
                    h[idx] = h[idx] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
