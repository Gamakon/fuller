// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var pv: array<vec4<f32>, 16>;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + (256u * k);
            let row = t + (idx >> 4u);
            let cc = (idx & 15u) * 4u;
            let col = (wid.x * 64u) + cc;
            let rr = (wid.y * 64u) + cc;
            v0 = vec4<f32>(0.0);
            v1 = vec4<f32>(0.0);
            if (row < cfg.m) {
                let b = row * cfg.d;
                if (col < cfg.d) { v0[0u] = phi[b + col]; }
                if (col + 1u < cfg.d) { v0[1u] = phi[b + col + 1u]; }
                if (col + 2u < cfg.d) { v0[2u] = phi[b + col + 2u]; }
                if (col + 3u < cfg.d) { v0[3u] = phi[b + col + 3u]; }
                if (rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) { v1[0u] = phi[b + cfg.r_off + rr]; }
                if (rr + 1u < cfg.r_rows && (cfg.r_off + rr + 1u) < cfg.d) { v1[1u] = phi[b + cfg.r_off + rr + 1u]; }
                if (rr + 2u < cfg.r_rows && (cfg.r_off + rr + 2u) < cfg.d) { v1[2u] = phi[b + cfg.r_off + rr + 2u]; }
                if (rr + 3u < cfg.r_rows && (cfg.r_off + rr + 3u) < cfg.d) { v1[3u] = phi[b + cfg.r_off + rr + 3u]; }
            }
            wg_va[idx & 511u] = v0;
            wg_vb[idx & 511u] = v1;
        }
        workgroupBarrier();
        v2 = vec4<f32>(0.0);
        v3 = vec4<f32>(0.0);
        v4 = vec4<f32>(0.0);
        v5 = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[((k * 16u) + lid.x) & 511u];
            let bb = wg_vb[((k * 16u) + lid.y) & 511u];
            v2 = v2 + (vec4<f32>(bb.x) * a);
            v3 = v3 + (vec4<f32>(bb.y) * a);
            v4 = v4 + (vec4<f32>(bb.z) * a);
            v5 = v5 + (vec4<f32>(bb.w) * a);
        }
        v6 = v6 + v2;
        v7 = v7 + v3;
        v8 = v8 + v4;
        v9 = v9 + v5;
        if (((t >> 5u) & 15u) == 15u) {
            v10 = v10 + v6;
            v11 = v11 + v7;
            v12 = v12 + v8;
            v13 = v13 + v9;
            v6 = vec4<f32>(0.0);
            v7 = vec4<f32>(0.0);
            v8 = vec4<f32>(0.0);
            v9 = vec4<f32>(0.0);
        }
        workgroupBarrier();
    }
    pv[0u] = v10 + v6;
    pv[1u] = v11 + v7;
    pv[2u] = v12 + v8;
    pv[3u] = v13 + v9;
    for (k = 0u; k < 4u; k = k + 1u) {
        if (((gid.y * 4u) + k) < cfg.r_rows && (cfg.r_off + ((gid.y * 4u) + k)) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                if (((gid.x * 4u) + k2) < cfg.d) {
                    let ix = (((gid.y * 4u) + k) * cfg.d) + ((gid.x * 4u) + k2);
                    h[ix] = h[ix] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
