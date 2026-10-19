// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0); var v16: vec4<f32> = vec4<f32>(0.0); var v17: vec4<f32> = vec4<f32>(0.0);
    var v18: vec4<f32> = vec4<f32>(0.0); var v19: vec4<f32> = vec4<f32>(0.0); var v20: vec4<f32> = vec4<f32>(0.0); var v21: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let cx = wid.x * 64u;
    let cy = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let tb = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let sl = lidx + 128u * k2;
                let it = tb + (sl >> 4u);
                let off = (sl & 15u) * 4u;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (it < cfg.m) {
                    let rowb = it * cfg.d;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        if (cx + off + j < cfg.d) {
                            a[j] = phi[rowb + cx + off + j];
                        }
                        if (cy + off + j < cfg.r_rows && cfg.r_off + cy + off + j < cfg.d) {
                            b[j] = phi[rowb + cfg.r_off + cy + off + j];
                        }
                    }
                }
                wg_va[sl & 511u] = a;
                wg_vb[sl & 511u] = b;
            }
            workgroupBarrier();
            v0 = vec4<f32>(0.0); v1 = vec4<f32>(0.0); v2 = vec4<f32>(0.0); v3 = vec4<f32>(0.0);
            v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let i8 = q * 32u;
                let i9 = i8 + lid.y * 2u;
                let v8 = wg_va[(i8 + lid.x) & 511u];
                let v9 = wg_va[(i8 + 16u + lid.x) & 511u];
                let v10 = wg_vb[i9 & 511u];
                let v11 = wg_vb[(i9 + 1u) & 511u];
                let v12 = wg_vb[(i9 + 16u) & 511u];
                let v13 = wg_vb[(i9 + 17u) & 511u];
                v0 = v0 + (vec4<f32>(v10.x) * v8 + vec4<f32>(v12.x) * v9);
                v1 = v1 + (vec4<f32>(v10.y) * v8 + vec4<f32>(v12.y) * v9);
                v2 = v2 + (vec4<f32>(v10.z) * v8 + vec4<f32>(v12.z) * v9);
                v3 = v3 + (vec4<f32>(v10.w) * v8 + vec4<f32>(v12.w) * v9);
                v4 = v4 + (vec4<f32>(v11.x) * v8 + vec4<f32>(v13.x) * v9);
                v5 = v5 + (vec4<f32>(v11.y) * v8 + vec4<f32>(v13.y) * v9);
                v6 = v6 + (vec4<f32>(v11.z) * v8 + vec4<f32>(v13.z) * v9);
                v7 = v7 + (vec4<f32>(v11.w) * v8 + vec4<f32>(v13.w) * v9);
            }
            v14 = v14 + v0; v15 = v15 + v1; v16 = v16 + v2; v17 = v17 + v3;
            v18 = v18 + v4; v19 = v19 + v5; v20 = v20 + v6; v21 = v21 + v7;
            workgroupBarrier();
        }
        pv[0] = pv[0] + v14; pv[1] = pv[1] + v15; pv[2] = pv[2] + v16; pv[3] = pv[3] + v17;
        pv[4] = pv[4] + v18; pv[5] = pv[5] + v19; pv[6] = pv[6] + v20; pv[7] = pv[7] + v21;
        v14 = vec4<f32>(0.0); v15 = vec4<f32>(0.0); v16 = vec4<f32>(0.0); v17 = vec4<f32>(0.0);
        v18 = vec4<f32>(0.0); v19 = vec4<f32>(0.0); v20 = vec4<f32>(0.0); v21 = vec4<f32>(0.0);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    h[lr * cfg.d + c] = h[lr * cfg.d + c] + pv[k][k2];
                }
            }
        }
    }
}
