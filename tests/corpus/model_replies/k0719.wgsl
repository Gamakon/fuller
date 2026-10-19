// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0); var v9: vec4<f32> = vec4<f32>(0.0); var v10: vec4<f32> = vec4<f32>(0.0); var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0); var v13: vec4<f32> = vec4<f32>(0.0); var v14: vec4<f32> = vec4<f32>(0.0); var v15: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var q: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (q = 0u; q < 8u; q = q + 1u) {
            let idx = lidx + 256u * q;
            let kk = idx >> 6u;
            let cc = idx & 63u;
            var fa: f32 = 0.0;
            var fb: f32 = 0.0;
            if ((t + kk) < cfg.m) {
                let base = (t + kk) * cfg.d;
                let lr = row0 + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    fa = phi[base + cfg.r_off + lr];
                }
                let c = col0 + cc;
                if (c < cfg.d) {
                    fb = phi[base + c];
                }
            }
            wg_a[idx & 2047u] = fa;
            wg_b[idx & 2047u] = fb;
        }
        workgroupBarrier();
        v4 = vec4<f32>(0.0); v5 = vec4<f32>(0.0); v6 = vec4<f32>(0.0); v7 = vec4<f32>(0.0);
        v8 = vec4<f32>(0.0); v9 = vec4<f32>(0.0); v10 = vec4<f32>(0.0); v11 = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 2u) {
            let o0 = k * 64u;
            let b0 = vec4<f32>(wg_b[(o0 + lid.x) & 2047u], wg_b[(o0 + lid.x + 16u) & 2047u], wg_b[(o0 + lid.x + 32u) & 2047u], wg_b[(o0 + lid.x + 48u) & 2047u]);
            v4 = v4 + vec4<f32>(wg_a[(o0 + lid.y) & 2047u]) * b0;
            v5 = v5 + vec4<f32>(wg_a[(o0 + lid.y + 16u) & 2047u]) * b0;
            v6 = v6 + vec4<f32>(wg_a[(o0 + lid.y + 32u) & 2047u]) * b0;
            v7 = v7 + vec4<f32>(wg_a[(o0 + lid.y + 48u) & 2047u]) * b0;
            let o1 = o0 + 64u;
            let b1 = vec4<f32>(wg_b[(o1 + lid.x) & 2047u], wg_b[(o1 + lid.x + 16u) & 2047u], wg_b[(o1 + lid.x + 32u) & 2047u], wg_b[(o1 + lid.x + 48u) & 2047u]);
            v8 = v8 + vec4<f32>(wg_a[(o1 + lid.y) & 2047u]) * b1;
            v9 = v9 + vec4<f32>(wg_a[(o1 + lid.y + 16u) & 2047u]) * b1;
            v10 = v10 + vec4<f32>(wg_a[(o1 + lid.y + 32u) & 2047u]) * b1;
            v11 = v11 + vec4<f32>(wg_a[(o1 + lid.y + 48u) & 2047u]) * b1;
        }
        if (((t >> 5u) & 1u) == 0u) {
            v0 = v0 + (v4 + v8);
            v1 = v1 + (v5 + v9);
            v2 = v2 + (v6 + v10);
            v3 = v3 + (v7 + v11);
        } else {
            v12 = v12 + (v4 + v8);
            v13 = v13 + (v5 + v9);
            v14 = v14 + (v6 + v10);
            v15 = v15 + (v7 + v11);
        }
        workgroupBarrier();
    }
    pv[0u] = v0 + v12;
    pv[1u] = v1 + v13;
    pv[2u] = v2 + v14;
    pv[3u] = v3 + v15;
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = row0 + lid.y + 16u * k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = col0 + lid.x + 16u * k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
