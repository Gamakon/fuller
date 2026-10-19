// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var j: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 4u);
            let q = idx & 15u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (j = 0u; j < 4u; j = j + 1u) {
                    let c = wid.x * 64u + q + 16u * j;
                    if (c < cfg.d) {
                        a[j] = phi[base + c];
                    }
                    let lr = wid.y * 64u + q + 16u * j;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        b[j] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx & 511u] = a;
            wg_vb[idx & 511u] = b;
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0);
        v1 = vec4<f32>(0.0);
        v2 = vec4<f32>(0.0);
        v3 = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b = wg_vb[(k * 16u + lid.y) & 511u];
            v0 = v0 + vec4<f32>(b.x) * a;
            v1 = v1 + vec4<f32>(b.y) * a;
            v2 = v2 + vec4<f32>(b.z) * a;
            v3 = v3 + vec4<f32>(b.w) * a;
        }
        v6 = v6 + v0;
        v7 = v7 + v1;
        v8 = v8 + v2;
        v9 = v9 + v3;
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
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 16u * j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + pv[k][j];
                }
            }
        }
    }
}
