// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var v0: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v2: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v3: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v4: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v5: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v6: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v7: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v8: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v9: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v10: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v11: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v12: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v13: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v14: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var v15: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var pv: array<vec4<f32>, 16>;
    let tid = lid.y * 16u + lid.x;
    let ba0 = lid.x * 4u;
    let bb0 = lid.y * 4u;
    let ba1 = 256u + lid.x * 4u;
    let bb1 = 256u + lid.y * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            f0 = 0.0;
            f1 = 0.0;
            let idx = tid + k * 256u;
            let it = t + (idx >> 6u);
            let cc = wid.x * 64u + (idx & 63u);
            let rr = wid.y * 64u + (idx & 63u);
            if (it < cfg.m) {
                if (cc < cfg.d) {
                    f0 = phi[it * cfg.d + cc];
                }
                if (rr < cfg.r_rows) {
                    if (cfg.r_off + rr < cfg.d) {
                        f1 = phi[it * cfg.d + cfg.r_off + rr];
                    }
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        v0 = vec4<f32>(wg_a[ba0], wg_a[ba0 + 1u], wg_a[ba0 + 2u], wg_a[ba0 + 3u]);
        v1 = vec4<f32>(wg_a[ba0 + 64u], wg_a[ba0 + 65u], wg_a[ba0 + 66u], wg_a[ba0 + 67u]);
        v2 = vec4<f32>(wg_a[ba0 + 128u], wg_a[ba0 + 129u], wg_a[ba0 + 130u], wg_a[ba0 + 131u]);
        v3 = vec4<f32>(wg_a[ba0 + 192u], wg_a[ba0 + 193u], wg_a[ba0 + 194u], wg_a[ba0 + 195u]);
        v4 = vec4<f32>(wg_b[bb0], wg_b[bb0 + 1u], wg_b[bb0 + 2u], wg_b[bb0 + 3u]);
        v5 = vec4<f32>(wg_b[bb0 + 64u], wg_b[bb0 + 65u], wg_b[bb0 + 66u], wg_b[bb0 + 67u]);
        v6 = vec4<f32>(wg_b[bb0 + 128u], wg_b[bb0 + 129u], wg_b[bb0 + 130u], wg_b[bb0 + 131u]);
        v7 = vec4<f32>(wg_b[bb0 + 192u], wg_b[bb0 + 193u], wg_b[bb0 + 194u], wg_b[bb0 + 195u]);
        v8 = v8 + ((v0 * v4.x + v1 * v5.x) + (v2 * v6.x + v3 * v7.x));
        v9 = v9 + ((v0 * v4.y + v1 * v5.y) + (v2 * v6.y + v3 * v7.y));
        v10 = v10 + ((v0 * v4.z + v1 * v5.z) + (v2 * v6.z + v3 * v7.z));
        v11 = v11 + ((v0 * v4.w + v1 * v5.w) + (v2 * v6.w + v3 * v7.w));
        v0 = vec4<f32>(wg_a[ba1], wg_a[ba1 + 1u], wg_a[ba1 + 2u], wg_a[ba1 + 3u]);
        v1 = vec4<f32>(wg_a[ba1 + 64u], wg_a[ba1 + 65u], wg_a[ba1 + 66u], wg_a[ba1 + 67u]);
        v2 = vec4<f32>(wg_a[ba1 + 128u], wg_a[ba1 + 129u], wg_a[ba1 + 130u], wg_a[ba1 + 131u]);
        v3 = vec4<f32>(wg_a[ba1 + 192u], wg_a[ba1 + 193u], wg_a[ba1 + 194u], wg_a[ba1 + 195u]);
        v4 = vec4<f32>(wg_b[bb1], wg_b[bb1 + 1u], wg_b[bb1 + 2u], wg_b[bb1 + 3u]);
        v5 = vec4<f32>(wg_b[bb1 + 64u], wg_b[bb1 + 65u], wg_b[bb1 + 66u], wg_b[bb1 + 67u]);
        v6 = vec4<f32>(wg_b[bb1 + 128u], wg_b[bb1 + 129u], wg_b[bb1 + 130u], wg_b[bb1 + 131u]);
        v7 = vec4<f32>(wg_b[bb1 + 192u], wg_b[bb1 + 193u], wg_b[bb1 + 194u], wg_b[bb1 + 195u]);
        v12 = v12 + ((v0 * v4.x + v1 * v5.x) + (v2 * v6.x + v3 * v7.x));
        v13 = v13 + ((v0 * v4.y + v1 * v5.y) + (v2 * v6.y + v3 * v7.y));
        v14 = v14 + ((v0 * v4.z + v1 * v5.z) + (v2 * v6.z + v3 * v7.z));
        v15 = v15 + ((v0 * v4.w + v1 * v5.w) + (v2 * v6.w + v3 * v7.w));
        workgroupBarrier();
    }
    pv[0u] = v8 + v12;
    pv[1u] = v9 + v13;
    pv[2u] = v10 + v14;
    pv[3u] = v11 + v15;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        if (gid.y * 4u + k < cfg.r_rows) {
            if (cfg.r_off + gid.y * 4u + k < cfg.d) {
                for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    if (gid.x * 4u + k2 < cfg.d) {
                        let o = (gid.y * 4u + k) * cfg.d + gid.x * 4u + k2;
                        h[o] = h[o] + pv[k][k2];
                    }
                }
            }
        }
    }
}
