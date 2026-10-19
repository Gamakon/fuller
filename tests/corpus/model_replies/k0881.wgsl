// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>) {
    var accA: array<vec4<f32>, 4>;
    var accB: array<vec4<f32>, 4>;
    for (var q = 0u; q < 4u; q = q + 1u) {
        accA[q & 3u] = vec4<f32>(0.0);
        accB[q & 3u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let ly4 = lid.y * 4u;
    let lx4 = lid.x * 4u;
    for (var t = 0u; t < m; t = t + 8u) {
        // fused cooperative load of both tiles
        for (var k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = t + (idx >> 6u);
            let lc = idx & 63u;
            let col = wid.x * 64u + lc;
            let lrow = wid.y * 64u + lc;
            let grow = cfg.r_off + lrow;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (it < m) {
                if (col < d) {
                    fb = phi[it * d + col];
                }
                if (lrow < cfg.r_rows && grow < d) {
                    fa = phi[it * d + grow];
                }
            }
            wg_a[idx & 511u] = fa;
            wg_b[idx & 511u] = fb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 8u; k = k + 2u) {
            let b0 = k * 64u + lx4;
            let b1 = b0 + 64u;
            let bv0 = vec4<f32>(wg_b[(b0) & 511u], wg_b[(b0 + 1u) & 511u], wg_b[(b0 + 2u) & 511u], wg_b[(b0 + 3u) & 511u]);
            let bv1 = vec4<f32>(wg_b[(b1) & 511u], wg_b[(b1 + 1u) & 511u], wg_b[(b1 + 2u) & 511u], wg_b[(b1 + 3u) & 511u]);
            let a0 = k * 64u + ly4;
            let a1 = a0 + 64u;
            for (var r = 0u; r < 4u; r = r + 1u) {
                let x0 = wg_a[(a0 + r) & 511u];
                let x1 = wg_a[(a1 + r) & 511u];
                accA[r & 3u] = fma(vec4<f32>(x0), bv0, accA[r & 3u]);
                accB[r & 3u] = fma(vec4<f32>(x1), bv1, accB[r & 3u]);
            }
        }
        workgroupBarrier();
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let lrow = wid.y * 64u + ly4 + r;
        if (lrow < cfg.r_rows && (cfg.r_off + lrow) < d) {
            let s = accA[r & 3u] + accB[r & 3u];
            let rb = lrow * d;
            for (var c = 0u; c < 4u; c = c + 1u) {
                let col = wid.x * 64u + lx4 + c;
                if (col < d) {
                    h[rb + col] = h[rb + col] + s[c];
                }
            }
        }
    }
}
