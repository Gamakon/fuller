// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        acc[j & 3u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t = 0u; t < m; t = t + 32u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let e = lidx + 256u * k;
            let item = t + (e >> 4u);
            let c4 = (e & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                for (var q = 0u; q < 4u; q = q + 1u) {
                    let c = colBase + c4 + q;
                    if (c < d) {
                        a[q] = phi[base + c];
                    }
                    let lr = rowBase + c4 + q;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                        b[q] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[e & 511u] = a;
            wg_vb[e & 511u] = b;
        }
        workgroupBarrier();
        for (var i = 0u; i < 32u; i = i + 1u) {
            let a = wg_va[(i * 16u + lid.x) & 511u];
            let b = wg_vb[(i * 16u + lid.y) & 511u];
            acc[0u] = acc[0u] + vec4<f32>(b.x) * a;
            acc[1u] = acc[1u] + vec4<f32>(b.y) * a;
            acc[2u] = acc[2u] + vec4<f32>(b.z) * a;
            acc[3u] = acc[3u] + vec4<f32>(b.w) * a;
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = gid.y * 4u + j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let v = acc[j & 3u];
            for (var q = 0u; q < 4u; q = q + 1u) {
                let c = gid.x * 4u + q;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + v[q];
                }
            }
        }
    }
}
