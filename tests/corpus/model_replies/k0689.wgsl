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
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        acc[k] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let idx = lidx + 256u * k2;
            let item = t + (idx >> 4u);
            let q = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < m) {
                let base = item * d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + q + j;
                    if (c < d) {
                        a[j] = phi[base + c];
                    }
                    let lr = rowBase + q + j;
                    let gr = cfg.r_off + lr;
                    if (lr < cfg.r_rows && gr < d) {
                        b[j] = phi[base + gr];
                    }
                }
            }
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b = wg_vb[k * 16u + lid.y];
            p0 = p0 + vec4<f32>(b.x) * a;
            p1 = p1 + vec4<f32>(b.y) * a;
            p2 = p2 + vec4<f32>(b.z) * a;
            p3 = p3 + vec4<f32>(b.w) * a;
        }
        acc[0] = acc[0] + p0;
        acc[1] = acc[1] + p1;
        acc[2] = acc[2] + p2;
        acc[3] = acc[3] + p3;
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[k][j];
                }
            }
        }
    }
}
