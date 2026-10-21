// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var p0 = vec4<f32>(0.0);
    var p1 = vec4<f32>(0.0);
    var p2 = vec4<f32>(0.0);
    var p3 = vec4<f32>(0.0);
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        var c0 = vec4<f32>(0.0);
        var c1 = vec4<f32>(0.0);
        var c2 = vec4<f32>(0.0);
        var c3 = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 4u; kk = kk + 1u) {
            for (var q: u32 = 0u; q < 2u; q = q + 1u) {
                let i4 = lidx + 256u * q;
                let item = t + kk * 32u + (i4 >> 4u);
                let c4 = (i4 & 15u) * 4u;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (item < cfg.m) {
                    let base = item * cfg.d;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        let cc = colBase + c4 + j;
                        if (cc < cfg.d) {
                            a[j] = phi[base + cc];
                        }
                        let rr = rowBase + c4 + j;
                        if (rr < cfg.r_rows && cfg.r_off + rr < cfg.d) {
                            b[j] = phi[base + cfg.r_off + rr];
                        }
                    }
                }
                wg_va[i4] = a;
                wg_vb[i4] = b;
            }
            workgroupBarrier();
            var s0 = vec4<f32>(0.0);
            var s1 = vec4<f32>(0.0);
            var s2 = vec4<f32>(0.0);
            var s3 = vec4<f32>(0.0);
            for (var it: u32 = 0u; it < 32u; it = it + 1u) {
                let va = wg_va[it * 16u + lid.x];
                let vb = wg_vb[it * 16u + lid.y];
                s0 = s0 + vec4<f32>(vb.x) * va;
                s1 = s1 + vec4<f32>(vb.y) * va;
                s2 = s2 + vec4<f32>(vb.z) * va;
                s3 = s3 + vec4<f32>(vb.w) * va;
            }
            c0 = c0 + s0;
            c1 = c1 + s1;
            c2 = c2 + s2;
            c3 = c3 + s3;
            workgroupBarrier();
        }
        p0 = p0 + c0;
        p1 = p1 + c1;
        p2 = p2 + c2;
        p3 = p3 + c3;
    }
    var pr: array<vec4<f32>, 4> = array<vec4<f32>, 4>(p0, p1, p2, p3);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pr[k][j];
                }
            }
        }
    }
}
