// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let li = lidx + 256u * k;
            let item = t + (li >> 4u);
            let c0 = wid.x * 64u + (li & 15u) * 4u;
            let r0 = wid.y * 64u + (li & 15u) * 4u;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    if (c0 + j < cfg.d) {
                        v0[j] = phi[base + c0 + j];
                    }
                    if (r0 + j < cfg.r_rows && cfg.r_off + r0 + j < cfg.d) {
                        v1[j] = phi[base + cfg.r_off + r0 + j];
                    }
                }
            }
            wg_va[li & 511u] = v0;
            wg_vb[li & 511u] = v1;
        }
        workgroupBarrier();
        var p0 = vec4<f32>(0.0);
        var p1 = vec4<f32>(0.0);
        var p2 = vec4<f32>(0.0);
        var p3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let bv = wg_vb[(k * 16u + lid.y) & 511u];
            let av = wg_va[(k * 16u + lid.x) & 511u];
            p0 = p0 + vec4<f32>(bv.x) * av;
            p1 = p1 + vec4<f32>(bv.y) * av;
            p2 = p2 + vec4<f32>(bv.z) * av;
            p3 = p3 + vec4<f32>(bv.w) * av;
        }
        if (((t >> 5u) & 1u) == 0u) {
            a0 = a0 + p0; a1 = a1 + p1; a2 = a2 + p2; a3 = a3 + p3;
        } else {
            b0 = b0 + p0; b1 = b1 + p1; b2 = b2 + p2; b3 = b3 + p3;
        }
        workgroupBarrier();
    }
    s0 = a0 + b0; s1 = a1 + b1; s2 = a2 + b2; s3 = a3 + b3;
    pv[0u] = s0; pv[1u] = s1; pv[2u] = s2; pv[3u] = s3;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
