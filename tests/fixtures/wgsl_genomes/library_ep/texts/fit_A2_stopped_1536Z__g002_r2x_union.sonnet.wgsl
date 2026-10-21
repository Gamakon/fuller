// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s0 = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var p0: vec4<f32> = vec4<f32>(0.0); var p1: vec4<f32> = vec4<f32>(0.0); var p2: vec4<f32> = vec4<f32>(0.0); var p3: vec4<f32> = vec4<f32>(0.0);
    var q0: vec4<f32> = vec4<f32>(0.0); var q1: vec4<f32> = vec4<f32>(0.0); var q2: vec4<f32> = vec4<f32>(0.0); var q3: vec4<f32> = vec4<f32>(0.0);
    var r0: vec4<f32> = vec4<f32>(0.0); var r1: vec4<f32> = vec4<f32>(0.0); var r2: vec4<f32> = vec4<f32>(0.0); var r3: vec4<f32> = vec4<f32>(0.0);
    var u0: vec4<f32> = vec4<f32>(0.0); var u1: vec4<f32> = vec4<f32>(0.0); var u2: vec4<f32> = vec4<f32>(0.0); var u3: vec4<f32> = vec4<f32>(0.0);
    var a0: vec4<f32>; var a1: vec4<f32>; var a2: vec4<f32>; var a3: vec4<f32>;
    var b0: vec4<f32>; var b1: vec4<f32>; var b2: vec4<f32>; var b3: vec4<f32>;
    var t: u32 = 0u; var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let i2 = lidx + 256u * k;
            let it = t + (i2 >> 4u);
            let c4 = wid.x * 64u + (i2 & 15u) * 4u;
            let r4 = wid.y * 64u + (i2 & 15u) * 4u;
            var x = vec4<f32>(0.0);
            var y = vec4<f32>(0.0);
            if (it < cfg.m) {
                let base = it * cfg.d;
                if (c4 < cfg.d) { x.x = phi[base + c4]; }
                if (c4 + 1u < cfg.d) { x.y = phi[base + c4 + 1u]; }
                if (c4 + 2u < cfg.d) { x.z = phi[base + c4 + 2u]; }
                if (c4 + 3u < cfg.d) { x.w = phi[base + c4 + 3u]; }
                let g = cfg.r_off + r4;
                if (r4 < cfg.r_rows && g < cfg.d) { y.x = phi[base + g]; }
                if (r4 + 1u < cfg.r_rows && g + 1u < cfg.d) { y.y = phi[base + g + 1u]; }
                if (r4 + 2u < cfg.r_rows && g + 2u < cfg.d) { y.z = phi[base + g + 2u]; }
                if (r4 + 3u < cfg.r_rows && g + 3u < cfg.d) { y.w = phi[base + g + 3u]; }
            }
            wg_va[i2 & 511u] = x;
            wg_vb[i2 & 511u] = y;
        }
        workgroupBarrier();
        a0 = vec4<f32>(0.0); a1 = vec4<f32>(0.0); a2 = vec4<f32>(0.0); a3 = vec4<f32>(0.0);
        b0 = vec4<f32>(0.0); b1 = vec4<f32>(0.0); b2 = vec4<f32>(0.0); b3 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let xa = wg_va[((2u * k) * 16u + lid.x) & 511u];
            let ya = wg_vb[((2u * k) * 16u + lid.y) & 511u];
            let xb = wg_va[((2u * k + 1u) * 16u + lid.x) & 511u];
            let yb = wg_vb[((2u * k + 1u) * 16u + lid.y) & 511u];
            a0 = a0 + vec4<f32>(ya.x) * xa;
            a1 = a1 + vec4<f32>(ya.y) * xa;
            a2 = a2 + vec4<f32>(ya.z) * xa;
            a3 = a3 + vec4<f32>(ya.w) * xa;
            b0 = b0 + vec4<f32>(yb.x) * xb;
            b1 = b1 + vec4<f32>(yb.y) * xb;
            b2 = b2 + vec4<f32>(yb.z) * xb;
            b3 = b3 + vec4<f32>(yb.w) * xb;
        }
        let sel = (t >> 5u) & 3u;
        if (sel == 0u) {
            p0 = p0 + (a0 + b0); p1 = p1 + (a1 + b1); p2 = p2 + (a2 + b2); p3 = p3 + (a3 + b3);
        } else if (sel == 1u) {
            q0 = q0 + (a0 + b0); q1 = q1 + (a1 + b1); q2 = q2 + (a2 + b2); q3 = q3 + (a3 + b3);
        } else if (sel == 2u) {
            r0 = r0 + (a0 + b0); r1 = r1 + (a1 + b1); r2 = r2 + (a2 + b2); r3 = r3 + (a3 + b3);
        } else {
            u0 = u0 + (a0 + b0); u1 = u1 + (a1 + b1); u2 = u2 + (a2 + b2); u3 = u3 + (a3 + b3);
        }
        workgroupBarrier();
    }
    s0[0] = (p0 + q0) + (r0 + u0);
    s0[1] = (p1 + q1) + (r1 + u1);
    s0[2] = (p2 + q2) + (r2 + u2);
    s0[3] = (p3 + q3) + (r3 + u3);
    for (k = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 64u + lid.y * 4u + k;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            let rv = s0[k];
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x * 4u + k2;
                if (col < cfg.d) {
                    let idx = row * cfg.d + col;
                    h[idx] = h[idx] + rv[k2];
                }
            }
        }
    }
}
