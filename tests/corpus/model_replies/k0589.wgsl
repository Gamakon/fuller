// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var s0: vec4<f32> = vec4<f32>(0.0); var s1: vec4<f32> = vec4<f32>(0.0); var s2: vec4<f32> = vec4<f32>(0.0); var s3: vec4<f32> = vec4<f32>(0.0);
    var s4: vec4<f32> = vec4<f32>(0.0); var s5: vec4<f32> = vec4<f32>(0.0); var s6: vec4<f32> = vec4<f32>(0.0); var s7: vec4<f32> = vec4<f32>(0.0);
    var s8: vec4<f32> = vec4<f32>(0.0); var s9: vec4<f32> = vec4<f32>(0.0); var s10: vec4<f32> = vec4<f32>(0.0); var s11: vec4<f32> = vec4<f32>(0.0);
    var s12: vec4<f32> = vec4<f32>(0.0); var s13: vec4<f32> = vec4<f32>(0.0); var s14: vec4<f32> = vec4<f32>(0.0); var s15: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let e = lidx + 256u * k;
            let it = t + (e >> 6u);
            let cc = e & 63u;
            let gc = wid.x * 64u + cc;
            let lr = wid.y * 64u + cc;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m) {
                if (gc < cfg.d) {
                    va = phi[it * cfg.d + gc];
                }
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    vb = phi[it * cfg.d + cfg.r_off + lr];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0); var p1: vec4<f32> = vec4<f32>(0.0); var p2: vec4<f32> = vec4<f32>(0.0); var p3: vec4<f32> = vec4<f32>(0.0);
        var q0: vec4<f32> = vec4<f32>(0.0); var q1: vec4<f32> = vec4<f32>(0.0); var q2: vec4<f32> = vec4<f32>(0.0); var q3: vec4<f32> = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let o0 = (2u * k) * 64u;
            let o1 = o0 + 64u;
            let a0 = vec4<f32>(wg_a[o0 + lid.x], wg_a[o0 + 16u + lid.x], wg_a[o0 + 32u + lid.x], wg_a[o0 + 48u + lid.x]);
            let b0 = vec4<f32>(wg_b[o0 + lid.y], wg_b[o0 + 16u + lid.y], wg_b[o0 + 32u + lid.y], wg_b[o0 + 48u + lid.y]);
            let a1 = vec4<f32>(wg_a[o1 + lid.x], wg_a[o1 + 16u + lid.x], wg_a[o1 + 32u + lid.x], wg_a[o1 + 48u + lid.x]);
            let b1 = vec4<f32>(wg_b[o1 + lid.y], wg_b[o1 + 16u + lid.y], wg_b[o1 + 32u + lid.y], wg_b[o1 + 48u + lid.y]);
            p0 = p0 + vec4<f32>(b0.x) * a0;
            p1 = p1 + vec4<f32>(b0.y) * a0;
            p2 = p2 + vec4<f32>(b0.z) * a0;
            p3 = p3 + vec4<f32>(b0.w) * a0;
            q0 = q0 + vec4<f32>(b1.x) * a1;
            q1 = q1 + vec4<f32>(b1.y) * a1;
            q2 = q2 + vec4<f32>(b1.z) * a1;
            q3 = q3 + vec4<f32>(b1.w) * a1;
        }
        let ph = (t >> 5u) & 3u;
        if (ph == 0u) {
            s0 = s0 + (p0 + q0); s1 = s1 + (p1 + q1); s2 = s2 + (p2 + q2); s3 = s3 + (p3 + q3);
        } else {
            if (ph == 1u) {
                s4 = s4 + (p0 + q0); s5 = s5 + (p1 + q1); s6 = s6 + (p2 + q2); s7 = s7 + (p3 + q3);
            } else {
                if (ph == 2u) {
                    s8 = s8 + (p0 + q0); s9 = s9 + (p1 + q1); s10 = s10 + (p2 + q2); s11 = s11 + (p3 + q3);
                } else {
                    s12 = s12 + (p0 + q0); s13 = s13 + (p1 + q1); s14 = s14 + (p2 + q2); s15 = s15 + (p3 + q3);
                }
            }
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0u] = (s0 + s4) + (s8 + s12);
    pv[1u] = (s1 + s5) + (s9 + s13);
    pv[2u] = (s2 + s6) + (s10 + s14);
    pv[3u] = (s3 + s7) + (s11 + s15);
    var i: u32 = 0u;
    var j: u32 = 0u;
    for (i = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + i * 16u + lid.y;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let col = wid.x * 64u + j * 16u + lid.x;
                if (col < cfg.d) {
                    let idx = lr * cfg.d + col;
                    h[idx] = h[idx] + pv[i][j];
                }
            }
        }
    }
}
