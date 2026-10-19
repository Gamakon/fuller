// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0); var p3 = vec4<f32>(0.0);
    var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let e = lidx + 256u * j;
            let it = t + (e >> 6u);
            let c = e & 63u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m) {
                if (cbase + c < cfg.d) {
                    va = phi[it * cfg.d + cbase + c];
                }
                if (rbase + c < cfg.r_rows && cfg.r_off + rbase + c < cfg.d) {
                    vb = phi[it * cfg.d + cfg.r_off + rbase + c];
                }
            }
            wg_a[e] = va;
            wg_b[e] = vb;
        }
        workgroupBarrier();
        var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let o = k * 64u;
            let a = vec4<f32>(wg_a[o + lid.x], wg_a[o + lid.x + 16u], wg_a[o + lid.x + 32u], wg_a[o + lid.x + 48u]);
            let b0 = wg_b[o + lid.y];
            let b1 = wg_b[o + lid.y + 16u];
            let b2 = wg_b[o + lid.y + 32u];
            let b3 = wg_b[o + lid.y + 48u];
            c0 = c0 + vec4<f32>(b0) * a;
            c1 = c1 + vec4<f32>(b1) * a;
            c2 = c2 + vec4<f32>(b2) * a;
            c3 = c3 + vec4<f32>(b3) * a;
        }
        if (((t >> 5u) & 1u) == 0u) {
            p0 = p0 + c0; p1 = p1 + c1; p2 = p2 + c2; p3 = p3 + c3;
        } else {
            q0 = q0 + c0; q1 = q1 + c1; q2 = q2 + c2; q3 = q3 + c3;
        }
        workgroupBarrier();
    }
    p0 = p0 + q0; p1 = p1 + q1; p2 = p2 + q2; p3 = p3 + q3;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rbase + lid.y + 16u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            var pv = p0;
            if (j == 1u) { pv = p1; }
            if (j == 2u) { pv = p2; }
            if (j == 3u) { pv = p3; }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let col = cbase + lid.x + 16u * i;
                if (col < cfg.d) {
                    let idx = lr * cfg.d + col;
                    h[idx] = h[idx] + pv[i];
                }
            }
        }
    }
}
