// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0: vec4<f32> = vec4<f32>(0.0); var e1: vec4<f32> = vec4<f32>(0.0); var e2: vec4<f32> = vec4<f32>(0.0); var e3: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0); var o1: vec4<f32> = vec4<f32>(0.0); var o2: vec4<f32> = vec4<f32>(0.0); var o3: vec4<f32> = vec4<f32>(0.0);
    var p0: vec4<f32> = vec4<f32>(0.0); var p1: vec4<f32> = vec4<f32>(0.0); var p2: vec4<f32> = vec4<f32>(0.0); var p3: vec4<f32> = vec4<f32>(0.0);
    var q0: vec4<f32> = vec4<f32>(0.0); var q1: vec4<f32> = vec4<f32>(0.0); var q2: vec4<f32> = vec4<f32>(0.0); var q3: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u;
    var k: u32 = 0u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = idx >> 6u;
            let col = idx & 63u;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            let ti = t + item;
            if (ti < cfg.m) {
                let gc = wid.x * 64u + col;
                if (gc < cfg.d) {
                    a = phi[ti * cfg.d + gc];
                }
                let lr = wid.y * 64u + col;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    b = phi[ti * cfg.d + cfg.r_off + lr];
                }
            }
            wg_a[idx] = a;
            wg_b[idx] = b;
        }
        workgroupBarrier();
        e0 = vec4<f32>(0.0); e1 = vec4<f32>(0.0); e2 = vec4<f32>(0.0); e3 = vec4<f32>(0.0);
        o0 = vec4<f32>(0.0); o1 = vec4<f32>(0.0); o2 = vec4<f32>(0.0); o3 = vec4<f32>(0.0);
        for (k = 0u; k < 16u; k = k + 1u) {
            let r0 = (2u * k) * 64u;
            let r1 = (2u * k + 1u) * 64u;
            let va0 = vec4<f32>(wg_a[r0 + lid.x], wg_a[r0 + lid.x + 16u], wg_a[r0 + lid.x + 32u], wg_a[r0 + lid.x + 48u]);
            let va1 = vec4<f32>(wg_a[r1 + lid.x], wg_a[r1 + lid.x + 16u], wg_a[r1 + lid.x + 32u], wg_a[r1 + lid.x + 48u]);
            e0 = e0 + wg_b[r0 + lid.y] * va0;
            e1 = e1 + wg_b[r0 + lid.y + 16u] * va0;
            e2 = e2 + wg_b[r0 + lid.y + 32u] * va0;
            e3 = e3 + wg_b[r0 + lid.y + 48u] * va0;
            o0 = o0 + wg_b[r1 + lid.y] * va1;
            o1 = o1 + wg_b[r1 + lid.y + 16u] * va1;
            o2 = o2 + wg_b[r1 + lid.y + 32u] * va1;
            o3 = o3 + wg_b[r1 + lid.y + 48u] * va1;
        }
        p0 = p0 + (e0 + o0);
        p1 = p1 + (e1 + o1);
        p2 = p2 + (e2 + o2);
        p3 = p3 + (e3 + o3);
        if (((t >> 5u) & 7u) == 7u) {
            q0 = q0 + p0; q1 = q1 + p1; q2 = q2 + p2; q3 = q3 + p3;
            p0 = vec4<f32>(0.0); p1 = vec4<f32>(0.0); p2 = vec4<f32>(0.0); p3 = vec4<f32>(0.0);
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 16>;
    pv[0u] = q0 + p0;
    pv[1u] = q1 + p1;
    pv[2u] = q2 + p2;
    pv[3u] = q3 + p3;
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x + 16u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k][j];
                }
            }
        }
    }
}
