// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;

fn put(lr: u32, c: u32, v: f32) {
    if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
        let idx = lr * cfg.d + c;
        h[idx] = h[idx] + v;
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    var r0 = vec4<f32>(0.0);
    var r1 = vec4<f32>(0.0);
    var r2 = vec4<f32>(0.0);
    var r3 = vec4<f32>(0.0);

    // initial tile into buffer 0
    for (var k = 0u; k < 2u; k = k + 1u) {
        let idx = lidx + k * 256u;
        let it = idx >> 6u;
        let col = idx & 63u;
        var fa = 0.0;
        var fb = 0.0;
        if (it < m) {
            let cb = wid.x * 64u + col;
            if (cb < d) { fb = phi[it * d + cb]; }
            let rl = wid.y * 64u + col;
            if (rl < cfg.r_rows && (cfg.r_off + rl) < d) { fa = phi[it * d + cfg.r_off + rl]; }
        }
        wg_a[idx & 1023u] = fa;
        wg_b[idx & 1023u] = fb;
    }
    workgroupBarrier();

    var buf = 0u;
    for (var t = 0u; t < m; t = t + 8u) {
        let nbuf = buf ^ 512u;
        let tn = t + 8u;
        for (var k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = idx >> 6u;
            let col = idx & 63u;
            var fa = 0.0;
            var fb = 0.0;
            if (tn + it < m) {
                let cb = wid.x * 64u + col;
                if (cb < d) { fb = phi[(tn + it) * d + cb]; }
                let rl = wid.y * 64u + col;
                if (rl < cfg.r_rows && (cfg.r_off + rl) < d) { fa = phi[(tn + it) * d + cfg.r_off + rl]; }
            }
            wg_a[(nbuf + idx) & 1023u] = fa;
            wg_b[(nbuf + idx) & 1023u] = fb;
        }
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            let ao = buf + kk * 64u + lid.y * 4u;
            let bo = buf + kk * 64u + lid.x * 4u;
            let a0 = wg_a[ao & 1023u];
            let a1 = wg_a[(ao + 1u) & 1023u];
            let a2 = wg_a[(ao + 2u) & 1023u];
            let a3 = wg_a[(ao + 3u) & 1023u];
            let bv = vec4<f32>(wg_b[bo & 1023u], wg_b[(bo + 1u) & 1023u], wg_b[(bo + 2u) & 1023u], wg_b[(bo + 3u) & 1023u]);
            r0 = fma(vec4<f32>(a0), bv, r0);
            r1 = fma(vec4<f32>(a1), bv, r1);
            r2 = fma(vec4<f32>(a2), bv, r2);
            r3 = fma(vec4<f32>(a3), bv, r3);
        }
        buf = nbuf;
        workgroupBarrier();
    }

    let lr0 = wid.y * 64u + lid.y * 4u;
    let c0 = wid.x * 64u + lid.x * 4u;
    put(lr0, c0, r0.x); put(lr0, c0 + 1u, r0.y); put(lr0, c0 + 2u, r0.z); put(lr0, c0 + 3u, r0.w);
    put(lr0 + 1u, c0, r1.x); put(lr0 + 1u, c0 + 1u, r1.y); put(lr0 + 1u, c0 + 2u, r1.z); put(lr0 + 1u, c0 + 3u, r1.w);
    put(lr0 + 2u, c0, r2.x); put(lr0 + 2u, c0 + 1u, r2.y); put(lr0 + 2u, c0 + 2u, r2.z); put(lr0 + 2u, c0 + 3u, r2.w);
    put(lr0 + 3u, c0, r3.x); put(lr0 + 3u, c0 + 1u, r3.y); put(lr0 + 3u, c0 + 2u, r3.z); put(lr0 + 3u, c0 + 3u, r3.w);
}
