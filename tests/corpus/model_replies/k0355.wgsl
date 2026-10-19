// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cb = gid.x * 4u;
    let rb = gid.y * 8u;
    let dm1 = cfg.d - 1u;
    let c0 = min(cb, dm1);
    let c1 = min(cb + 1u, dm1);
    let c2 = min(cb + 2u, dm1);
    let c3 = min(cb + 3u, dm1);
    var pv: array<vec4<f32>, 16>;
    var pw: array<vec4<f32>, 16>;
    var pi: array<u32, 16>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        pv[i] = vec4<f32>(0.0);
        pw[i] = vec4<f32>(0.0);
        pi[i] = min(cfg.r_off + rb + i, dm1);
    }
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 64u) {
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            pw[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 64u; k = k + 1u) {
            let t = t0 + k;
            if (t < cfg.m) {
                let base = t * cfg.d;
                let cv = vec4<f32>(phi[base + c0], phi[base + c1], phi[base + c2], phi[base + c3]);
                for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                    pw[i] = pw[i] + phi[base + pi[i]] * cv;
                }
            }
        }
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            pv[i] = pv[i] + pw[i];
        }
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = rb + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let acc = pv[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[j];
                }
            }
        }
    }
}
