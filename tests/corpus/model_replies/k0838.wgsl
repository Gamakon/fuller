// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sc: array<f32, 256>;
var<workgroup> sr: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 8u + lx;
    let cb = wid.x * 32u;
    let rb = wid.y * 32u;
    let d = cfg.d;
    let m = cfg.m;

    var accs: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        accs[i] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = tid + j * 64u;
            let tt = idx >> 6u;
            let k = idx & 63u;
            let item = t0 + tt;
            if (k < 32u) {
                let col = cb + k;
                var v = 0.0;
                if (item < m && col < d) {
                    v = phi[item * d + col];
                }
                sc[tt * 32u + k] = v;
            } else {
                let kk = k - 32u;
                let lr = rb + kk;
                let gr = cfg.r_off + lr;
                var v = 0.0;
                if (item < m && lr < cfg.r_rows && gr < d) {
                    v = phi[item * d + gr];
                }
                sr[tt * 32u + kk] = v;
            }
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 8u; tt = tt + 1u) {
            let bo = tt * 32u + lx * 4u;
            let ao = tt * 32u + ly * 4u;
            let b = vec4<f32>(sc[bo], sc[bo + 1u], sc[bo + 2u], sc[bo + 3u]);
            accs[0] = accs[0] + sr[ao] * b;
            accs[1] = accs[1] + sr[ao + 1u] * b;
            accs[2] = accs[2] + sr[ao + 2u] * b;
            accs[3] = accs[3] + sr[ao + 3u] * b;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rb + ly * 4u + i;
        if (lr < cfg.r_rows) {
            let a = accs[i];
            let c0 = cb + lx * 4u;
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + a.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + a.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + a.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + a.w; }
        }
    }
}