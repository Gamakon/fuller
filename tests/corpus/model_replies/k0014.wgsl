// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var p = 0u; p < 4u; p = p + 1u) {
            let e = lid + p * 256u;
            let tt = e >> 6u;
            let j = e & 63u;
            let t = t0 + tt;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;
            if (t < m) {
                let lr = rb + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                let c = cb + j;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            As[e & 1023u] = av;
            Bs[e & 1023u] = bv;
        }
        workgroupBarrier();
        for (var tt = 0u; tt < 16u; tt = tt + 1u) {
            let ab = tt * 64u + ty * 4u;
            let bb = tt * 64u + tx * 4u;
            let b = vec4<f32>(Bs[bb & 1023u], Bs[(bb + 1u) & 1023u],
                              Bs[(bb + 2u) & 1023u], Bs[(bb + 3u) & 1023u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = As[(ab + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + a * b;
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rb + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let c0 = cb + tx * 4u;
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
