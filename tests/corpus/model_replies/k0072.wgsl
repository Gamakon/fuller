// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrw = rbase + j;
                let gr = cfg.r_off + lrw;
                if (lrw < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = cbase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ao = k * 64u + ty * 4u;
            let bo = k * 64u + tx * 4u;
            let b = vec4<f32>(sb[bo & 1023u], sb[(bo + 1u) & 1023u],
                              sb[(bo + 2u) & 1023u], sb[(bo + 3u) & 1023u]);
            let x0 = sa[ao & 1023u];
            let x1 = sa[(ao + 1u) & 1023u];
            let x2 = sa[(ao + 2u) & 1023u];
            let x3 = sa[(ao + 3u) & 1023u];
            a0 = a0 + x0 * b;
            a1 = a1 + x1 * b;
            a2 = a2 + x2 * b;
            a3 = a3 + x3 * b;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let c0 = cbase + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var v = a0;
            if (i == 1u) { v = a1; }
            if (i == 2u) { v = a2; }
            if (i == 3u) { v = a3; }
            let rb = lr * d + c0;
            if (c0 < d) { h[rb] = h[rb] + v.x; }
            if (c0 + 1u < d) { h[rb + 1u] = h[rb + 1u] + v.y; }
            if (c0 + 2u < d) { h[rb + 2u] = h[rb + 2u] + v.z; }
            if (c0 + 3u < d) { h[rb + 3u] = h[rb + 3u] + v.w; }
        }
    }
}
