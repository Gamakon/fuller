// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 512>;
var<workgroup> sB: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lid + k * 256u;
            let it = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + it;
            let col = colBase + cc;
            let lrow = rowBase + cc;
            let grow = cfg.r_off + lrow;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (col < d) { vb = phi[t * d + col]; }
                if (lrow < cfg.r_rows && grow < d) { va = phi[t * d + grow]; }
            }
            sA[idx & 511u] = va;
            sB[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var it: u32 = 0u; it < 8u; it = it + 1u) {
            let o = it * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sA[(o + ty * 4u + i) & 511u];
                b[i & 3u] = sB[(o + tx * 4u + i) & 511u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + a[i & 3u] * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
