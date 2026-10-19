// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc0: array<f32, 16>;
    var acc1: array<f32, 16>;
    for (var q = 0u; q < 16u; q = q + 1u) {
        acc0[q & 15u] = 0.0;
        acc1[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var q = 0u; q < 4u; q = q + 1u) {
            let e = lid + q * 256u;
            let k = e >> 6u;
            let cc = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                let lrw = rowBase + cc;
                let grw = cfg.r_off + lrw;
                if (lrw < cfg.r_rows && grw < d) {
                    va = phi[base + grw];
                }
                let cw = colBase + cc;
                if (cw < d) {
                    vb = phi[base + cw];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 2u) {
            var ra: array<f32, 4>;
            var rb: array<f32, 4>;
            let o0 = k * 64u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                ra[i & 3u] = sa[(o0 + ty + i * 16u) & 1023u];
                rb[i & 3u] = sb[(o0 + tx + i * 16u) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let ix = (i * 4u + j) & 15u;
                    acc0[ix] = acc0[ix] + ra[i & 3u] * rb[j & 3u];
                }
            }
            let o1 = o0 + 64u;
            for (var i = 0u; i < 4u; i = i + 1u) {
                ra[i & 3u] = sa[(o1 + ty + i * 16u) & 1023u];
                rb[i & 3u] = sb[(o1 + tx + i * 16u) & 1023u];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let ix = (i * 4u + j) & 15u;
                    acc1[ix] = acc1[ix] + ra[i & 3u] * rb[j & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty + i * 16u;
        if (lr < cfg.r_rows) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + j * 16u;
                if (c < d) {
                    let ix = (i * 4u + j) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (acc0[ix] + acc1[ix]);
                }
            }
        }
    }
}
