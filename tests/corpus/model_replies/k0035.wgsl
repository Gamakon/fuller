// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    let d = cfg.d;
    let m = cfg.m;

    var a0: array<f32, 16>;
    var a1: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        a0[q & 15u] = 0.0;
        a1[q & 15u] = 0.0;
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + 256u * k;
            let ti = e >> 6u;
            let cc = e & 63u;
            let t = t0 + ti;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrr = rowBase + cc;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var p: u32 = 0u; p < 8u; p = p + 1u) {
            for (var s: u32 = 0u; s < 2u; s = s + 1u) {
                let ti = p * 2u + s;
                let o = ti * 64u;
                var rv: array<f32, 4>;
                var cv: array<f32, 4>;
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    rv[i & 3u] = sa[(o + ty + 16u * i) & 1023u];
                    cv[i & 3u] = sb[(o + tx + 16u * i) & 1023u];
                }
                if (s == 0u) {
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                            let q = (j * 4u + i) & 15u;
                            a0[q] = a0[q] + rv[j & 3u] * cv[i & 3u];
                        }
                    }
                } else {
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                            let q = (j * 4u + i) & 15u;
                            a1[q] = a1[q] + rv[j & 3u] * cv[i & 3u];
                        }
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ty + 16u * j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colBase + tx + 16u * i;
                if (c < d) {
                    let q = (j * 4u + i) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (a0[q] + a1[q]);
                }
            }
        }
    }
}
