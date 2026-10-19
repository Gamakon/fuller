// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let flat = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var acc0: array<f32, 16>;
    var acc1: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        acc0[i & 15u] = 0.0;
        acc1[i & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    while (t0 < m) {
        // cooperative load: 512 elements each for rows and columns
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = flat + k * 256u;
            let ti = e >> 6u;
            let j = e & 63u;
            let t = t0 + ti;
            let c = colBase + j;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                if (c < d) { vb = phi[t * d + c]; }
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sb[e & 511u] = vb;
            sa[e & 511u] = va;
        }
        workgroupBarrier();

        for (var ti: u32 = 0u; ti < 8u; ti = ti + 2u) {
            var b0: array<f32, 4>;
            var a0: array<f32, 4>;
            var b1: array<f32, 4>;
            var a1: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                b0[i & 3u] = sb[(ti * 64u + tx + 16u * i) & 511u];
                a0[i & 3u] = sa[(ti * 64u + ty + 16u * i) & 511u];
                b1[i & 3u] = sb[((ti + 1u) * 64u + tx + 16u * i) & 511u];
                a1[i & 3u] = sa[((ti + 1u) * 64u + ty + 16u * i) & 511u];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let q = (j * 4u + i) & 15u;
                    acc0[q] = acc0[q] + a0[j & 3u] * b0[i & 3u];
                    acc1[q] = acc1[q] + a1[j & 3u] * b1[i & 3u];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + ty + 16u * j;
        if (lr < cfg.r_rows) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colBase + tx + 16u * i;
                if (c < d) {
                    let q = (j * 4u + i) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (acc0[q] + acc1[q]);
                }
            }
        }
    }
}
