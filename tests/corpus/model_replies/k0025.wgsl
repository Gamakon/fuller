// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;

    var e0: array<f32, 16>;
    var o0: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        e0[q & 15u] = 0.0;
        o0[q & 15u] = 0.0;
    }

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lid + q * 256u;
            let k = e >> 6u;
            let cc = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrw = rowBase + cc;
                let gr = cfg.r_off + lrw;
                if (lrw < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let col = colBase + cc;
                if (col < d) {
                    vb = phi[t * d + col];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let k0 = (kk * 2u) * 64u;
            let k1 = k0 + 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a0 = sa[(k0 + ty + i * 16u) & 1023u];
                let a1 = sa[(k1 + ty + i * 16u) & 1023u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b0 = sb[(k0 + tx + j * 16u) & 1023u];
                    let b1 = sb[(k1 + tx + j * 16u) & 1023u];
                    let ix = (i * 4u + j) & 15u;
                    e0[ix] = e0[ix] + a0 * b0;
                    o0[ix] = o0[ix] + a1 * b1;
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty + i * 16u;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx + j * 16u;
                if (c < d) {
                    let ix = (i * 4u + j) & 15u;
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (e0[ix] + o0[ix]);
                }
            }
        }
    }
}
