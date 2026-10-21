// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 256>; // 8 items x 32 rows
var<workgroup> sb: array<f32, 256>; // 8 items x 32 cols

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let lk = tid >> 5u;
    let lj = tid & 31u;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;

    let ldCol = colBase + lj;
    let ldLr = rowBase + lj;
    let ldGr = cfg.r_off + ldLr;
    let colOk = ldCol < d;
    let rowOk = (ldLr < cfg.r_rows) && (ldGr < d);

    var e0 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var o0 = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + lk;
        var av: f32 = 0.0;
        var bv: f32 = 0.0;
        if (t < m) {
            if (rowOk) { av = phi[t * d + ldGr]; }
            if (colOk) { bv = phi[t * d + ldCol]; }
        }
        sa[tid] = av;
        sb[tid] = bv;
        workgroupBarrier();

        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let p = (2u * k) * 32u;
            let q = (2u * k + 1u) * 32u;
            let a0 = sa[p + lid.y];
            let a1 = sa[p + lid.y + 16u];
            let b0 = sb[p + lid.x];
            let b1 = sb[p + lid.x + 16u];
            e0[0] = fma(a0, b0, e0[0]);
            e0[1] = fma(a0, b1, e0[1]);
            e0[2] = fma(a1, b0, e0[2]);
            e0[3] = fma(a1, b1, e0[3]);
            let c0 = sa[q + lid.y];
            let c1 = sa[q + lid.y + 16u];
            let d0 = sb[q + lid.x];
            let d1 = sb[q + lid.x + 16u];
            o0[0] = fma(c0, d0, o0[0]);
            o0[1] = fma(c0, d1, o0[1]);
            o0[2] = fma(c1, d0, o0[2]);
            o0[3] = fma(c1, d1, o0[3]);
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
        let lr = rowBase + lid.y + 16u * j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 2u; i = i + 1u) {
                let c = colBase + lid.x + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + (e0[j * 2u + i] + o0[j * 2u + i]);
                }
            }
        }
    }
}
