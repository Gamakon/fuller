// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var accA: array<f32, 16>;
    var accB: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        accA[q & 15u] = 0.0;
        accB[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let idx = lin + s * 256u;
            let k = idx >> 6u;
            let cc = idx & 63u;
            let t = t0 + k;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let cg = colBase + cc;
                if (cg < d) { vb = phi[t * d + cg]; }
                let lrow = rowBase + cc;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 2u) {
            let b0 = kk * 64u;
            let b1 = b0 + 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a0 = sa[(b0 + ty * 4u + i) & 511u];
                let a1 = sa[(b1 + ty * 4u + i) & 511u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c0 = sb[(b0 + tx * 4u + j) & 511u];
                    let c1 = sb[(b1 + tx * 4u + j) & 511u];
                    let q = (i * 4u + j) & 15u;
                    accA[q] = fma(a0, c0, accA[q]);
                    accB[q] = fma(a1, c1, accB[q]);
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + tx * 4u + j;
                if (c < d) {
                    let q = (i * 4u + j) & 15u;
                    let ix = lr * d + c;
                    h[ix] = h[ix] + (accA[q] + accB[q]);
                }
            }
        }
    }
}
