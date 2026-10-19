// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let lin = ty * 16u + tx;
    let row_base = wid.y * 32u;
    let col_base = wid.x * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // cooperative load: 512 entries per array, 2 per thread
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = lin + k * 256u;
            let tt = t0 + (e >> 5u);
            let j = e & 31u;
            let lrw = row_base + j;
            let gr = cfg.r_off + lrw;
            let cc = col_base + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tt < m) {
                if (lrw < cfg.r_rows && gr < d) {
                    va = phi[tt * d + gr];
                }
                if (cc < d) {
                    vb = phi[tt * d + cc];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let r0 = sa[(o + ty) & 511u];
            let r1 = sa[(o + ty + 16u) & 511u];
            let c0 = sb[(o + tx) & 511u];
            let c1 = sb[(o + tx + 16u) & 511u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr0 = row_base + ty;
    let lr1 = lr0 + 16u;
    let cA = col_base + tx;
    let cB = cA + 16u;
    if (lr0 < cfg.r_rows) {
        if (cA < d) {
            let i = lr0 * d + cA;
            h[i] = h[i] + a00;
        }
        if (cB < d) {
            let i = lr0 * d + cB;
            h[i] = h[i] + a01;
        }
    }
    if (lr1 < cfg.r_rows) {
        if (cA < d) {
            let i = lr1 * d + cA;
            h[i] = h[i] + a10;
        }
        if (cB < d) {
            let i = lr1 * d + cB;
            h[i] = h[i] + a11;
        }
    }
}
