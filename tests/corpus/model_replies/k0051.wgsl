// grid 16 16 2 2
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let cx0 = wid.x * 32u;
    let ry0 = wid.y * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + k * 256u;
            let tt = e >> 5u;
            let j = e & 31u;
            let t = t0 + tt;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (t < m) {
                let col = cx0 + j;
                if (col < d) {
                    vb = phi[t * d + col];
                }
                let lr = ry0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sb[e & 511u] = vb;
            sa[e & 511u] = va;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 32u;
            let r0 = sa[(o + ly) & 511u];
            let r1 = sa[(o + ly + 16u) & 511u];
            let c0 = sb[(o + lx) & 511u];
            let c1 = sb[(o + lx + 16u) & 511u];
            a00 = fma(r0, c0, a00);
            a01 = fma(r0, c1, a01);
            a10 = fma(r1, c0, a10);
            a11 = fma(r1, c1, a11);
        }
        workgroupBarrier();
    }

    let lrA = ry0 + ly;
    let lrB = lrA + 16u;
    let cA = cx0 + lx;
    let cB = cA + 16u;
    let okRA = lrA < cfg.r_rows && (cfg.r_off + lrA) < d;
    let okRB = lrB < cfg.r_rows && (cfg.r_off + lrB) < d;
    let okCA = cA < d;
    let okCB = cB < d;
    if (okRA && okCA) {
        let i = lrA * d + cA;
        h[i] = h[i] + a00;
    }
    if (okRA && okCB) {
        let i = lrA * d + cB;
        h[i] = h[i] + a01;
    }
    if (okRB && okCA) {
        let i = lrB * d + cA;
        h[i] = h[i] + a10;
    }
    if (okRB && okCB) {
        let i = lrB * d + cB;
        h[i] = h[i] + a11;
    }
}
