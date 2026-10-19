// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
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
    let li = ly * 16u + lx;
    let colbase = wid.x * 32u;
    let rowbase = wid.y * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = li + 256u * k;
            let tt = e >> 5u;
            let j = e & 31u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = rowbase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colbase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
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
        t0 = t0 + 16u;
    }

    let lrA = rowbase + ly;
    let lrB = rowbase + ly + 16u;
    let cA = colbase + lx;
    let cB = colbase + lx + 16u;
    let okA = lrA < cfg.r_rows && (cfg.r_off + lrA) < d;
    let okB = lrB < cfg.r_rows && (cfg.r_off + lrB) < d;
    if (okA && cA < d) {
        let i = lrA * d + cA;
        h[i] = h[i] + a00;
    }
    if (okA && cB < d) {
        let i = lrA * d + cB;
        h[i] = h[i] + a01;
    }
    if (okB && cA < d) {
        let i = lrB * d + cA;
        h[i] = h[i] + a10;
    }
    if (okB && cB < d) {
        let i = lrB * d + cB;
        h[i] = h[i] + a11;
    }
}
