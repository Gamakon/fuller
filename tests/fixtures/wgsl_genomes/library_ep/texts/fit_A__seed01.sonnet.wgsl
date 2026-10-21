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
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i = tid + k * 256u;
            let tt = i >> 5u;
            let j = i & 31u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = row0 + j;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[i] = va;
            sb[i] = vb;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 32u;
            let r0 = sa[o + ly];
            let r1 = sa[o + ly + 16u];
            let c0 = sb[o + lx];
            let c1 = sb[o + lx + 16u];
            a00 = fma(r0, c0, a00);
            a01 = fma(r0, c1, a01);
            a10 = fma(r1, c0, a10);
            a11 = fma(r1, c1, a11);
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr0 = row0 + ly;
    let lr1 = row0 + ly + 16u;
    let cc0 = col0 + lx;
    let cc1 = col0 + lx + 16u;
    let ok0 = lr0 < cfg.r_rows;
    let ok1 = lr1 < cfg.r_rows;
    if (ok0 && cc0 < d) {
        let idx = lr0 * d + cc0;
        h[idx] = h[idx] + a00;
    }
    if (ok0 && cc1 < d) {
        let idx = lr0 * d + cc1;
        h[idx] = h[idx] + a01;
    }
    if (ok1 && cc0 < d) {
        let idx = lr1 * d + cc0;
        h[idx] = h[idx] + a10;
    }
    if (ok1 && cc1 < d) {
        let idx = lr1 * d + cc1;
        h[idx] = h[idx] + a11;
    }
}
