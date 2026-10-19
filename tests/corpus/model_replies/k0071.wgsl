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
    let id = ty * 16u + tx;
    let rowBase = wid.y * 32u;
    let colBase = wid.x * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        // cooperative loads
        for (var s: u32 = 0u; s < 2u; s = s + 1u) {
            let e = id + s * 256u;
            let k = e >> 5u;
            let i = e & 31u;
            let t = t0 + k;
            let lr = rowBase + i;
            let gr = cfg.r_off + lr;
            let c = colBase + i;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let x0 = sa[(o + ty) & 511u];
            let x1 = sa[(o + ty + 16u) & 511u];
            let y0 = sb[(o + tx) & 511u];
            let y1 = sb[(o + tx + 16u) & 511u];
            a00 = a00 + x0 * y0;
            a01 = a01 + x0 * y1;
            a10 = a10 + x1 * y0;
            a11 = a11 + x1 * y1;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr0 = rowBase + ty;
    let lr1 = lr0 + 16u;
    let c0 = colBase + tx;
    let c1 = c0 + 16u;
    let ok0 = lr0 < cfg.r_rows && (cfg.r_off + lr0) < d;
    let ok1 = lr1 < cfg.r_rows && (cfg.r_off + lr1) < d;
    if (ok0 && c0 < d) {
        let idx = lr0 * d + c0;
        h[idx] = h[idx] + a00;
    }
    if (ok0 && c1 < d) {
        let idx = lr0 * d + c1;
        h[idx] = h[idx] + a01;
    }
    if (ok1 && c0 < d) {
        let idx = lr1 * d + c0;
        h[idx] = h[idx] + a10;
    }
    if (ok1 && c1 < d) {
        let idx = lr1 * d + c1;
        h[idx] = h[idx] + a11;
    }
}
