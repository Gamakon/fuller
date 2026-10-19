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
    let lin = ly * 16u + lx;
    let rowBase = wid.y * 32u;
    let colBase = wid.x * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i = lin + k * 256u;
            let j = i >> 5u;
            let q = i & 31u;
            let t = t0 + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrq = rowBase + q;
                let grq = cfg.r_off + lrq;
                if (lrq < cfg.r_rows && grq < d) {
                    va = phi[t * d + grq];
                }
                let cq = colBase + q;
                if (cq < d) {
                    vb = phi[t * d + cq];
                }
            }
            sa[i & 511u] = va;
            sb[i & 511u] = vb;
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let o = j * 32u;
            let x0 = sa[(o + ly) & 511u];
            let x1 = sa[(o + ly + 16u) & 511u];
            let y0 = sb[(o + lx) & 511u];
            let y1 = sb[(o + lx + 16u) & 511u];
            a00 = a00 + x0 * y0;
            a01 = a01 + x0 * y1;
            a10 = a10 + x1 * y0;
            a11 = a11 + x1 * y1;
        }
        workgroupBarrier();
    }

    let lr0 = rowBase + ly;
    let lr1 = lr0 + 16u;
    let c0 = colBase + lx;
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
