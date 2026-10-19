// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var e: u32 = 0u; e < 2u; e = e + 1u) {
            let idx = tid + e * 256u;
            let k = idx >> 5u;
            let j = idx & 31u;
            let t = t0 + k;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            let c = colBase + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
                if (c < d) { vb = phi[t * d + c]; }
            }
            sa[idx & 511u] = va;
            sb[idx & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 32u;
            let r0 = sa[(o + ly) & 511u];
            let r1 = sa[(o + ly + 16u) & 511u];
            let c0 = sb[(o + lx) & 511u];
            let c1 = sb[(o + lx + 16u) & 511u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr0 = rowBase + ly;
    let lr1 = lr0 + 16u;
    let cc0 = colBase + lx;
    let cc1 = cc0 + 16u;
    let ok0 = lr0 < cfg.r_rows && (cfg.r_off + lr0) < d;
    let ok1 = lr1 < cfg.r_rows && (cfg.r_off + lr1) < d;
    if (ok0 && cc0 < d) { let i = lr0 * d + cc0; h[i] = h[i] + a00; }
    if (ok0 && cc1 < d) { let i = lr0 * d + cc1; h[i] = h[i] + a01; }
    if (ok1 && cc0 < d) { let i = lr1 * d + cc0; h[i] = h[i] + a10; }
    if (ok1 && cc1 < d) { let i = lr1 * d + cc1; h[i] = h[i] + a11; }
}
