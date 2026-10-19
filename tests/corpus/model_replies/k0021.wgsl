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
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let colBase = wid.x * 32u;
    let rowBase = wid.y * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + k * 256u;
            let tt = e >> 5u;
            let j = e & 31u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrj = rowBase + j;
                let grj = cfg.r_off + lrj;
                if (lrj < cfg.r_rows && grj < d) {
                    va = phi[t * d + grj];
                }
                let cj = colBase + j;
                if (cj < d) {
                    vb = phi[t * d + cj];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
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

    let lrA = rowBase + ly;
    let lrB = lrA + 16u;
    let cA = colBase + lx;
    let cB = cA + 16u;
    if (lrA < cfg.r_rows) {
        if (cA < d) { h[lrA * d + cA] = h[lrA * d + cA] + a00; }
        if (cB < d) { h[lrA * d + cB] = h[lrA * d + cB] + a01; }
    }
    if (lrB < cfg.r_rows) {
        if (cA < d) { h[lrB * d + cA] = h[lrB * d + cA] + a10; }
        if (cB < d) { h[lrB * d + cB] = h[lrB * d + cB] + a11; }
    }
}
