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
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + k * 256u;
            let t = e >> 5u;
            let j = e & 31u;
            let tt = t0 + t;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (tt < m) {
                let lrw = row0 + j;
                if (lrw < cfg.r_rows && cfg.r_off + lrw < d) {
                    va = phi[tt * d + cfg.r_off + lrw];
                }
                let cc = col0 + j;
                if (cc < d) {
                    vb = phi[tt * d + cc];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var t: u32 = 0u; t < 16u; t = t + 1u) {
            let o = t * 32u;
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

    let rA = row0 + ly;
    let rB = rA + 16u;
    let cA = col0 + lx;
    let cB = cA + 16u;
    if (rA < cfg.r_rows) {
        if (cA < d) {
            let i = rA * d + cA;
            h[i] = h[i] + a00;
        }
        if (cB < d) {
            let i = rA * d + cB;
            h[i] = h[i] + a01;
        }
    }
    if (rB < cfg.r_rows) {
        if (cA < d) {
            let i = rB * d + cA;
            h[i] = h[i] + a10;
        }
        if (cB < d) {
            let i = rB * d + cB;
            h[i] = h[i] + a11;
        }
    }
}
