// grid 8 8 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 8u + lx;
    let colBase = wg.x * 16u;
    let rowBase = wg.y * 16u;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 64u;
            let k = idx >> 4u;
            let j = idx & 15u;
            let t = t0 + k;
            let lr = rowBase + j;
            let gr = cfg.r_off + lr;
            let col = colBase + j;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                if (lr < cfg.r_rows && gr < d) { va = phi[t * d + gr]; }
                if (col < d) { vb = phi[t * d + col]; }
            }
            sa[idx] = va;
            sb[idx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = sa[k * 16u + ly];
            let a1 = sa[k * 16u + ly + 8u];
            let b0 = sb[k * 16u + lx];
            let b1 = sb[k * 16u + lx + 8u];
            a00 = fma(a0, b0, a00);
            a01 = fma(a0, b1, a01);
            a10 = fma(a1, b0, a10);
            a11 = fma(a1, b1, a11);
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let r0 = rowBase + ly;
    let r1 = r0 + 8u;
    let c0 = colBase + lx;
    let c1 = c0 + 8u;
    let ok0 = r0 < cfg.r_rows && (cfg.r_off + r0) < d;
    let ok1 = r1 < cfg.r_rows && (cfg.r_off + r1) < d;
    if (ok0 && c0 < d) { let i = r0 * d + c0; h[i] = h[i] + a00; }
    if (ok0 && c1 < d) { let i = r0 * d + c1; h[i] = h[i] + a01; }
    if (ok1 && c0 < d) { let i = r1 * d + c0; h[i] = h[i] + a10; }
    if (ok1 && c1 < d) { let i = r1 * d + c1; h[i] = h[i] + a11; }
}
