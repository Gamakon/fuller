// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let row0 = wg.y * 16u;
    let col0 = wg.x * 16u;
    let lr = row0 + ly;
    let c = col0 + lx;
    // loader indices
    let lrow = row0 + lx;
    let gload = cfg.r_off + lrow;
    let aok = (lrow < cfg.r_rows) && (gload < d);
    let cload = col0 + lx;
    let bok = cload < d;

    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    var a3: f32 = 0.0;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + ly;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (aok) { va = phi[base + gload]; }
            if (bok) { vb = phi[base + cload]; }
        }
        ta[ly * 16u + lx] = va;
        tb[ly * 16u + lx] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 4u) {
            a0 = a0 + ta[k * 16u + ly] * tb[k * 16u + lx];
            a1 = a1 + ta[(k + 1u) * 16u + ly] * tb[(k + 1u) * 16u + lx];
            a2 = a2 + ta[(k + 2u) * 16u + ly] * tb[(k + 2u) * 16u + lx];
            a3 = a3 + ta[(k + 3u) * 16u + ly] * tb[(k + 3u) * 16u + lx];
        }
        workgroupBarrier();
    }

    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + ((a0 + a1) + (a2 + a3));
    }
}