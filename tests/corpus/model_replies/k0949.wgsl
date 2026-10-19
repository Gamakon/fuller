// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let lr = wg.y * 16u + ly;
    let c = wg.x * 16u + lx;
    // load-side indices
    let lrl = wg.y * 16u + lx;
    let grl = cfg.r_off + lrl;
    let rowOk = (lrl < cfg.r_rows) && (grl < d);
    let colOk = c < d;
    let sidx = ly * 16u + lx;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ly;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (rowOk) { va = phi[base + grl]; }
            if (colOk) { vb = phi[base + c]; }
        }
        sa[sidx & 255u] = va;
        sb[sidx & 255u] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 2u) {
            acc0 = acc0 + sa[(k * 16u + ly) & 255u] * sb[(k * 16u + lx) & 255u];
            acc1 = acc1 + sa[((k + 1u) * 16u + ly) & 255u] * sb[((k + 1u) * 16u + lx) & 255u];
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + (acc0 + acc1);
    }
}
