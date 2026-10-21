// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let row0 = wid.y * 16u;
    let col0 = wid.x * 16u;

    // row/col this thread loads into shared memory
    let lrl = row0 + lx;              // local row index for the A tile
    let grl = cfg.r_off + lrl;        // global row
    let rowOk = (lrl < cfg.r_rows) && (grl < d);
    let cl = col0 + lx;
    let colOk = cl < d;

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ly;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (rowOk) { va = phi[base + grl]; }
            if (colOk) { vb = phi[base + cl]; }
        }
        sa[ly * 16u + lx] = va;
        sb[ly * 16u + lx] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + sa[k * 16u + ly] * sb[k * 16u + lx];
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    let lr = row0 + ly;
    let c = col0 + lx;
    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
