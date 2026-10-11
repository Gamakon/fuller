// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> rowT: array<f32, 256>;
var<workgroup> colT: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let c0 = wid.x * 16u;
    let lr0 = wid.y * 16u;
    let c = c0 + lx;
    let lr = lr0 + ly;
    let gr = cfg.r_off + lr;

    // loader coordinates: this thread loads column index (c0+lx) and row index (lr0+lx) for item t0+ly
    let lc = c0 + lx;
    let lrl = lr0 + lx;
    let lgr = cfg.r_off + lrl;
    let colOk = lc < d;
    let rowOk = (lrl < cfg.r_rows) && (lgr < d);

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ly;
        var cv: f32 = 0.0;
        var rv: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (colOk) { cv = phi[base + lc]; }
            if (rowOk) { rv = phi[base + lgr]; }
        }
        colT[ly * 16u + lx] = cv;
        rowT[ly * 16u + lx] = rv;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + rowT[k * 16u + ly] * colT[k * 16u + lx];
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }

    if (lr < cfg.r_rows && c < d && gr < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
