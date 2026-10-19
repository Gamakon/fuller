// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> colS: array<f32, 512>;
var<workgroup> rowS: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = li.x;
    let ly = li.y;
    let c0 = wg.x * 16u;
    let lr0 = wg.y * 16u;
    let c = c0 + lx;
    let lr = lr0 + ly;
    // row loaded by this thread for staging (indexed by lx)
    let lrl = lr0 + lx;
    let grl = cfg.r_off + lrl;
    let rowOk = (lrl < cfg.r_rows) && (grl < d);
    let colOk = c < d;

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let tl = ly + j * 16u;
            let t = t0 + tl;
            var cv: f32 = 0.0;
            var rv: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (colOk) { cv = phi[base + c]; }
                if (rowOk) { rv = phi[base + grl]; }
            }
            colS[tl * 16u + lx] = cv;
            rowS[tl * 16u + lx] = rv;
        }
        workgroupBarrier();
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            a0 = a0 + rowS[k * 16u + ly] * colS[k * 16u + lx];
            a1 = a1 + rowS[(k + 1u) * 16u + ly] * colS[(k + 1u) * 16u + lx];
        }
        acc = acc + a0 + a1;
        workgroupBarrier();
        t0 = t0 + 32u;
    }

    if (lr < cfg.r_rows && colOk) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
