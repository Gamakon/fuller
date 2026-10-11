// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ct: array<f32, 512>;
var<workgroup> rt: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let c = wg.x * 16u + lx;
    let lr = wg.y * 16u + ly;
    // row index loaded by this thread (indexed by lx) for the row tile
    let lrl = wg.y * 16u + lx;
    let grl = cfg.r_off + lrl;
    let cOk = c < d;
    let rOk = (lrl < cfg.r_rows) && (grl < d);
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
                if (cOk) { cv = phi[base + c]; }
                if (rOk) { rv = phi[base + grl]; }
            }
            ct[tl * 16u + lx] = cv;
            rt[tl * 16u + lx] = rv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = acc + rt[k * 16u + ly] * ct[k * 16u + lx];
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
