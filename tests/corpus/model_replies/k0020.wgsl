// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = li.x;
    let ly = li.y;
    let tid = ly * 16u + lx;
    let lr0 = wg.y * 16u;
    let c0 = wg.x * 16u;
    let j = tid & 15u;
    let tt = tid >> 4u;
    let rowj = lr0 + j;
    let grj = cfg.r_off + rowj;
    let colj = c0 + j;
    let rvalid = rowj < cfg.r_rows && grj < d;
    let cvalid = colj < d;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 1u; k = k + 1u) {
            let item = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let base = item * d;
                if (rvalid) { va = phi[base + grj]; }
                if (cvalid) { vb = phi[base + colj]; }
            }
            sa[(tid) & 511u] = va;
            sb[(tid) & 511u] = vb;
        }
        workgroupBarrier();
        for (var t: u32 = 0u; t < 16u; t = t + 2u) {
            acc0 = acc0 + sa[(t * 16u + ly) & 511u] * sb[(t * 16u + lx) & 511u];
            acc1 = acc1 + sa[((t + 1u) * 16u + ly) & 511u] * sb[((t + 1u) * 16u + lx) & 511u];
        }
        workgroupBarrier();
    }
    let lr = lr0 + ly;
    let c = c0 + lx;
    if (lr < cfg.r_rows && c < d && cfg.r_off + lr < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + (acc0 + acc1);
    }
}
