// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    // row/column this thread loads for (indexed by lx)
    let ld_lr = wg.y * 16u + lx;
    let ld_gr = cfg.r_off + ld_lr;
    let ld_c = wg.x * 16u + lx;
    let a_ok = (ld_lr < cfg.r_rows) && (ld_gr < d);
    let b_ok = ld_c < d;

    let my_lr = wg.y * 16u + ly;
    let my_gr = cfg.r_off + my_lr;
    let my_c = wg.x * 16u + lx;

    var acc: f32 = 0.0;
    var tt: u32 = 0u;
    loop {
        if (tt >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let tl = ly + 16u * k;
            let t = tt + tl;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (a_ok) { va = phi[base + ld_gr]; }
                if (b_ok) { vb = phi[base + ld_c]; }
            }
            let s = tl * 16u + lx;
            sa[s & 511u] = va;
            sb[s & 511u] = vb;
        }
        workgroupBarrier();
        for (var j: u32 = 0u; j < 32u; j = j + 1u) {
            acc = fma(sa[(j * 16u + ly) & 511u], sb[(j * 16u + lx) & 511u], acc);
        }
        workgroupBarrier();
        tt = tt + 32u;
    }

    if (my_lr < cfg.r_rows && my_gr < d && my_c < d) {
        let idx = my_lr * d + my_c;
        h[idx] = h[idx] + acc;
    }
}
