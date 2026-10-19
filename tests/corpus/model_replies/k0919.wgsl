// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let lx = li.x;
    let ly = li.y;
    let d = cfg.d;
    let m = cfg.m;
    let lr = wg.y * 16u + ly;
    let c = wg.x * 16u + lx;

    // loader indices
    let ld_row = wg.y * 16u + lx;
    let ld_col = wg.x * 16u + lx;
    let row_ok = (ld_row < cfg.r_rows) && ((cfg.r_off + ld_row) < d);
    let col_ok = ld_col < d;
    let ld_gr = cfg.r_off + ld_row;

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t1 = t0 + ly;
        let t2 = t0 + ly + 16u;
        var a1: f32 = 0.0;
        var b1: f32 = 0.0;
        var a2: f32 = 0.0;
        var b2: f32 = 0.0;
        if (t1 < m) {
            let base = t1 * d;
            if (row_ok) { a1 = phi[base + ld_gr]; }
            if (col_ok) { b1 = phi[base + ld_col]; }
        }
        if (t2 < m) {
            let base = t2 * d;
            if (row_ok) { a2 = phi[base + ld_gr]; }
            if (col_ok) { b2 = phi[base + ld_col]; }
        }
        sa[(ly * 16u + lx) & 511u] = a1;
        sb[(ly * 16u + lx) & 511u] = b1;
        sa[((ly + 16u) * 16u + lx) & 511u] = a2;
        sb[((ly + 16u) * 16u + lx) & 511u] = b2;
        workgroupBarrier();
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        for (var k: u32 = 0u; k < 32u; k = k + 2u) {
            s0 = s0 + sa[(k * 16u + ly) & 511u] * sb[(k * 16u + lx) & 511u];
            s1 = s1 + sa[((k + 1u) * 16u + ly) & 511u] * sb[((k + 1u) * 16u + lx) & 511u];
        }
        acc = acc + (s0 + s1);
        workgroupBarrier();
        t0 = t0 + 32u;
    }

    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
