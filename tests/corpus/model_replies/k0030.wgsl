// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let row0 = wid.y * 16u;
    let col0 = wid.x * 16u;
    let lr = row0 + ly;
    let c = col0 + lx;

    var acc: f32 = 0.0;

    // loader mapping: two slots per thread
    let j = li & 15u;
    let k0 = li >> 4u;           // 0..15
    let k1 = k0 + 16u;           // 16..31
    let lrow = row0 + j;
    let grow = cfg.r_off + lrow;
    let rowok = (lrow < cfg.r_rows) && (grow < d);
    let lcol = col0 + j;
    let colok = lcol < d;

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let ta = t0 + k0;
        let tb = t0 + k1;
        var a0: f32 = 0.0;
        var a1: f32 = 0.0;
        var b0: f32 = 0.0;
        var b1: f32 = 0.0;
        if (ta < m) {
            let base = ta * d;
            if (rowok) { a0 = phi[base + grow]; }
            if (colok) { b0 = phi[base + lcol]; }
        }
        if (tb < m) {
            let base = tb * d;
            if (rowok) { a1 = phi[base + grow]; }
            if (colok) { b1 = phi[base + lcol]; }
        }
        sa[(k0 * 16u + j) & 511u] = a0;
        sa[(k1 * 16u + j) & 511u] = a1;
        sb[(k0 * 16u + j) & 511u] = b0;
        sb[(k1 * 16u + j) & 511u] = b1;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = fma(sa[(k * 16u + ly) & 511u], sb[(k * 16u + lx) & 511u], acc);
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }

    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
