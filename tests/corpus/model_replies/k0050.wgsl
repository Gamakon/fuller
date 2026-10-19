// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wa: array<f32, 256>;
var<workgroup> wb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let lr0 = wid.y * 16u;
    let c0 = wid.x * 16u;
    // load-role indices: thread loads item ly, row/col offset lx
    let ldr = lr0 + lx;
    let ldg = cfg.r_off + ldr;
    let ldc = c0 + lx;
    let okr = (ldr < cfg.r_rows) && (ldg < d);
    let okc = ldc < d;
    var acc: f32 = 0.0;
    let slot = (ly * 16u + lx) & 255u;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ly;
        var va: f32 = 0.0;
        var vb: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (okr) { va = phi[base + ldg]; }
            if (okc) { vb = phi[base + ldc]; }
        }
        wa[slot] = va;
        wb[slot] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = acc + wa[(k * 16u + ly) & 255u] * wb[(k * 16u + lx) & 255u];
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    let lr = lr0 + ly;
    let c = c0 + lx;
    if (lr < cfg.r_rows && (cfg.r_off + lr) < d && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
