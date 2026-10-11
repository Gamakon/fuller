// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let c = wid.x * 16u + lx;
    let lr = wid.y * 16u + ly;
    let gr_me = cfg.r_off + lr;
    // row used for the A-tile load by this thread (indexed by lx)
    let lrA = wid.y * 16u + lx;
    let grA = cfg.r_off + lrA;
    let validA = (lrA < cfg.r_rows) && (grA < d);
    let validB = c < d;
    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        let t = t0 + ly;
        var a: f32 = 0.0;
        var b: f32 = 0.0;
        if (t < m) {
            let base = t * d;
            if (validA) { a = phi[base + grA]; }
            if (validB) { b = phi[base + c]; }
        }
        ta[ly * 16u + lx] = a;
        tb[ly * 16u + lx] = b;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            acc = fma(ta[k * 16u + ly], tb[k * 16u + lx], acc);
        }
        workgroupBarrier();
        t0 = t0 + 16u;
    }
    if (lr < cfg.r_rows && gr_me < d && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
