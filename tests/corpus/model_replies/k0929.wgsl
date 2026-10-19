// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let col0 = wid.x * 16u;
    let row0 = wid.y * 16u;
    let c = col0 + lx;
    let lr = row0 + ly;
    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let e = tid + j * 256u;
            let tt = e >> 4u;
            let cc = e & 15u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrr = row0 + cc;
                let grr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && grr < d) {
                    va = phi[t * d + grr];
                }
                let cl = col0 + cc;
                if (cl < d) {
                    vb = phi[t * d + cl];
                }
            }
            sa[e & 511u] = va;
            sb[e & 511u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = acc + sa[(k * 16u + ly) & 511u] * sb[(k * 16u + lx) & 511u];
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }
    if (lr < cfg.r_rows && c < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
