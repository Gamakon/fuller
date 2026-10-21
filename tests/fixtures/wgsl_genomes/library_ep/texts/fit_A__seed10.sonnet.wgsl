// grid 16 16 1 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let row0 = wg.y * 16u;
    let col0 = wg.x * 16u;
    let lrL = row0 + lx;            // local row loaded by this thread
    let grL = cfg.r_off + lrL;
    let cL = col0 + lx;
    let okA = (lrL < cfg.r_rows) && (grL < d);
    let okB = cL < d;

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var h2: u32 = 0u; h2 < 2u; h2 = h2 + 1u) {
            let tt = ly + h2 * 16u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (okA) { va = phi[base + grL]; }
                if (okB) { vb = phi[base + cL]; }
            }
            sa[tt * 16u + lx] = va;
            sb[tt * 16u + lx] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            acc = fma(sa[k * 16u + ly], sb[k * 16u + lx], acc);
        }
        workgroupBarrier();
        t0 = t0 + 32u;
    }

    let lr = row0 + ly;
    let c = col0 + lx;
    if (lr < cfg.r_rows && c < d && (cfg.r_off + lr) < d) {
        let idx = lr * d + c;
        h[idx] = h[idx] + acc;
    }
}
